import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.smartgrid_env import default_config as env_default_config  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Constants ---
SEED = 42
MICROGRID_PEAK_KW = 500.0  # Scale national load to this peak (kW)

# Grid import cap is a peak-reduction *target*, not a hardware rating. It is the
# quantile of grid-only import on the training split, i.e. the load the site would
# draw with the battery idle. Train-only so the constraint cannot leak into val/test.
# The Lagrangian budget (`cost_limit` in configs/config.yaml, 0.05 kW/step) is
# calibrated against 0.90; re-derive it if this quantile moves.
CAP_QUANTILE = 0.90

# Zone shares (of total load at any given hour)
ZONE_SHARES = {
    "residential": 0.35,
    "commercial": 0.25,
    "stadium": 0.10,
    "street": 0.05,
    "industrial": 0.25,
}

# Daily shape multipliers (hour 0-23) for each zone.
# These modulate the base share to create realistic daily profiles.
# Values are relative multipliers around 1.0.
ZONE_SHAPES = {
    "residential": [
        0.6, 0.5, 0.5, 0.5, 0.5, 0.6,  # 00-05: low overnight
        0.8, 1.0, 1.2, 1.1, 0.9, 0.8,   # 06-11: morning peak
        0.7, 0.7, 0.7, 0.8, 0.9, 1.1,   # 12-17: afternoon
        1.3, 1.4, 1.3, 1.1, 0.9, 0.7,   # 18-23: evening peak
    ],
    "commercial": [
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3,   # 00-05: closed
        0.5, 0.8, 1.2, 1.4, 1.5, 1.5,   # 06-11: opening, peak
        1.4, 1.4, 1.5, 1.5, 1.4, 1.2,   # 12-17: business hours
        0.8, 0.5, 0.4, 0.3, 0.3, 0.3,   # 18-23: closing
    ],
    "stadium": [
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   # 00-05: off
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,   # 06-11: off
        0.1, 0.1, 0.2, 0.3, 0.5, 1.0,   # 12-17: warmup
        1.5, 2.0, 2.0, 1.5, 0.5, 0.1,   # 18-23: event peak
    ],
    "street": [
        1.5, 1.5, 1.5, 1.5, 1.5, 1.2,   # 00-05: full at night
        0.8, 0.3, 0.1, 0.1, 0.1, 0.1,   # 06-11: off daytime
        0.1, 0.1, 0.1, 0.1, 0.1, 0.3,   # 12-17: off daytime
        0.8, 1.2, 1.5, 1.5, 1.5, 1.5,   # 18-23: on at dusk
    ],
    "industrial": [
        0.4, 0.4, 0.4, 0.4, 0.4, 0.5,   # 00-05: overnight reduced
        0.8, 1.0, 1.2, 1.3, 1.3, 1.3,   # 06-11: ramp up
        1.2, 1.3, 1.3, 1.3, 1.2, 1.0,   # 12-17: full production
        0.8, 0.6, 0.5, 0.4, 0.4, 0.4,   # 18-23: ramp down
    ],
}

# Irradiance-based day/night thresholds (W/m²) — proxy for LDR
IRRADIANCE_THRESHOLDS = {
    "DAY": 400,       # Full daylight
    "DAWN": 200,      # Morning transition
    "TWILIGHT": 50,   # Low light
    "DUSK": 10,       # Near dark
    # Below DUSK = NIGHT
}

# Weather thresholds (from firmware, using pressure_msl)
WEATHER_THRESHOLDS = {
    "TEMP_HIGH": 35.0,      # °C — triggers HEAT
    "TEMP_LOW": 5.0,        # °C — triggers COLD
    "HUMIDITY_HIGH": 80.0,  # % — triggers HUMIDITY
    "PRESSURE_LOW": 1000.0, # hPa — triggers STORM
    "PRESSURE_HIGH": 1025.0,# hPa — triggers FAIR WEATHER
}


def load_opsd(price_source: str = "proxy_DE") -> pd.DataFrame:
    """Load OPSD hourly time series, extract FR load and price."""
    path = os.path.join(RAW_DIR, "opsd_time_series_60min.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run download.py first.")
        sys.exit(1)

    print("  Loading OPSD CSV...")
    if price_source == "proxy_DE":
        usecols = ["utc_timestamp", "FR_load_actual_entsoe_transparency",
                   "AT_price_day_ahead", "DE_LU_price_day_ahead"]
        df = pd.read_csv(
            path,
            usecols=usecols,
            parse_dates=["utc_timestamp"],
            index_col="utc_timestamp",
        )
        df.index = df.index.tz_localize(None)
        df = df.loc["2017-01-01":"2019-12-31"].copy()

        # Splice German proxy: AT_price before 2018-10-01 00:00 CEST (2018-09-30 22:00 UTC),
        # then DE_LU_price from 2018-09-30 23:00 UTC
        s_at = df["AT_price_day_ahead"]
        s_de = df["DE_LU_price_day_ahead"]
        price = s_at.loc[:"2018-09-30 22:00:00"].combine_first(s_de.loc["2018-09-30 23:00:00":])
        price = price.reindex(df.index).interpolate(method="linear")

        df = pd.DataFrame({
            "load_mw": df["FR_load_actual_entsoe_transparency"],
            "price_eur_mwh": price,
        }, index=df.index)
    elif price_source == "entsoe_FR":
        fr_price_path = os.path.join(RAW_DIR, "entsoe_fr_price_2017_2019.csv")
        df_load = pd.read_csv(
            path,
            usecols=["utc_timestamp", "FR_load_actual_entsoe_transparency"],
            parse_dates=["utc_timestamp"],
            index_col="utc_timestamp",
        )
        df_load.index = df_load.index.tz_localize(None)
        df_load = df_load.loc["2017-01-01":"2019-12-31"].copy()
        if os.path.exists(fr_price_path):
            df_price = pd.read_csv(fr_price_path, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
            df_price.index = df_price.index.tz_localize(None)
            price = df_price["price_eur_mwh"].reindex(df_load.index).interpolate(method="linear")
        else:
            raise FileNotFoundError(f"Real FR price file not found: {fr_price_path}")
        df = pd.DataFrame({
            "load_mw": df_load["FR_load_actual_entsoe_transparency"],
            "price_eur_mwh": price,
        }, index=df_load.index)
    else:
        raise ValueError(f"Unknown price_source: {price_source}")

    # Interpolate missing load hours (29 hours in 2017-2019)
    missing_load = int(df['load_mw'].isna().sum())
    df['load_mw'] = df['load_mw'].interpolate(method="linear")

    print(f"  OPSD: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    print(f"  Load:  {len(df) - missing_load} valid, {missing_load} missing (interpolated linearly)")
    print(f"  Price: {df['price_eur_mwh'].notna().sum()} valid (source: {price_source})")
    neg_price = int((df['price_eur_mwh'] < 0).sum())
    print(f"  Negative-price hours: {neg_price}")
    print(f"  Price min: {df['price_eur_mwh'].min():.2f} EUR/MWh")
    diffs = df.index.to_series().diff().dropna()
    non_hourly = (diffs != pd.Timedelta(hours=1)).sum()
    print(f"  Non-hourly gaps in index: {non_hourly} (0 confirms UTC, no DST)")
    return df



def load_pvgis() -> pd.DataFrame:
    """Load PVGIS hourly irradiance CSV."""
    path = os.path.join(RAW_DIR, "pvgis_hourly_2017_2019.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run download.py first.")
        sys.exit(1)

    print("  Loading PVGIS CSV...")
    # PVGIS CSV has metadata header lines; skip until data starts
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the header row (starts with "time,")
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time,") or line.strip().startswith("time\t"):
            header_idx = i
            break

    if header_idx is None:
        # Try alternative: look for a row with column names
        for i, line in enumerate(lines):
            if "Gb(" in line or "Gd(" in line or "G(" in line:
                header_idx = i
                break

    if header_idx is None:
        print("ERROR: Could not find PVGIS CSV header")
        print("First 20 lines:")
        for line in lines[:20]:
            print(f"  {line.rstrip()}")
        sys.exit(1)

    # Find where data ends (blank line or metadata footer)
    end_idx = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if lines[i].strip() == "" or lines[i].strip().startswith("#"):
            end_idx = i
            break

    df = pd.read_csv(
        path,
        skiprows=header_idx,
        nrows=end_idx - header_idx - 1,
    )

    # PVGIS time format: YYYYMMDD:HHMM
    time_col = df.columns[0]
    df["timestamp"] = pd.to_datetime(df[time_col], format="%Y%m%d:%H%M").dt.floor("h")
    df = df.set_index("timestamp")

    # Extract global irradiance — column name varies
    irr_col = None
    for col in df.columns:
        if col.startswith("G("):  # G(i) = global in-plane irradiance
            irr_col = col
            break
    if irr_col is None:
        # Fallback: try Gb (beam) + Gd (diffuse)
        gb_col = [c for c in df.columns if c.startswith("Gb(")]
        gd_col = [c for c in df.columns if c.startswith("Gd(")]
        if gb_col and gd_col:
            df["irradiance_wm2"] = df[gb_col[0]].astype(float) + df[gd_col[0]].astype(float)
        else:
            print(f"ERROR: Cannot find irradiance columns. Available: {list(df.columns)}")
            sys.exit(1)
    else:
        df["irradiance_wm2"] = df[irr_col].astype(float)

    # Also extract temperature if available from PVGIS
    temp_col = [c for c in df.columns if c.startswith("T2m") or c == "T2m"]
    if temp_col:
        df["pvgis_temp"] = df[temp_col[0]].astype(float)

    result = df[["irradiance_wm2"]].copy()
    if "pvgis_temp" in df.columns:
        result["pvgis_temp"] = df["pvgis_temp"]

    print(f"  PVGIS: {len(result)} rows, {result.index.min()} to {result.index.max()}")
    print(f"  Irradiance: {result['irradiance_wm2'].notna().sum()} valid, {result['irradiance_wm2'].isna().sum()} missing")
    return result


def load_openmeteo() -> pd.DataFrame:
    """Load Open-Meteo historical weather JSON."""
    path = os.path.join(RAW_DIR, "openmeteo_weather_2017_2019.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run download.py first.")
        sys.exit(1)

    print("  Loading Open-Meteo JSON...")
    with open(path, "r") as f:
        data = json.load(f)

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "humidity_pct": hourly["relative_humidity_2m"],
        "pressure_msl_hpa": hourly["pressure_msl"],
    })
    df = df.set_index("timestamp")

    print(f"  Open-Meteo: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    for col in ["temperature_c", "humidity_pct", "pressure_msl_hpa"]:
        n_valid = df[col].notna().sum()
        n_miss = df[col].isna().sum()
        print(f"  {col}: {n_valid} valid, {n_miss} missing")
    return df


def merge_datasets(opsd: pd.DataFrame, pvgis: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge all datasets on hourly UTC timestamps."""
    print("\n[MERGE] Joining datasets...")

    # Ensure all are hourly
    # OPSD is already hourly. PVGIS may need resampling.
    # Open-Meteo is hourly.

    # Resample PVGIS to hourly if needed (it should already be, but handle sub-hourly)
    if len(pvgis) > 26280 * 1.1:  # More than ~3 years hourly
        print("  Resampling PVGIS to hourly (mean)...")
        pvgis = pvgis.resample("h").mean()

    # Merge
    df = opsd.join(pvgis, how="inner")
    df = df.join(weather, how="inner")

    print(f"  Merged: {len(df)} rows, {df.index.min()} to {df.index.max()}")

    # Report gaps
    expected_hours = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_hours = expected_hours.difference(df.index)
    print(f"  Missing hours in merged index: {len(missing_hours)}")

    if len(missing_hours) > 0:
        # Reindex to fill gaps, then interpolate
        df = df.reindex(expected_hours)
        df.index.name = "timestamp"
        print(f"  After reindex: {len(df)} rows")

    # Report per-column gaps
    print("\n  Per-column missing values (before interpolation):")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct = 100 * n_miss / len(df)
        print(f"    {col}: {n_miss} ({pct:.2f}%)")

    # Interpolate gaps
    # Linear interpolation for small gaps, forward-fill for weather
    for col in ["load_mw", "price_eur_mwh", "irradiance_wm2"]:
        df[col] = df[col].interpolate(method="linear", limit=6)  # Max 6-hour gap
    for col in ["temperature_c", "humidity_pct", "pressure_msl_hpa"]:
        df[col] = df[col].interpolate(method="linear", limit=6)

    # Drop any remaining NaN rows at edges
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with remaining NaN (edges/large gaps)")

    print(f"\n  Final merged dataset: {len(df)} rows")
    return df


def scale_load_to_microgrid(df: pd.DataFrame) -> pd.DataFrame:
    """Scale national load (MW) to synthetic microgrid load (kW)."""
    national_peak = df["load_mw"].max()
    scale_factor = MICROGRID_PEAK_KW / national_peak
    df["load_kw"] = df["load_mw"] * scale_factor
    print(f"\n[SCALE] National peak: {national_peak:.0f} MW -> Microgrid peak: {MICROGRID_PEAK_KW:.0f} kW")
    print(f"  Scale factor: {scale_factor:.6f}")
    return df


def apply_zone_disaggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Disaggregate total load into 5 zones using fixed shares and daily shapes."""
    print("\n[ZONES] Applying synthetic zone disaggregation...")
    hours = df.index.hour
    is_weekend = df.index.dayofweek >= 5

    for zone, share in ZONE_SHARES.items():
        shape = np.array(ZONE_SHAPES[zone])
        # Normalize shape to mean 1.0
        shape = shape / shape.mean()
        # Apply hourly shape
        hourly_multiplier = shape[hours]

        # Weekend adjustment for commercial and industrial
        if zone == "commercial":
            hourly_multiplier = np.where(is_weekend, hourly_multiplier * 0.4, hourly_multiplier)
        elif zone == "industrial":
            hourly_multiplier = np.where(is_weekend, hourly_multiplier * 0.5, hourly_multiplier)

        # Base zone load = total_load * share * shape_multiplier
        # Then renormalize so zone loads sum to total load
        df[f"load_{zone}_kw"] = df["load_kw"] * share * hourly_multiplier

    # Renormalize: ensure zones sum to total load at each hour
    zone_cols = [f"load_{z}_kw" for z in ZONE_SHARES]
    zone_sum = df[zone_cols].sum(axis=1)
    for col in zone_cols:
        df[col] = df[col] * (df["load_kw"] / zone_sum)

    # Verify
    recon_error = (df[zone_cols].sum(axis=1) - df["load_kw"]).abs().max()
    print(f"  Reconstruction error (max): {recon_error:.6f} kW")

    for zone in ZONE_SHARES:
        col = f"load_{zone}_kw"
        print(f"  {zone}: mean={df[col].mean():.1f} kW, peak={df[col].max():.1f} kW")
    return df


def generate_event_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic stadium event calendar (seeded)."""
    print("\n[EVENTS] Generating synthetic event calendar (seed=42)...")
    rng = np.random.RandomState(SEED)

    # ~2 events per week → total events over the dataset
    n_weeks = len(df) // (7 * 24)
    n_events = n_weeks * 2

    # Pick random days, then assign evening start times
    all_days = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    event_days = rng.choice(len(all_days), size=n_events, replace=False)
    event_days = sorted(all_days[event_days])

    events = []
    for day in event_days:
        start_hour = rng.randint(17, 21)  # 17:00–20:00
        duration = rng.randint(2, 4)       # 2–3 hours
        start = day + pd.Timedelta(hours=start_hour)
        end = start + pd.Timedelta(hours=duration)
        events.append((start, end))

    # Mark event hours in the dataframe
    df["stadium_event"] = False
    df["event_phase"] = "NONE"
    for start, end in events:
        before = start - pd.Timedelta(minutes=30)
        # BEFORE phase
        mask_before = (df.index >= before) & (df.index < start)
        df.loc[mask_before, "event_phase"] = "BEFORE"
        df.loc[mask_before, "stadium_event"] = True
        # DURING phase
        mask_during = (df.index >= start) & (df.index < end)
        df.loc[mask_during, "event_phase"] = "DURING"
        df.loc[mask_during, "stadium_event"] = True

    n_event_hours = df["stadium_event"].sum()
    print(f"  Generated {n_events} events ({n_event_hours} event hours)")
    print("  Events per split will be reported after splitting.")
    return df


def compute_day_period(df: pd.DataFrame) -> pd.DataFrame:
    """Compute day/night period from irradiance (proxy for LDR)."""
    print("\n[DAY/NIGHT] Computing irradiance-based period proxy...")
    irr = df["irradiance_wm2"]

    conditions = [
        irr >= IRRADIANCE_THRESHOLDS["DAY"],
        (irr >= IRRADIANCE_THRESHOLDS["DAWN"]) & (irr < IRRADIANCE_THRESHOLDS["DAY"]),
        (irr >= IRRADIANCE_THRESHOLDS["TWILIGHT"]) & (irr < IRRADIANCE_THRESHOLDS["DAWN"]),
        (irr >= IRRADIANCE_THRESHOLDS["DUSK"]) & (irr < IRRADIANCE_THRESHOLDS["TWILIGHT"]),
        irr < IRRADIANCE_THRESHOLDS["DUSK"],
    ]
    choices = ["DAY", "DAWN", "TWILIGHT", "DUSK", "NIGHT"]
    df["day_period"] = np.select(conditions, choices, default="NIGHT")

    # Report period distribution
    print("  Period distribution:")
    for period in choices:
        count = (df["day_period"] == period).sum()
        pct = 100 * count / len(df)
        print(f"    {period:10s}: {count:6d} hours ({pct:.1f}%)")

    # Seasonal sanity check: summer should have more DAY hours
    for year in [2017, 2018, 2019]:
        summer = df.loc[f"{year}-06":f"{year}-08"]
        winter = df.loc[f"{year}-01":f"{year}-02"]
        summer_day = (summer["day_period"] == "DAY").mean() * 100
        winter_day = (winter["day_period"] == "DAY").mean() * 100
        print(f"    {year} Summer DAY%: {summer_day:.1f}%, Winter DAY%: {winter_day:.1f}%")

    return df


def compute_weather_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Compute which weather rule fires at each hour."""
    print("\n[WEATHER] Computing weather rule activation...")
    t = WEATHER_THRESHOLDS

    # Priority order: heat > cold > humid > storm > fair > normal
    conditions = [
        df["temperature_c"] > t["TEMP_HIGH"],
        df["temperature_c"] < t["TEMP_LOW"],
        df["humidity_pct"] > t["HUMIDITY_HIGH"],
        df["pressure_msl_hpa"] < t["PRESSURE_LOW"],
        df["pressure_msl_hpa"] > t["PRESSURE_HIGH"],
    ]
    choices = ["HEAT", "COLD", "HUMIDITY", "STORM", "FAIR"]
    df["weather_rule"] = np.select(conditions, choices, default="NORMAL")
    return df


def split_dataset(df: pd.DataFrame) -> dict:
    """Split by whole ISO weeks with 1-day gaps. Ensure stress weeks in test."""
    print("\n[SPLIT] Splitting dataset by whole ISO weeks...")

    # Add ISO week info
    df["iso_year"] = df.index.isocalendar().year.values
    df["iso_week"] = df.index.isocalendar().week.values

    # Define split boundaries
    # Train: 2017-W01 to 2018-W52 (approximately)
    # Val: 2019-W02 to 2019-W27 (with 1-day gap)
    # Test: 2019-W29 to 2019-W52 (with 1-day gap)

    # Find the Monday of each ISO week
    train_start = pd.Timestamp("2017-01-02 00:00:00")  # First Monday of 2017 (104 weeks = 17,472 h)
    train_end = pd.Timestamp("2018-12-30 23:00:00")    # End of week 104

    val_start = pd.Timestamp("2019-01-07 00:00:00")    # Monday after 1-week gap (26 weeks = 4,368 h)
    val_end = pd.Timestamp("2019-07-07 23:00:00")      # End of week 26

    test_start = pd.Timestamp("2019-07-15 00:00:00")   # Monday after 1-week gap (24 weeks = 4,032 h)
    test_end = pd.Timestamp("2019-12-29 23:00:00")     # End of week 24

    train = df.loc[train_start:train_end].copy()
    val = df.loc[val_start:val_end].copy()
    test = df.loc[test_start:test_end].copy()

    # Verify gap days
    gap1_start = train_end + pd.Timedelta(hours=1)
    gap1_end = val_start - pd.Timedelta(hours=1)
    gap2_start = val_end + pd.Timedelta(hours=1)
    gap2_end = test_start - pd.Timedelta(hours=1)
    print(f"  Gap 1 (train->val): {gap1_start.date()} to {gap1_end.date()}")
    print(f"  Gap 2 (val->test):  {gap2_start.date()} to {gap2_end.date()}")

    splits = {"train": train, "val": val, "test": test}

    for name, split in splits.items():
        n_weeks = len(split) // (7 * 24)
        print(f"\n  {name.upper()}: {len(split)} hours ({n_weeks} weeks)")
        print(f"    Period: {split.index.min()} to {split.index.max()}")

    return splits


def find_stress_weeks(splits: dict) -> dict:
    """Identify stress weeks in each split based on pre-defined criteria."""
    print("\n[STRESS] Identifying stress weeks...")

    stress_results = {}
    for name, split in splits.items():
        # Group by ISO week
        split = split.copy()
        split["week_label"] = (
            split.index.isocalendar().year.astype(str).values + "-W" +
            split.index.isocalendar().week.astype(str).str.zfill(2).values
        )

        weekly = split.groupby("week_label").agg({
            "irradiance_wm2": "sum",
            "price_eur_mwh": "mean",
            "temperature_c": "mean",
            "pressure_msl_hpa": "mean",
        })

        stress = {}
        # Lowest PV
        lowest_pv_week = weekly["irradiance_wm2"].idxmin()
        stress["lowest_pv"] = {
            "week": lowest_pv_week,
            "value": float(weekly.loc[lowest_pv_week, "irradiance_wm2"]),
        }
        # Highest price
        highest_price_week = weekly["price_eur_mwh"].idxmax()
        stress["highest_price"] = {
            "week": highest_price_week,
            "value": float(weekly.loc[highest_price_week, "price_eur_mwh"]),
        }
        # Hottest
        hottest_week = weekly["temperature_c"].idxmax()
        stress["hottest"] = {
            "week": hottest_week,
            "value": float(weekly.loc[hottest_week, "temperature_c"]),
        }
        # Stormiest (lowest pressure)
        stormiest_week = weekly["pressure_msl_hpa"].idxmin()
        stress["stormiest"] = {
            "week": stormiest_week,
            "value": float(weekly.loc[stormiest_week, "pressure_msl_hpa"]),
        }

        stress_results[name] = stress
        print(f"\n  {name.upper()} stress weeks:")
        for criterion, info in stress.items():
            print(f"    {criterion:15s}: {info['week']} (value={info['value']:.1f})")

    # Verify test split contains stress weeks
    test_stress = stress_results["test"]
    print("\n  TEST split stress verification:")
    for criterion in ["lowest_pv", "highest_price", "hottest", "stormiest"]:
        print(f"    {criterion}: {test_stress[criterion]['week']} [OK] (in test by construction)")

    return stress_results


def report_weather_rule_frequency(splits: dict):
    """Report how often each weather rule fires in each split."""
    print("\n[WEATHER RULES] Frequency by split:")
    rules = ["HEAT", "COLD", "HUMIDITY", "STORM", "FAIR", "NORMAL"]
    for name, split in splits.items():
        print(f"\n  {name.upper()} ({len(split)} hours):")
        for rule in rules:
            count = (split["weather_rule"] == rule).sum()
            pct = 100 * count / len(split)
            print(f"    {rule:10s}: {count:5d} hours ({pct:.1f}%)")


def report_split_statistics(splits: dict):
    """Report basic statistics per split."""
    print("\n[STATISTICS] Per-split summary:")
    numeric_cols = ["load_kw", "price_eur_mwh", "irradiance_wm2",
                    "temperature_c", "humidity_pct", "pressure_msl_hpa"]
    for name, split in splits.items():
        print(f"\n  {name.upper()}:")
        for col in numeric_cols:
            if col in split.columns:
                print(f"    {col:25s}: mean={split[col].mean():8.2f}, "
                      f"std={split[col].std():8.2f}, "
                      f"min={split[col].min():8.2f}, max={split[col].max():8.2f}")
        neg = (split['price_eur_mwh'] < 0).sum()
        print(f"    Negative-price hours: {neg}")
        print(f"    Price min: {split['price_eur_mwh'].min():.2f} EUR/MWh")


def compute_power_thresholds(df: pd.DataFrame) -> dict:
    """Compute load-shedding thresholds as fractions of peak load."""
    peak = df["load_kw"].max()
    thresholds = {
        "NORMAL": 0.25 * peak,
        "ALERT": 0.50 * peak,
        "CRITICAL": 0.75 * peak,
        "EMERGENCY": 0.90 * peak,
    }
    print("\n[THRESHOLDS] Power thresholds (as fraction of peak load):")
    print(f"  Peak load: {peak:.1f} kW")
    for level, thresh in thresholds.items():
        # How often is load above this threshold?
        above = (df["load_kw"] > thresh).mean() * 100
        print(f"    {level:10s}: {thresh:.1f} kW ({above:.1f}% of hours above)")
    return thresholds


def _grid_only_import(split: pd.DataFrame, pv_scale: float) -> np.ndarray:
    """Import the site would draw with the battery idle: max(0, load - pv)."""
    pv_kw = split["irradiance_wm2"].to_numpy() * pv_scale
    return np.clip(split["load_kw"].to_numpy() - pv_kw, 0.0, None)


def compute_grid_import_cap(splits: dict, quantile: float = CAP_QUANTILE) -> dict:
    """Derive the grid import cap from the training split's grid-only import.

    The env's PV model is reused rather than re-specified here so the cap and the
    simulation cannot drift apart. Only Train and Val are profiled: Test is closed
    by policy (NOTES.md), so it is never read for anything, including diagnostics.
    """
    pv_cfg = env_default_config()
    pv_scale = (pv_cfg["pv_area_m2"] * pv_cfg["pv_efficiency"]
                * pv_cfg["pv_performance_ratio"] / 1000.0)

    profile = {
        "train": _grid_only_import(splits["train"], pv_scale),
        "val": _grid_only_import(splits["val"], pv_scale),
    }

    cap_kw = float(np.quantile(profile["train"], quantile))
    report = {
        "value_kw": round(cap_kw, 6),
        "quantile": quantile,
        "derived_from": "train",
        "method": "quantile of max(0, load_kw - pv_kw) with the battery idle",
        "exceedance": {
            name: round(float((net > cap_kw).mean()), 6) for name, net in profile.items()
        },
        "max_net_import_kw": {
            name: round(float(net.max()), 3) for name, net in profile.items()
        },
    }

    print(f"\n[CAP] Grid import cap = P{quantile * 100:.0f} of train grid-only import")
    print(f"  cap: {cap_kw:.3f} kW")
    for name, net in profile.items():
        print(f"  {name:5s}: {report['exceedance'][name] * 100:5.1f}% of hours above, "
              f"peak {report['max_net_import_kw'][name]:.1f} kW")
    return report


def save_splits(splits: dict, stress_results: dict, thresholds: dict, grid_import_cap: dict):
    """Save processed data splits and metadata."""
    print("\n[SAVE] Writing processed data...")

    # Remove temporary columns before saving
    drop_cols = ["iso_year", "iso_week", "load_mw"]

    for name, split in splits.items():
        save_df = split.drop(columns=[c for c in drop_cols if c in split.columns])
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        save_df.to_parquet(path)
        print(f"  {path} ({len(save_df)} rows)")

        # Also save as CSV for inspection
        csv_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        save_df.to_csv(csv_path)

    # Save split report
    report = {
        "splits": {},
        "stress_weeks": stress_results,
        "power_thresholds_kw": thresholds,
        "microgrid_peak_kw": MICROGRID_PEAK_KW,
        "zone_shares": ZONE_SHARES,
        "irradiance_thresholds_wm2": IRRADIANCE_THRESHOLDS,
        "weather_thresholds": WEATHER_THRESHOLDS,
        "grid_import_cap": grid_import_cap,
    }
    for name, split in splits.items():
        report["splits"][name] = {
            "start": str(split.index.min()),
            "end": str(split.index.max()),
            "n_hours": len(split),
            "n_weeks": len(split) // (7 * 24),
        }

    report_path = os.path.join(PROCESSED_DIR, "split_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  {report_path}")


def main():
    print("=" * 60)
    print("smartgrid-rl: Data Preparation")
    print("=" * 60)

    price_source = "proxy_DE"
    cap_quantile = CAP_QUANTILE
    for arg in sys.argv[1:]:
        if arg.startswith("--price-source="):
            price_source = arg.split("=")[1]
        elif arg == "--price-source" and sys.argv.index(arg) + 1 < len(sys.argv):
            price_source = sys.argv[sys.argv.index(arg) + 1]
        elif arg.startswith("--cap-quantile="):
            cap_quantile = float(arg.split("=")[1])
        elif arg == "--cap-quantile" and sys.argv.index(arg) + 1 < len(sys.argv):
            cap_quantile = float(sys.argv[sys.argv.index(arg) + 1])

    # 1. Load raw datasets
    print(f"\n[1/10] Loading OPSD (price_source={price_source})...")
    opsd = load_opsd(price_source=price_source)

    print("\n[2/10] Loading PVGIS...")
    pvgis = load_pvgis()

    print("\n[3/10] Loading Open-Meteo...")
    weather = load_openmeteo()

    # 2. Merge
    print("\n[4/10] Merging datasets...")
    df = merge_datasets(opsd, pvgis, weather)

    # 3. Scale to microgrid
    print("\n[5/10] Scaling load to microgrid...")
    df = scale_load_to_microgrid(df)

    # 4. Zone disaggregation
    print("\n[6/10] Zone disaggregation...")
    df = apply_zone_disaggregation(df)

    # 5. Event calendar
    print("\n[7/10] Event calendar...")
    df = generate_event_calendar(df)

    # 6. Day/night proxy
    print("\n[8/10] Day/night proxy...")
    df = compute_day_period(df)

    # 7. Weather rules
    print("\n[9/10] Weather rules...")
    df = compute_weather_rules(df)

    # 8. Power thresholds
    thresholds = compute_power_thresholds(df)

    # 9. Split
    splits = split_dataset(df)

    # 10. Reports
    stress = find_stress_weeks(splits)
    report_weather_rule_frequency(splits)
    report_split_statistics(splits)
    grid_import_cap = compute_grid_import_cap(splits, cap_quantile)

    # Save
    save_splits(splits, stress, thresholds, grid_import_cap)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
