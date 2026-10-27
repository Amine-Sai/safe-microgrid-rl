import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from env.smartgrid_env import default_config as env_default_config  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "config.yaml")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_preparation_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the `data_preparation` configuration block from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    prep_cfg = full_cfg.get("data_preparation")
    if not prep_cfg:
        raise KeyError(f"Missing 'data_preparation' section in {config_path}")
    return prep_cfg


def load_opsd(prep_cfg: dict) -> pd.DataFrame:
    """Load OPSD hourly time series, extract FR load and day-ahead price."""
    path = os.path.join(RAW_DIR, "opsd_time_series_60min.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run download.py first.")
        sys.exit(1)

    price_source = prep_cfg["price_source"]
    date_start = prep_cfg["date_range"]["start"]
    date_end = prep_cfg["date_range"]["end"]
    splice_utc = pd.Timestamp(prep_cfg["de_lu_price_splice_utc"])
    splice_next_hour_utc = splice_utc + pd.Timedelta(hours=1)

    print("  Loading OPSD CSV...")
    if price_source == "proxy_DE":
        usecols = [
            "utc_timestamp",
            "FR_load_actual_entsoe_transparency",
            "AT_price_day_ahead",
            "DE_LU_price_day_ahead",
        ]
        df = pd.read_csv(
            path,
            usecols=usecols,
            parse_dates=["utc_timestamp"],
            index_col="utc_timestamp",
        )
        df.index = df.index.tz_localize(None)
        df = df.loc[date_start:date_end].copy()

        # Splice German proxy around the AT / DE-LU bidding-zone split timestamp
        s_at = df["AT_price_day_ahead"]
        s_de = df["DE_LU_price_day_ahead"]
        price = s_at.loc[:splice_utc].combine_first(s_de.loc[splice_next_hour_utc:])
        price = price.reindex(df.index).interpolate(method="linear")

        df = pd.DataFrame(
            {
                "load_mw": df["FR_load_actual_entsoe_transparency"],
                "price_eur_mwh": price,
            },
            index=df.index,
        )
    elif price_source == "entsoe_FR":
        fr_price_path = os.path.join(RAW_DIR, "entsoe_fr_price_2017_2019.csv")
        df_load = pd.read_csv(
            path,
            usecols=["utc_timestamp", "FR_load_actual_entsoe_transparency"],
            parse_dates=["utc_timestamp"],
            index_col="utc_timestamp",
        )
        df_load.index = df_load.index.tz_localize(None)
        df_load = df_load.loc[date_start:date_end].copy()
        if os.path.exists(fr_price_path):
            df_price = pd.read_csv(
                fr_price_path,
                parse_dates=["utc_timestamp"],
                index_col="utc_timestamp",
            )
            df_price.index = df_price.index.tz_localize(None)
            price = df_price["price_eur_mwh"].reindex(df_load.index).interpolate(method="linear")
        else:
            raise FileNotFoundError(f"Real FR price file not found: {fr_price_path}")
        df = pd.DataFrame(
            {
                "load_mw": df_load["FR_load_actual_entsoe_transparency"],
                "price_eur_mwh": price,
            },
            index=df_load.index,
        )
    else:
        raise ValueError(f"Unknown price_source: {price_source}")

    missing_load = int(df["load_mw"].isna().sum())
    df["load_mw"] = df["load_mw"].interpolate(method="linear")

    print(f"  OPSD: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    print(f"  Load:  {len(df) - missing_load} valid, {missing_load} missing (interpolated linearly)")
    print(f"  Price: {df['price_eur_mwh'].notna().sum()} valid (source: {price_source})")
    neg_price = int((df["price_eur_mwh"] < 0).sum())
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
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("time,") or line.strip().startswith("time\t"):
            header_idx = i
            break

    if header_idx is None:
        for i, line in enumerate(lines):
            if "Gb(" in line or "Gd(" in line or "G(" in line:
                header_idx = i
                break

    if header_idx is None:
        print("ERROR: Could not find PVGIS CSV header")
        sys.exit(1)

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

    time_col = df.columns[0]
    df["timestamp"] = pd.to_datetime(df[time_col], format="%Y%m%d:%H%M").dt.floor("h")
    df = df.set_index("timestamp")

    irr_col = None
    for col in df.columns:
        if col.startswith("G("):
            irr_col = col
            break
    if irr_col is None:
        gb_col = [c for c in df.columns if c.startswith("Gb(")]
        gd_col = [c for c in df.columns if c.startswith("Gd(")]
        if gb_col and gd_col:
            df["irradiance_wm2"] = df[gb_col[0]].astype(float) + df[gd_col[0]].astype(float)
        else:
            print(f"ERROR: Cannot find irradiance columns. Available: {list(df.columns)}")
            sys.exit(1)
    else:
        df["irradiance_wm2"] = df[irr_col].astype(float)

    temp_col = [c for c in df.columns if c.startswith("T2m") or c == "T2m"]
    if temp_col:
        df["pvgis_temp"] = df[temp_col[0]].astype(float)

    result = df[["irradiance_wm2"]].copy()
    if "pvgis_temp" in df.columns:
        result["pvgis_temp"] = df["pvgis_temp"]

    print(f"  PVGIS: {len(result)} rows, {result.index.min()} to {result.index.max()}")
    print(
        f"  Irradiance: {result['irradiance_wm2'].notna().sum()} valid, "
        f"{result['irradiance_wm2'].isna().sum()} missing"
    )
    return result


def load_openmeteo() -> pd.DataFrame:
    """Load Open-Meteo historical weather JSON."""
    path = os.path.join(RAW_DIR, "openmeteo_weather_2017_2019.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run download.py first.")
        sys.exit(1)

    print("  Loading Open-Meteo JSON...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hourly = data["hourly"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature_c": hourly["temperature_2m"],
            "humidity_pct": hourly["relative_humidity_2m"],
            "pressure_msl_hpa": hourly["pressure_msl"],
        }
    )
    df = df.set_index("timestamp")

    print(f"  Open-Meteo: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    for col in ["temperature_c", "humidity_pct", "pressure_msl_hpa"]:
        n_valid = df[col].notna().sum()
        n_miss = df[col].isna().sum()
        print(f"  {col}: {n_valid} valid, {n_miss} missing")
    return df


def merge_datasets(
    opsd: pd.DataFrame,
    pvgis: pd.DataFrame,
    weather: pd.DataFrame,
    prep_cfg: dict,
) -> pd.DataFrame:
    """Merge all datasets on hourly UTC timestamps."""
    print("\n[MERGE] Joining datasets...")

    max_gap_hours = int(prep_cfg["max_interpolation_gap_hours"])
    subhourly_factor = float(prep_cfg.get("pvgis_subhourly_row_factor", 1.1))

    if len(pvgis) > len(opsd) * subhourly_factor:
        print("  Resampling PVGIS to hourly (mean)...")
        pvgis = pvgis.resample("h").mean()

    df = opsd.join(pvgis, how="inner")
    df = df.join(weather, how="inner")

    print(f"  Merged: {len(df)} rows, {df.index.min()} to {df.index.max()}")

    expected_hours = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_hours = expected_hours.difference(df.index)
    print(f"  Missing hours in merged index: {len(missing_hours)}")

    if len(missing_hours) > 0:
        df = df.reindex(expected_hours)
        df.index.name = "timestamp"
        print(f"  After reindex: {len(df)} rows")

    print("\n  Per-column missing values (before interpolation):")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct = 100 * n_miss / len(df)
        print(f"    {col}: {n_miss} ({pct:.2f}%)")

    interp_cols = [
        "load_mw",
        "price_eur_mwh",
        "irradiance_wm2",
        "temperature_c",
        "humidity_pct",
        "pressure_msl_hpa",
    ]
    for col in interp_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=max_gap_hours)

    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with remaining NaN (edges/large gaps)")

    print(f"\n  Final merged dataset: {len(df)} rows")
    return df


def scale_load_to_microgrid(df: pd.DataFrame, prep_cfg: dict) -> pd.DataFrame:
    """Scale national load (MW) to microgrid load (kW)."""
    microgrid_peak_kw = float(prep_cfg["microgrid_peak_kw"])
    national_peak = df["load_mw"].max()
    scale_factor = microgrid_peak_kw / national_peak
    df["load_kw"] = df["load_mw"] * scale_factor
    print(
        f"\n[SCALE] National peak: {national_peak:.0f} MW -> "
        f"Microgrid peak: {microgrid_peak_kw:.0f} kW"
    )
    print(f"  Scale factor: {scale_factor:.6f}")
    return df


def generate_event_calendar(df: pd.DataFrame, prep_cfg: dict) -> pd.DataFrame:
    """Generate seeded stadium event calendar in local time before zone disaggregation."""
    seed = int(prep_cfg["seed"])
    local_tz = prep_cfg["local_timezone"]
    events_cfg = prep_cfg["events"]
    events_per_week = int(events_cfg["events_per_week"])
    start_hour_min = int(events_cfg["start_hour_min"])
    start_hour_max_excl = int(events_cfg["start_hour_max_exclusive"])
    dur_min = int(events_cfg["duration_hours_min"])
    dur_max_excl = int(events_cfg["duration_hours_max_exclusive"])
    warmup_hours = int(events_cfg["warmup_hours"])

    print(f"\n[EVENTS] Generating stadium event calendar (seed={seed}, tz={local_tz})...")
    rng = np.random.RandomState(seed)

    local_index = df.index.tz_localize("UTC").tz_convert(local_tz)
    n_weeks = len(df) // (7 * 24)
    n_events = n_weeks * events_per_week

    all_local_days = pd.date_range(
        local_index.min().normalize(),
        local_index.max().normalize(),
        freq="D",
    )
    chosen_indices = rng.choice(len(all_local_days), size=n_events, replace=False)
    event_days = sorted(all_local_days[chosen_indices])

    df["stadium_event"] = False
    df["event_warmup"] = False

    for day in event_days:
        start_hour = int(rng.randint(start_hour_min, start_hour_max_excl))
        duration = int(rng.randint(dur_min, dur_max_excl))
        start_local = day + pd.Timedelta(hours=start_hour)
        end_local = start_local + pd.Timedelta(hours=duration)
        warmup_local = start_local - pd.Timedelta(hours=warmup_hours)

        mask_warmup = (local_index >= warmup_local) & (local_index < start_local)
        mask_active = (local_index >= start_local) & (local_index < end_local)

        df.loc[mask_warmup, "event_warmup"] = True
        df.loc[mask_active, "stadium_event"] = True

    # Ensure active events take precedence if back-to-back events ever overlap
    df.loc[df["stadium_event"], "event_warmup"] = False

    n_active_hours = int(df["stadium_event"].sum())
    n_warmup_hours = int(df["event_warmup"].sum())
    print(
        f"  Generated {n_events} events "
        f"({n_active_hours} active event hours, {n_warmup_hours} pre-event warmup hours)"
    )
    return df


def _circular_hour_distance(hours: np.ndarray, center_hour: float) -> np.ndarray:
    """Shortest distance on a 24-hour clock between `hours` and `center_hour`."""
    diff = np.abs(hours - center_hour)
    return np.minimum(diff, 24.0 - diff)


def compute_zone_multiplier(
    zone_cfg: dict,
    local_hours: np.ndarray,
    irradiance_wm2: np.ndarray,
    stadium_event: np.ndarray,
    event_warmup: np.ndarray,
) -> np.ndarray:
    """Compute unnormalized hourly activity multiplier for a single zone from config."""
    profile_type = zone_cfg["profile_type"]

    if profile_type == "bimodal_peaks":
        baseload = float(zone_cfg["baseload_factor"])
        m_hour = float(zone_cfg["morning_peak_hour"])
        m_amp = float(zone_cfg["morning_amplitude"])
        m_width = float(zone_cfg["morning_width_hours"])
        e_hour = float(zone_cfg["evening_peak_hour"])
        e_amp = float(zone_cfg["evening_amplitude"])
        e_width = float(zone_cfg["evening_width_hours"])

        d_morning = _circular_hour_distance(local_hours, m_hour)
        d_evening = _circular_hour_distance(local_hours, e_hour)
        morning_peak = m_amp * np.exp(-0.5 * (d_morning / m_width) ** 2)
        evening_peak = e_amp * np.exp(-0.5 * (d_evening / e_width) ** 2)
        return baseload + morning_peak + evening_peak

    if profile_type == "business_window":
        baseload = float(zone_cfg["baseload_factor"])
        amplitude = float(zone_cfg["active_amplitude"])
        open_hour = float(zone_cfg["open_hour"])
        close_hour = float(zone_cfg["close_hour"])
        ramp_width = float(zone_cfg["ramp_width_hours"])

        ramp_up = 1.0 / (1.0 + np.exp(-(local_hours - open_hour) / ramp_width))
        ramp_down = 1.0 - 1.0 / (1.0 + np.exp(-(local_hours - close_hour) / ramp_width))
        return baseload + amplitude * (ramp_up * ramp_down)

    if profile_type == "irradiance_driven":
        night_thresh = float(zone_cfg["night_irradiance_threshold_wm2"])
        day_thresh = float(zone_cfg["day_irradiance_threshold_wm2"])
        night_mult = float(zone_cfg["night_multiplier"])
        twilight_mult = float(zone_cfg["twilight_multiplier"])
        day_mult = float(zone_cfg["day_multiplier"])

        # Smooth transition from night -> twilight -> day as solar irradiance rises
        mid_thresh = 0.5 * (night_thresh + day_thresh)
        mult = np.where(
            irradiance_wm2 <= night_thresh,
            night_mult,
            np.where(
                irradiance_wm2 >= day_thresh,
                day_mult,
                np.where(
                    irradiance_wm2 <= mid_thresh,
                    night_mult
                    + (twilight_mult - night_mult)
                    * (irradiance_wm2 - night_thresh)
                    / max(1e-6, mid_thresh - night_thresh),
                    twilight_mult
                    + (day_mult - twilight_mult)
                    * (irradiance_wm2 - mid_thresh)
                    / max(1e-6, day_thresh - mid_thresh),
                ),
            ),
        )
        return mult.astype(np.float64)

    if profile_type == "event_driven":
        baseload = float(zone_cfg["baseload_factor"])
        day_maint = float(zone_cfg["daytime_maintenance_factor"])
        day_start = float(zone_cfg["daytime_start_hour"])
        day_end = float(zone_cfg["daytime_end_hour"])
        warmup_mult = float(zone_cfg["warmup_multiplier"])
        event_mult = float(zone_cfg["active_event_multiplier"])

        in_daytime = (local_hours >= day_start) & (local_hours < day_end)
        mult = np.where(in_daytime, day_maint, baseload)
        mult = np.where(event_warmup, warmup_mult, mult)
        mult = np.where(stadium_event, event_mult, mult)
        return mult.astype(np.float64)

    raise ValueError(f"Unknown zone profile_type: {profile_type!r}")


def apply_zone_disaggregation(df: pd.DataFrame, prep_cfg: dict) -> pd.DataFrame:
    """Disaggregate total load into zones using parametric local-time and physical profiles."""
    local_tz = prep_cfg["local_timezone"]
    zones_cfg = prep_cfg["zones"]
    print(f"\n[ZONES] Applying parametric zone disaggregation (tz={local_tz})...")

    local_index = df.index.tz_localize("UTC").tz_convert(local_tz)
    local_hours = np.asarray(local_index.hour, dtype=np.float64)
    is_weekend = np.asarray(local_index.day_of_week >= 5, dtype=bool)

    irradiance_wm2 = df["irradiance_wm2"].to_numpy(dtype=np.float64)
    stadium_event = df["stadium_event"].to_numpy(dtype=bool)
    event_warmup = df["event_warmup"].to_numpy(dtype=bool)

    for zone_name, zone_cfg in zones_cfg.items():
        share = float(zone_cfg["share"])
        weekend_mult = float(zone_cfg.get("weekend_multiplier", 1.0))

        raw_multiplier = compute_zone_multiplier(
            zone_cfg=zone_cfg,
            local_hours=local_hours,
            irradiance_wm2=irradiance_wm2,
            stadium_event=stadium_event,
            event_warmup=event_warmup,
        )
        # Normalize shape to mean 1.0 so `share` controls the zone's baseline energy weight
        normalized_multiplier = raw_multiplier / raw_multiplier.mean()

        if weekend_mult != 1.0:
            normalized_multiplier = np.where(
                is_weekend,
                normalized_multiplier * weekend_mult,
                normalized_multiplier,
            )

        df[f"load_{zone_name}_kw"] = df["load_kw"] * share * normalized_multiplier

    # Renormalize at every hour so zone loads sum exactly to total load_kw
    zone_cols = [f"load_{zone_name}_kw" for zone_name in zones_cfg]
    zone_sum = df[zone_cols].sum(axis=1)
    for col in zone_cols:
        df[col] = df[col] * (df["load_kw"] / zone_sum)

    recon_error = (df[zone_cols].sum(axis=1) - df["load_kw"]).abs().max()
    print(f"  Reconstruction error (max): {recon_error:.6f} kW")

    for zone_name in zones_cfg:
        col = f"load_{zone_name}_kw"
        print(f"  {zone_name:12s}: mean={df[col].mean():6.1f} kW, peak={df[col].max():6.1f} kW")

    return df


def split_dataset(df: pd.DataFrame, prep_cfg: dict) -> dict:
    """Split dataset by configured whole-week boundaries."""
    print("\n[SPLIT] Splitting dataset by whole ISO weeks...")

    splits_cfg = prep_cfg["splits"]
    train_start = pd.Timestamp(splits_cfg["train"]["start"])
    train_end = pd.Timestamp(splits_cfg["train"]["end"])
    val_start = pd.Timestamp(splits_cfg["val"]["start"])
    val_end = pd.Timestamp(splits_cfg["val"]["end"])
    test_start = pd.Timestamp(splits_cfg["test"]["start"])
    test_end = pd.Timestamp(splits_cfg["test"]["end"])

    train = df.loc[train_start:train_end].copy()
    val = df.loc[val_start:val_end].copy()
    test = df.loc[test_start:test_end].copy()

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
    """Identify stress weeks in each split based on irradiance, price, temperature, and pressure."""
    print("\n[STRESS] Identifying stress weeks...")

    stress_results = {}
    for name, split in splits.items():
        split_copy = split.copy()
        split_copy["week_label"] = (
            split_copy.index.isocalendar().year.astype(str).values
            + "-W"
            + split_copy.index.isocalendar().week.astype(str).str.zfill(2).values
        )

        weekly = split_copy.groupby("week_label").agg(
            {
                "irradiance_wm2": "sum",
                "price_eur_mwh": "mean",
                "temperature_c": "mean",
                "pressure_msl_hpa": "mean",
            }
        )

        lowest_pv_week = weekly["irradiance_wm2"].idxmin()
        highest_price_week = weekly["price_eur_mwh"].idxmax()
        hottest_week = weekly["temperature_c"].idxmax()
        stormiest_week = weekly["pressure_msl_hpa"].idxmin()

        stress = {
            "lowest_pv": {
                "week": lowest_pv_week,
                "value": float(weekly.loc[lowest_pv_week, "irradiance_wm2"]),
            },
            "highest_price": {
                "week": highest_price_week,
                "value": float(weekly.loc[highest_price_week, "price_eur_mwh"]),
            },
            "hottest": {
                "week": hottest_week,
                "value": float(weekly.loc[hottest_week, "temperature_c"]),
            },
            "stormiest": {
                "week": stormiest_week,
                "value": float(weekly.loc[stormiest_week, "pressure_msl_hpa"]),
            },
        }

        stress_results[name] = stress
        print(f"\n  {name.upper()} stress weeks:")
        for criterion, info in stress.items():
            print(f"    {criterion:15s}: {info['week']} (value={info['value']:.1f})")

    return stress_results


def report_split_statistics(splits: dict):
    """Report basic statistics per split."""
    print("\n[STATISTICS] Per-split summary:")
    numeric_cols = [
        "load_kw",
        "price_eur_mwh",
        "irradiance_wm2",
        "temperature_c",
        "humidity_pct",
        "pressure_msl_hpa",
    ]
    for name, split in splits.items():
        print(f"\n  {name.upper()}:")
        for col in numeric_cols:
            if col in split.columns:
                print(
                    f"    {col:25s}: mean={split[col].mean():8.2f}, "
                    f"std={split[col].std():8.2f}, "
                    f"min={split[col].min():8.2f}, max={split[col].max():8.2f}"
                )
        neg = int((split["price_eur_mwh"] < 0).sum())
        n_event_hours = int(split["stadium_event"].sum())
        print(f"    Negative-price hours: {neg}")
        print(f"    Price min: {split['price_eur_mwh'].min():.2f} EUR/MWh")
        print(f"    Active stadium event hours: {n_event_hours}")


def _grid_only_import(split: pd.DataFrame, pv_scale: float) -> np.ndarray:
    """Import the site would draw with the battery idle: max(0, load - pv)."""
    pv_kw = split["irradiance_wm2"].to_numpy() * pv_scale
    return np.clip(split["load_kw"].to_numpy() - pv_kw, 0.0, None)


def compute_grid_import_cap(splits: dict, quantile: float) -> dict:
    """Derive the grid import cap from the training split's grid-only import."""
    pv_cfg = env_default_config()
    pv_scale = (
        pv_cfg["pv_area_m2"]
        * pv_cfg["pv_efficiency"]
        * pv_cfg["pv_performance_ratio"]
        / 1000.0
    )

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
    for name in profile:
        print(
            f"  {name:5s}: {report['exceedance'][name] * 100:5.1f}% of hours above, "
            f"peak {report['max_net_import_kw'][name]:.1f} kW"
        )
    return report


def save_splits(
    splits: dict,
    stress_results: dict,
    grid_import_cap: dict,
    prep_cfg: dict,
):
    """Save processed data splits and metadata report."""
    print("\n[SAVE] Writing processed data...")

    drop_cols = ["load_mw", "event_warmup"]

    for name, split in splits.items():
        save_df = split.drop(columns=[c for c in drop_cols if c in split.columns])
        path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
        save_df.to_parquet(path)
        print(f"  {path} ({len(save_df)} rows)")

        csv_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        save_df.to_csv(csv_path)

    zone_shares = {
        zone_name: float(zone_cfg["share"])
        for zone_name, zone_cfg in prep_cfg["zones"].items()
    }
    report = {
        "splits": {},
        "stress_weeks": stress_results,
        "microgrid_peak_kw": float(prep_cfg["microgrid_peak_kw"]),
        "local_timezone": prep_cfg["local_timezone"],
        "zone_shares": zone_shares,
        "zone_profiles": prep_cfg["zones"],
        "events_config": prep_cfg["events"],
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
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare smartgrid-rl dataset splits.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file containing `data_preparation`.",
    )
    parser.add_argument(
        "--price-source",
        default=None,
        help="Override price source (e.g., proxy_DE or entsoe_FR).",
    )
    parser.add_argument(
        "--cap-quantile",
        type=float,
        default=None,
        help="Override grid import cap quantile (default from config: 0.90).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed for synthetic event generation.",
    )
    return parser.parse_args()


def main():
    print("=" * 60)
    print("smartgrid-rl: Data Preparation")
    print("=" * 60)

    args = parse_args()
    prep_cfg = load_preparation_config(args.config)
    if args.price_source is not None:
        prep_cfg["price_source"] = args.price_source
    if args.cap_quantile is not None:
        prep_cfg["cap_quantile"] = args.cap_quantile
    if args.seed is not None:
        prep_cfg["seed"] = args.seed

    cap_quantile = float(prep_cfg["cap_quantile"])

    print(f"\n[1/8] Loading OPSD (price_source={prep_cfg['price_source']})...")
    opsd = load_opsd(prep_cfg)

    print("\n[2/8] Loading PVGIS...")
    pvgis = load_pvgis()

    print("\n[3/8] Loading Open-Meteo...")
    weather = load_openmeteo()

    print("\n[4/8] Merging datasets...")
    df = merge_datasets(opsd, pvgis, weather, prep_cfg)

    print("\n[5/8] Scaling load to microgrid...")
    df = scale_load_to_microgrid(df, prep_cfg)

    print("\n[6/8] Generating stadium event calendar...")
    df = generate_event_calendar(df, prep_cfg)

    print("\n[7/8] Disaggregating zone loads...")
    df = apply_zone_disaggregation(df, prep_cfg)

    print("\n[8/8] Splitting dataset and computing cap...")
    splits = split_dataset(df, prep_cfg)
    stress = find_stress_weeks(splits)
    report_split_statistics(splits)
    grid_import_cap = compute_grid_import_cap(splits, cap_quantile)

    save_splits(splits, stress, grid_import_cap, prep_cfg)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
