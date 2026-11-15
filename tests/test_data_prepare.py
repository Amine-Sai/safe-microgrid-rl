"""Unit tests for data preparation module (data/prepare.py).

Tests verify:
1. `compute_pv_scale` matches `env_default_config()` canonically and avoids stale defaults.
2. `find_stress_weeks` accepts float, dict (from compute_grid_import_cap or explicit keys), and defaults.
3. Equivalence between dict-based and float-based cap dispatch in `find_stress_weeks`.
4. Stale pv_scale default is replaced with canonical `env_default_config()` scale.
"""

import numpy as np
import pandas as pd
import pytest

from data.prepare import (
    compute_grid_import_cap,
    compute_pv_scale,
    find_stress_weeks,
)
from env.smartgrid_env import default_config as env_default_config


@pytest.fixture
def synthetic_split() -> pd.DataFrame:
    """Generate 2 full ISO weeks (336 hours) of synthetic microgrid time series."""
    idx = pd.date_range("2017-01-02 00:00:00", periods=336, freq="h")
    rng = np.random.RandomState(42)

    return pd.DataFrame(
        {
            "load_kw": rng.uniform(100.0, 450.0, size=len(idx)),
            "irradiance_wm2": rng.uniform(0.0, 900.0, size=len(idx)),
            "price_eur_mwh": rng.uniform(10.0, 80.0, size=len(idx)),
            "temperature_c": rng.uniform(5.0, 30.0, size=len(idx)),
            "pressure_msl_hpa": rng.uniform(990.0, 1030.0, size=len(idx)),
        },
        index=idx,
    )


def test_compute_pv_scale_canonical_default():
    """Verify compute_pv_scale uses env_default_config and rejects stale default."""
    cfg = env_default_config()
    expected = (
        cfg["pv_area_m2"]
        * cfg["pv_efficiency"]
        * cfg["pv_performance_ratio"]
        / 1000.0
    )

    # Canonical default from env: 1471 * 0.20 * 0.85 / 1000 = 0.25007
    assert compute_pv_scale() == pytest.approx(expected)
    assert compute_pv_scale(None) == pytest.approx(expected)
    assert compute_pv_scale({}) == pytest.approx(expected)
    assert compute_pv_scale() == pytest.approx(0.25007)

    # Stale legacy default: 1500 * 0.18 * 0.85 / 1000 = 0.2295
    stale_pv_scale = 1500.0 * 0.18 * 0.85 / 1000.0
    assert compute_pv_scale() != pytest.approx(stale_pv_scale)


def test_compute_pv_scale_with_overrides():
    """Verify compute_pv_scale respects partial and full prep_cfg overrides."""
    cfg = env_default_config()

    # Partial override of pv_area_m2
    res = compute_pv_scale({"pv_area_m2": 2000.0})
    expected = 2000.0 * cfg["pv_efficiency"] * cfg["pv_performance_ratio"] / 1000.0
    assert res == pytest.approx(expected)

    # Full override of all PV parameters
    custom_cfg = {
        "pv_area_m2": 1000.0,
        "pv_efficiency": 0.25,
        "pv_performance_ratio": 0.80,
    }
    assert compute_pv_scale(custom_cfg) == pytest.approx(1000.0 * 0.25 * 0.80 / 1000.0)


def test_find_stress_weeks_accepts_float(synthetic_split):
    """Verify find_stress_weeks runs with a float grid_import_cap."""
    splits = {"train": synthetic_split}
    stress = find_stress_weeks(splits, grid_import_cap=350.0)

    assert "train" in stress
    for metric in ["lowest_pv", "highest_price", "hottest", "stormiest", "highest_exceedance"]:
        assert metric in stress["train"]
        assert "week" in stress["train"][metric]
        assert isinstance(stress["train"][metric]["value"], float)


def test_find_stress_weeks_accepts_dict_from_compute_grid_import_cap(synthetic_split):
    """Verify find_stress_weeks accepts dict from compute_grid_import_cap identically to float."""
    splits = {"train": synthetic_split, "val": synthetic_split}

    cap_report = compute_grid_import_cap(splits, quantile=0.90)
    assert isinstance(cap_report, dict)
    assert "value_kw" in cap_report
    cap_float = cap_report["value_kw"]

    # Call with dict (main prepare.py flow) and with extracted float
    res_from_dict = find_stress_weeks(splits, grid_import_cap=cap_report)
    res_from_float = find_stress_weeks(splits, grid_import_cap=cap_float)

    assert res_from_dict == res_from_float


def test_find_stress_weeks_accepts_dict_alternative_key(synthetic_split):
    """Verify find_stress_weeks accepts dict with 'grid_import_cap_kw'."""
    splits = {"train": synthetic_split}
    res_dict = find_stress_weeks(splits, grid_import_cap={"grid_import_cap_kw": 368.15})
    res_float = find_stress_weeks(splits, grid_import_cap=368.15)

    assert res_dict == res_float


def test_find_stress_weeks_invalid_dict_raises(synthetic_split):
    """Verify find_stress_weeks raises KeyError when dict misses expected keys."""
    splits = {"train": synthetic_split}
    with pytest.raises(KeyError, match="must contain 'value_kw' or 'grid_import_cap_kw'"):
        find_stress_weeks(splits, grid_import_cap={"invalid_key": 368.15})


def test_find_stress_weeks_defaults_and_none(synthetic_split):
    """Verify find_stress_weeks handles default parameter and explicit None cleanly."""
    splits = {"train": synthetic_split}
    res_default = find_stress_weeks(splits)
    res_none = find_stress_weeks(splits, grid_import_cap=None)
    res_float = find_stress_weeks(splits, grid_import_cap=368.15)

    assert res_default == res_float
    assert res_none == res_float


def test_find_stress_weeks_uses_canonical_pv_scale():
    """Verify find_stress_weeks uses canonical pv_scale rather than stale default in exceedance."""
    idx = pd.date_range("2017-01-02 00:00:00", periods=168, freq="h")
    # At irradiance=1000 W/m2 and load=240 kW:
    # Canonical pv_scale (0.25007): pv_kw = 250.07 kW > load (240 kW) -> net_load = 0 kW -> cap exceedance = 0 kW
    # Stale pv_scale (0.22950): pv_kw = 229.50 kW < load (240 kW) -> net_load = 10.5 kW -> cap exceedance > 0 kW
    df = pd.DataFrame(
        {
            "load_kw": [240.0] * 168,
            "irradiance_wm2": [1000.0] * 168,
            "price_eur_mwh": [50.0] * 168,
            "temperature_c": [20.0] * 168,
            "pressure_msl_hpa": [1013.0] * 168,
        },
        index=idx,
    )
    splits = {"train": df}
    stress = find_stress_weeks(splits, grid_import_cap=5.0)

    # Exceedance must be exactly 0 under canonical pv_scale
    assert stress["train"]["highest_exceedance"]["value"] == pytest.approx(0.0)
