import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _create_test_data():
    """Create minimal synthetic parquet for testing."""
    import pandas as pd
    data_dir = os.path.join(os.path.dirname(__file__), "test_data", "processed")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "train.parquet")
    if os.path.exists(path):
        return data_dir
    n = 168 * 4
    rng = np.random.RandomState(123)
    ts = pd.date_range("2019-01-01", periods=n, freq="h")
    h = ts.hour
    load = np.clip(300 + 100 * np.sin(2 * np.pi * h / 24 - np.pi / 2) + rng.normal(0, 20, n), 50, 500)
    irr = np.clip(np.where((h >= 6) & (h <= 18), 800 * np.sin(np.pi * (h - 6) / 12), 0) + rng.normal(0, 50, n), 0, 1000)
    price = np.clip(40 + 20 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 5, n), -10, 200)
    zones = {"residential": 0.35, "commercial": 0.25, "stadium": 0.10, "street": 0.05, "industrial": 0.25}
    d = {"load_kw": load, "price_eur_mwh": price, "irradiance_wm2": irr,
         "temperature_c": 10 + rng.normal(0, 2, n), "humidity_pct": np.clip(60 + rng.normal(0, 10, n), 20, 100),
         "pressure_msl_hpa": 1013 + rng.normal(0, 8, n), "stadium_event": np.zeros(n, dtype=bool),
         "event_phase": ["NONE"] * n, "day_period": ["DAY"] * n, "weather_rule": ["NORMAL"] * n}
    for z, s in zones.items():
        d[f"load_{z}_kw"] = load * s
    df = pd.DataFrame(d, index=ts); df.index.name = "timestamp"
    df.to_parquet(path)
    with open(os.path.join(data_dir, "split_report.json"), "w") as f:
        json.dump({"splits": {}, "microgrid_peak_kw": 500.0}, f)
    return data_dir


def make_env(seed=42, **kw):
    from env.smartgrid_env import SmartGridEnv
    data_dir = _create_test_data()
    # The synthetic report has no derived cap, so pin one explicitly: tests must not
    # depend on data/processed/split_report.json (which reflects the real dataset).
    cfg = {"data_path": data_dir, "split": "train", "seed": seed,
           "episode_length_hours": 48, "grid_import_cap_kw": 400.0}
    cfg.update(kw)
    return SmartGridEnv(cfg)


def test_energy_balance():
    """PV + grid + battery_discharge = served + battery_charge + curtailment."""
    env = make_env()
    env.reset(seed=42)
    for _ in range(24):
        obs, _, _, _, info = env.step(env.action_space.sample())
        net = info["total_served_kw"] + info["battery_power_kw"] - info["pv_kw"]
        assert abs(info["grid_import_kw"] - max(0.0, net)) < 1e-4
        assert abs(info["pv_curtailed_kw"] - max(0.0, -net)) < 1e-4


def test_battery_bounds_extreme():
    """SoC stays in [soc_min, soc_max] under extreme charge/discharge."""
    env = make_env()
    env.reset(seed=42)
    for i in range(48):
        a = np.zeros(6, dtype=np.float32)
        a[0] = 1.0 if i % 2 == 0 else -1.0
        _, _, _, _, info = env.step(a)
        assert env.bat_soc_min - 1e-6 <= info["soc"] <= env.bat_soc_max + 1e-6


def test_truncated_not_terminated():
    """Episode always ends with truncated=True, terminated=False at step 168."""
    env = make_env()
    env.reset(seed=42)
    for step in range(env.episode_length):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert terminated is False
        if step < env.episode_length - 1:
            assert truncated is False
    assert truncated is True


def test_industrial_off_or_at_least_70():
    """Industrial shed: either off (1.0) or <=0.30 (motor >= 70%). Continuous off reachable."""
    env = make_env()
    env.reset(seed=42)
    for _ in range(48):
        _, _, _, _, info = env.step(env.action_space.sample())
        s = info["shed_fractions"][4]
        assert s <= 0.30 + 1e-6 or abs(s - 1.0) < 1e-6
    # Verify off is reachable with action = 1.0
    env.reset(seed=42)
    a = np.array([0, 0, 0, 0, 0, 1.0], dtype=np.float32)
    _, _, _, _, info = env.step(a)
    assert abs(info["shed_fractions"][4] - 1.0) < 1e-6


def test_determinism():
    """Same seed, same actions -> same trajectory."""
    def run(seed):
        env = make_env(seed=seed)
        env.reset(seed=seed)
        rs = []
        for _ in range(24):
            _, r, _, _, _ = env.step(np.array([0.3, -0.2, 0.1, 0, -0.5, 0.4], dtype=np.float32))
            rs.append(r)
        return rs
    np.testing.assert_array_almost_equal(run(42), run(42), decimal=6)


def test_constraint_cost():
    """Cost IS the grid exceedance in kW (never a mixed kW/kWh quantity)."""
    env = make_env(grid_import_cap_kw=10.0)
    env.reset(seed=42)
    any_violation = False
    for _ in range(48):
        a = np.array([0, -1, -1, -1, -1, -1], dtype=np.float32)  # No shed
        _, _, _, _, info = env.step(a)
        assert info["cost"] >= 0.0
        assert np.isclose(info["cost"], info["grid_exceedance_kw"])
        assert np.isclose(info["cost"], info["grid_violation_kw"])
        assert np.isclose(info["cost"], max(0.0, info["grid_import_kw"] - env.grid_import_cap_kw))
        if info["cost"] > 0.0:
            any_violation = True
            assert info["grid_import_kw"] > 10.0
    assert any_violation


def test_cost_ignores_soc_deficit():
    """A SoC reserve breach must NOT enter info["cost"]; it is a kWh diagnostic only."""
    # Huge cap so the grid constraint can never bind.
    env = make_env(grid_import_cap_kw=100000.0)
    env.reset(seed=42)
    saw_deficit = False
    for _ in range(6):
        # Full discharge, no shedding: drives SoC below the 0.20 reserve floor.
        _, _, _, _, info = env.step(np.array([-1.0, -1, -1, -1, -1, -1.0], dtype=np.float32))
        assert np.isclose(info["cost"], max(0.0, info["grid_import_kw"] - env.grid_import_cap_kw))
        if info["soc_deficit_kwh"] > 0.0:
            saw_deficit = True
            assert info["cost"] == 0.0, "SoC deficit leaked into the Lagrangian constraint cost"
            assert info["soc_violation"] > 0.0
            assert np.isclose(info["soc_deficit_kwh"],
                              (env.soc_reserve_floor - info["soc"]) * env.bat_capacity_kwh)
    assert saw_deficit, "test setup never drove SoC below the 0.20 reserve floor"


def test_shed_action_mapping_linear():
    """Pin the shed mapping as implemented: s = (a+1)/2, no dead zone.

    env/smartgrid_env.py:183-194. For a=-1 -> s=0 (no shed); a=0 -> s=0.5;
    a=1 -> s=1.0. Industrial: off if s>0.9 else 0.3*s/0.9.
    A dead-zone variant s=max(0,a) was considered and not adopted; this test
    pins the mapping above so that change cannot slip in unnoticed.
    """
    env = make_env(grid_import_cap_kw=100000.0)
    env.reset(seed=42)
    cases = [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0)]
    for a_val, expected_s in cases:
        a = np.array([0.0, a_val, a_val, a_val, a_val, a_val], dtype=np.float32)
        _, _, _, _, info = env.step(a)
        sf = np.array(info["shed_fractions"])  # 5 zone shed fractions
        assert np.allclose(sf[:4], expected_s, atol=1e-6), f"a={a_val}: zones0-3 shed {sf[:4]} != {expected_s}"
    # industrial: a=0.8 -> s=0.9 -> not >0.9 -> shed 0.3; a=0.81 -> s=0.905>0.9 -> off
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.array([0.0, 0, 0, 0, 0, 0.8], dtype=np.float32))
    assert np.isclose(info["shed_fractions"][4], 0.3, atol=1e-6)
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.array([0.0, 0, 0, 0, 0, 0.81], dtype=np.float32))
    assert np.isclose(info["shed_fractions"][4], 1.0, atol=1e-6)


def test_episode_start_window_within_split():
    """A sampled Train episode window can never cross the split boundary.

    reset() draws start_idx = rng.randint(0, n_timesteps - 168). This test proves that
    for every admissible start index i, i + 168 <= n, so no window reads past the end of
    the split (and therefore cannot reach into Gap 1 or any other split).
    """
    env = make_env(episode_length_hours=168, data_path=os.path.join(
        os.path.dirname(__file__), "test_data", "processed"), split="train")
    n = env._n_timesteps
    assert n >= 168
    max_start = n - env.episode_length
    assert max_start % 168 == 0, "max start index is a whole-week multiple, so windows are week-aligned"

    # Exhaustive over every admissible start index (cheap: no env.stepping needed).
    for i in range(max_start):
        assert i + env.episode_length <= n
    assert max_start + env.episode_length == n

    # Observed behaviour: starts are arbitrary HOUR offsets, not Monday-00:00 aligned.
    offsets = set()
    for s in range(40):
        _, _ = env.reset(seed=s)
        offsets.add(int(env._start_idx % 168))
    assert len(offsets) > 1, "reset() returns week-aligned starts; arbitrary hour offsets expected"

    # Boundary: the last admissible start must produce a full episode without IndexError.
    obs, _ = env.reset(options={"start_idx": max_start})
    for _ in range(env.episode_length):
        obs, _, _, trunc, _ = env.step(np.zeros(6, dtype=np.float32))
    assert trunc is True


def test_bootstrap_not_zeroed_on_truncation():
    """Truncation must NOT zero the bootstrap (only termination should)."""
    # In this env terminated is always False, so bootstrapping must continue.
    env = make_env()
    env.reset(seed=42)
    for step in range(env.episode_length):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    # At truncation, terminated must be False (bootstrap continues)
    assert truncated is True
    assert terminated is False


def test_cap_resolved_from_split_report(tmp_path):
    """With no explicit cap, the env reads the value derived by data/prepare.py."""
    from env.smartgrid_env import resolve_grid_import_cap
    with open(tmp_path / "split_report.json", "w") as f:
        json.dump({"grid_import_cap": {"value_kw": 123.4, "quantile": 0.9}}, f)
    assert resolve_grid_import_cap({"data_path": str(tmp_path)}) == 123.4


def test_cap_explicit_override_wins(tmp_path):
    """An explicit config value beats the prepared report."""
    from env.smartgrid_env import resolve_grid_import_cap
    with open(tmp_path / "split_report.json", "w") as f:
        json.dump({"grid_import_cap": {"value_kw": 123.4}}, f)
    cfg = {"data_path": str(tmp_path), "grid_import_cap_kw": 250.0}
    assert resolve_grid_import_cap(cfg) == 250.0


def test_cap_missing_raises(tmp_path):
    """No explicit cap and no report entry is an error, never a silent fallback."""
    from env.smartgrid_env import resolve_grid_import_cap
    with pytest.raises(ValueError, match="grid_import_cap_kw"):
        resolve_grid_import_cap({"data_path": str(tmp_path)})
    with open(tmp_path / "split_report.json", "w") as f:
        json.dump({"grid_import_cap": {}}, f)
    with pytest.raises(ValueError, match="grid_import_cap_kw"):
        resolve_grid_import_cap({"data_path": str(tmp_path)})


def test_cap_shield_defaults_to_env_cap():
    """CapShield must shave to the env's cap, not to a stale module constant."""
    from env.cap_shield import CapShield
    env = make_env()  # synthetic cap = 400.0
    assert CapShield(env).cap_kw == 400.0
    assert CapShield(env, cap_kw=50.0).cap_kw == 50.0


def test_persisted_cap_matches_recomputed_p90():
    """split_report.json must agree with a fresh derivation from the train split."""
    import pandas as pd
    from env.smartgrid_env import default_config

    cfg = default_config()
    data_dir = cfg["data_path"]
    csv_path = os.path.join(data_dir, "train.csv")
    report_path = os.path.join(data_dir, "split_report.json")
    if not (os.path.exists(csv_path) and os.path.exists(report_path)):
        pytest.skip("processed data not available")

    with open(report_path) as f:
        report = json.load(f)
    derived = report.get("grid_import_cap")
    if not derived:
        pytest.skip("split_report.json predates the derived cap; re-run data/prepare.py")

    df = pd.read_csv(csv_path)
    pv_kw = (df["irradiance_wm2"] * cfg["pv_area_m2"] * cfg["pv_efficiency"]
             * cfg["pv_performance_ratio"] / 1000.0)
    net_import = (df["load_kw"] - pv_kw).clip(lower=0.0)
    expected = round(float(np.quantile(net_import, derived["quantile"])), 6)

    assert derived["value_kw"] == pytest.approx(expected, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-x"])
