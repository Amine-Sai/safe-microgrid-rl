"""Tests for tail-risk metrics, shield interventions, latency, MPC baselines, and OOD stress suite."""

import hashlib
import os
import numpy as np
import pytest

from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
from agents.mpc import MPCPolicy
from env.cap_shield import CapShield
from env.smartgrid_env import SmartGridEnv
from tools.report.evaluate_test_split import (
    DEFAULT_OOD_CONTINGENCIES,
    OODPerturbationWrapper,
    ShieldIdlePolicy,
    compute_tail_and_safety_metrics,
    evaluate_mpc_baselines,
    evaluate_ood_stress_suite,
    evaluate_policy_on_split,
    evaluate_single_week,
    get_ood_contingency_config,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_LOCK_PATH = os.path.join(REPO_ROOT, "results", "runs", "test_lock.json")


def _read_lock_hash() -> str:
    with open(TEST_LOCK_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# =========================================================================
# === 1. Synthetic Tail-Risk & Safety Breach Helper Tests =================
# =========================================================================

def test_compute_tail_and_safety_metrics_synthetic():
    # 100 sample hourly costs: 1.0 to 100.0
    costs = list(range(1, 101))
    # 95th percentile is 95.05. Tail (>= 95.05) is 96, 97, 98, 99, 100
    # Mean of tail is (96 + 97 + 98 + 99 + 100) / 5 = 98.0
    excs = [0.0] * 95 + [10.0, 20.0, 30.0, 40.0, 50.0]
    # Tail of excs (>= 95th percentile ~10.0) is 10, 20, 30, 40, 50. Mean is 30.0
    # Consecutive violations: [0, 1, 1, 1, 0, 1, 1, 0] -> max run is 3
    flags = [False, True, True, True, False, True, True, False]

    metrics = compute_tail_and_safety_metrics(
        hourly_costs=costs,
        hourly_exceedances=excs,
        violation_flags=flags,
    )

    assert metrics["cvar95_cost_eur"] == 98.0
    assert metrics["cvar95_exceedance_kw"] == 30.0
    assert metrics["max_hourly_exceedance_kw"] == 50.0
    assert metrics["max_consecutive_violation_hours"] == 3


def test_compute_tail_and_safety_metrics_zero_and_empty():
    # Empty inputs
    empty_res = compute_tail_and_safety_metrics([], [])
    assert empty_res["cvar95_cost_eur"] == 0.0
    assert empty_res["cvar95_exceedance_kw"] == 0.0
    assert empty_res["max_hourly_exceedance_kw"] == 0.0
    assert empty_res["max_consecutive_violation_hours"] == 0

    # Zero exceedance trajectory
    zero_res = compute_tail_and_safety_metrics(
        hourly_costs=[25.0] * 50,
        hourly_exceedances=[0.0] * 50,
    )
    assert zero_res["cvar95_cost_eur"] == 25.0
    assert zero_res["cvar95_exceedance_kw"] == 0.0
    assert zero_res["max_hourly_exceedance_kw"] == 0.0
    assert zero_res["max_consecutive_violation_hours"] == 0


def test_max_consecutive_violation_hours_derived_from_exceedances():
    # Flags derived automatically from exceedance > tol
    excs = [0.0, 5.0, 8.0, 0.0, 2.0, 3.0, 4.0, 1.0, 0.0]
    res = compute_tail_and_safety_metrics(
        hourly_costs=[10.0] * len(excs),
        hourly_exceedances=excs,
        violation_tolerance_kw=1e-4,
    )
    # Consecutive runs: [5.0, 8.0] (len 2), [2.0, 3.0, 4.0, 1.0] (len 4)
    assert res["max_consecutive_violation_hours"] == 4
    assert res["max_hourly_exceedance_kw"] == 8.0


# =========================================================================
# === 2. Shield Intervention Rate & Delta L2 Tests ========================
# =========================================================================

def test_shield_intervention_metrics_unshielded_vs_shielded():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    grid_pol = GridOnlyPolicy()

    # Unshielded environment: shield intervention rate and delta L2 must be exactly 0
    res_unshielded = evaluate_single_week(env, start_idx=0, policy=grid_pol)
    assert res_unshielded["shield_intervention_rate"] == 0.0
    assert res_unshielded["shield_mean_delta_l2"] == 0.0

    # Shielded environment with CapShield
    shield_env = CapShield(SmartGridEnv({"split": "val", "charge_clamp": True}))
    shield_idle = ShieldIdlePolicy()
    res_shielded = evaluate_single_week(shield_env, start_idx=0, policy=shield_idle)

    assert "shield_intervention_rate" in res_shielded
    assert "shield_mean_delta_l2" in res_shielded
    assert 0.0 <= res_shielded["shield_intervention_rate"] <= 1.0
    assert res_shielded["shield_mean_delta_l2"] >= 0.0


def test_shield_intervention_positive_on_cap_exceeding_week():
    # Find a week where CapShield intervenes (e.g. demand exceeds cap)
    shield_env = CapShield(SmartGridEnv({"split": "val", "charge_clamp": True}))
    psm_pol = PeakShaveMinPolicy()

    # Rollout week 0
    res = evaluate_single_week(shield_env, start_idx=0, policy=psm_pol)
    assert res["mean_exceedance_kw"] == 0.0  # CapShield guarantees zero exceedance
    assert res["max_hourly_exceedance_kw"] == 0.0
    assert res["max_consecutive_violation_hours"] == 0


# =========================================================================
# === 3. Latency Statistics Tests (P50, P99, Mean) ========================
# =========================================================================

def test_latency_statistics_p50_p99_ordering():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    grid_pol = GridOnlyPolicy()

    res = evaluate_single_week(env, start_idx=0, policy=grid_pol)
    assert "latency_p50_ms" in res
    assert "latency_p99_ms" in res
    assert "latency_mean_ms" in res
    assert res["latency_p50_ms"] >= 0.0
    assert res["latency_p99_ms"] >= res["latency_p50_ms"]
    assert res["latency_mean_ms"] >= 0.0


def test_latency_compatibility_with_mpc_solver():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    # Fast short-horizon MPC policy (horizon=4)
    mpc_pol = MPCPolicy(env=env, horizon=4, forecast_mode="persistence")

    res = evaluate_single_week(env, start_idx=0, policy=mpc_pol)
    assert res["latency_p50_ms"] > 0.0
    assert res["latency_p99_ms"] >= res["latency_p50_ms"]
    assert len(mpc_pol.solve_times_ms) == 168


# =========================================================================
# === 4. OOD Contingency Suite Tests ======================================
# =========================================================================

def test_ood_contingency_forecast_noise():
    env = SmartGridEnv({"split": "val", "charge_clamp": True, "forecast_horizon_hours": 6})
    obs_clean, _ = env.reset(options={"start_idx": 0})

    cfg = get_ood_contingency_config("forecast_noise", {"noise_std": 0.50})
    with OODPerturbationWrapper(env, contingency="forecast_noise", contingency_config=cfg) as ood_env:
        obs_noisy, _ = ood_env.reset(options={"start_idx": 0})
        # Observation future slice [17:23] should be perturbed
        assert obs_noisy.shape == obs_clean.shape
        assert not np.allclose(obs_noisy[17:], obs_clean[17:], atol=1e-3)
        # Base physical state (SoC, PV, current loads) remains unchanged
        assert np.allclose(obs_noisy[:7], obs_clean[:7], atol=1e-5)


def test_ood_contingency_pv_inverter_trip_and_restoration():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    base_pv_orig = env._pv_kw_series.copy()

    cfg = get_ood_contingency_config("pv_inverter_trip", {"trip_start_hour": 11, "trip_end_hour": 15})
    with OODPerturbationWrapper(env, contingency="pv_inverter_trip", contingency_config=cfg) as ood_env:
        import pandas as pd
        hours = pd.DatetimeIndex(env._timestamps).hour.values
        peak_mask = (hours >= 11) & (hours <= 15)
        # Inverter tripped: PV must be 0 during peak hours
        assert np.all(ood_env._pv_kw_series[peak_mask] == 0.0)

    # Context exited: non-destructive restore must be bit-for-bit identical
    assert np.array_equal(env._pv_kw_series, base_pv_orig)


def test_ood_contingency_bess_derate_and_restoration():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    orig_power = float(env.bat_max_power_kw)
    assert orig_power == 250.0

    cfg = get_ood_contingency_config("bess_derate", {"derate_factor": 0.50})
    with OODPerturbationWrapper(env, contingency="bess_derate", contingency_config=cfg) as ood_env:
        assert ood_env.bat_max_power_kw == 125.0

    # Restored
    assert env.bat_max_power_kw == orig_power


def test_ood_contingency_grid_cap_reduction_and_restoration():
    shield_env = CapShield(SmartGridEnv({"split": "val", "charge_clamp": True}))
    orig_env_cap = float(shield_env.grid_import_cap_kw)
    orig_shield_cap = float(shield_env.cap_kw)

    cfg = get_ood_contingency_config("grid_cap_reduction", {"cap_factor": 0.80})
    with OODPerturbationWrapper(shield_env, contingency="grid_cap_reduction", contingency_config=cfg) as ood_env:
        assert ood_env.grid_import_cap_kw == pytest.approx(orig_env_cap * 0.80, rel=1e-5)
        assert ood_env.cap_kw == pytest.approx(orig_shield_cap * 0.80, rel=1e-5)

    # Restored
    assert shield_env.grid_import_cap_kw == orig_env_cap
    assert shield_env.cap_kw == orig_shield_cap


def test_evaluate_ood_stress_suite_all_four_contingencies():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    grid_pol = GridOnlyPolicy()

    results = evaluate_ood_stress_suite(env, policy=grid_pol, start_idx=0)
    expected_keys = ["nominal", "forecast_noise", "pv_inverter_trip", "bess_derate", "grid_cap_reduction"]
    for k in expected_keys:
        assert k in results, f"Missing contingency key {k} in stress suite results"
        res = results[k]
        assert "cost_eur" in res
        assert "cvar95_cost_eur" in res
        assert "cvar95_exceedance_kw" in res
        assert "max_hourly_exceedance_kw" in res
        assert "max_consecutive_violation_hours" in res
        assert "shield_intervention_rate" in res
        assert "latency_p50_ms" in res


# =========================================================================
# === 5. Receding-Horizon MPC Baseline Evaluation Hook ====================
# =========================================================================

def test_evaluate_mpc_baselines_hook():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    # Fast lookahead horizon=6
    results = evaluate_mpc_baselines(
        env,
        start_idx=0,
        forecast_modes=["perfect", "persistence"],
        horizon=6,
    )
    assert "MPC_perfect" in results
    assert "MPC_persistence" in results
    for mode in ["MPC_perfect", "MPC_persistence"]:
        res = results[mode]
        assert res["cost_eur"] > 0.0
        assert res["cvar95_cost_eur"] > 0.0
        assert res["mean_exceedance_kw"] >= 0.0
        assert res["latency_p50_ms"] > 0.0
        assert res["max_hourly_exceedance_kw"] >= 0.0


# =========================================================================
# === 6. Provenance & Non-Destructive Test-Lock Guard Test =================
# =========================================================================

def test_frozen_test_lock_json_unmutated_after_all_evaluations():
    # Check that test_lock.json exists and has not been mutated
    assert os.path.exists(TEST_LOCK_PATH)
    lock_hash = _read_lock_hash()
    assert len(lock_hash) == 64

    # Run evaluate_single_week and OOD suite again
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    evaluate_single_week(env, start_idx=0, policy=GridOnlyPolicy())
    evaluate_ood_stress_suite(env, policy=GridOnlyPolicy(), start_idx=0)

    # Hash must be bit-for-bit identical
    new_lock_hash = _read_lock_hash()
    assert new_lock_hash == lock_hash, "test_lock.json was modified during evaluation!"
