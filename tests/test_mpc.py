"""Unit and integration tests for receding-horizon MPCPolicy (Task T6).

Verifies:
- Valid 6-dimensional continuous actions in [-1.0, 1.0]^6.
- Strict 0 cap violations across stress-week episodes.
- Lower economic operational cost than GridOnlyPolicy on Val and Test weeks.
- Support for forecast_mode='perfect', forecast_mode='persistence', and Gaussian forecast noise.
- Per-step solve latency tracking in milliseconds (last_solve_ms, solve_times_ms).
- Dynamic parameter resolution without hardcoded magic numbers.
- HiGHS LP relaxation and exact MILP modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.heuristics import GridOnlyPolicy
from agents.mpc import MPCPolicy, mpc_prime_policy
from env.smartgrid_env import SmartGridEnv, default_config


@pytest.fixture
def val_env() -> SmartGridEnv:
    """Fixture providing a standard validation environment."""
    return SmartGridEnv({"split": "val", "seed": 42})


@pytest.fixture
def test_env() -> SmartGridEnv:
    """Fixture providing a standard test environment."""
    return SmartGridEnv({"split": "test", "seed": 42})


def test_mpc_action_space_and_bounds(val_env: SmartGridEnv) -> None:
    """Verify MPCPolicy produces valid 6-D continuous actions in [-1, 1]."""
    policy = MPCPolicy(env=val_env, horizon=24, forecast_mode="perfect")
    obs, info = val_env.reset(options={"start_idx": 0})

    action = policy.get_action(obs, info)
    assert isinstance(action, np.ndarray)
    assert action.shape == (6,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    # Test alias interfaces
    act_res = policy.act(obs, info)
    assert np.allclose(action, act_res)

    pred_action, state = policy.predict(obs)
    assert np.allclose(action, pred_action)
    assert state is None


def test_mpc_solve_latency_tracking(val_env: SmartGridEnv) -> None:
    """Verify solve latency in ms is measured and tracked per step."""
    policy = MPCPolicy(env=val_env, horizon=24, forecast_mode="perfect")
    obs, info = val_env.reset(options={"start_idx": 0})

    assert policy.last_solve_ms == 0.0
    assert len(policy.solve_times_ms) == 0

    num_steps = 5
    for _ in range(num_steps):
        action = policy.get_action(obs, info)
        obs, _, _, _, info = val_env.step(action)
        assert policy.last_solve_ms > 0.0

    assert len(policy.solve_times_ms) == num_steps
    assert policy.mean_solve_latency_ms > 0.0
    assert policy.max_solve_latency_ms >= policy.mean_solve_latency_ms
    assert policy.total_solve_latency_ms > 0.0

    # Verify reset clears latency history
    policy.reset(val_env)
    assert policy.last_solve_ms == 0.0
    assert len(policy.solve_times_ms) == 0
    assert policy.mean_solve_latency_ms == 0.0


def test_mpc_forecast_modes(val_env: SmartGridEnv) -> None:
    """Verify perfect, persistence, and noisy forecast modes."""
    obs, info = val_env.reset(options={"start_idx": 168})

    # 1. Perfect forecast mode
    pol_perfect = MPCPolicy(env=val_env, horizon=24, forecast_mode="perfect")
    act_perf = pol_perfect.get_action(obs, info)
    assert act_perf.shape == (6,)
    assert np.all(act_perf >= -1.0) and np.all(act_perf <= 1.0)

    # 2. Persistence forecast mode
    pol_persist = MPCPolicy(env=val_env, horizon=24, forecast_mode="persistence")
    act_pers = pol_persist.get_action(obs, info)
    assert act_pers.shape == (6,)
    assert np.all(act_pers >= -1.0) and np.all(act_pers <= 1.0)

    # Persistence can also run purely from obs without env
    pol_standalone = MPCPolicy(horizon=24, forecast_mode="persistence")
    act_standalone = pol_standalone.get_action(obs)
    assert act_standalone.shape == (6,)
    assert np.all(act_standalone >= -1.0) and np.all(act_standalone <= 1.0)

    # 3. Noisy forecast mode (additive and relative Gaussian noise)
    pol_noisy_add = MPCPolicy(
        env=val_env,
        horizon=24,
        forecast_mode="perfect",
        forecast_noise_std=10.0,
        forecast_noise_relative=False,
        seed=123,
    )
    act_noisy_add = pol_noisy_add.get_action(obs, info)
    assert act_noisy_add.shape == (6,)
    assert np.all(act_noisy_add >= -1.0) and np.all(act_noisy_add <= 1.0)

    pol_noisy_rel = MPCPolicy(
        env=val_env,
        horizon=24,
        forecast_mode="perfect",
        forecast_noise_std=0.10,
        forecast_noise_relative=True,
        seed=456,
    )
    act_noisy_rel = pol_noisy_rel.get_action(obs, info)
    assert act_noisy_rel.shape == (6,)
    assert np.all(act_noisy_rel >= -1.0) and np.all(act_noisy_rel <= 1.0)

    # 4. Invalid forecast mode raises ValueError
    pol_invalid = MPCPolicy(env=val_env, horizon=24, forecast_mode="invalid_mode")
    with pytest.raises(ValueError, match="Unknown forecast_mode"):
        pol_invalid.get_action(obs, info)


def test_mpc_zero_cap_violations_across_week_episode(val_env: SmartGridEnv) -> None:
    """Verify MPCPolicy maintains exact 0 cap violations across a full 168-hour stress week."""
    # Val week 0 has raw net load peaking at ~448 kW, far exceeding the 368 kW cap
    policy = MPCPolicy(env=val_env, horizon=24, forecast_mode="perfect")
    obs, info = val_env.reset(options={"start_idx": 0})
    policy.reset(val_env)

    max_exceedance = 0.0
    violation_hours = 0
    done = False

    while not done:
        action = policy.get_action(obs, info)
        obs, _, term, trunc, info = val_env.step(action)
        exc = float(info["grid_exceedance_kw"])
        max_exceedance = max(max_exceedance, exc)
        if info.get("is_violation", exc > val_env.violation_tolerance_kw):
            violation_hours += 1
        done = term or trunc

    assert max_exceedance <= val_env.violation_tolerance_kw, (
        f"Max exceedance {max_exceedance} kW exceeds tolerance {val_env.violation_tolerance_kw}"
    )
    assert violation_hours == 0, f"Expected 0 violation hours, got {violation_hours}"


def test_mpc_lower_cost_than_grid_only_val_week(val_env: SmartGridEnv) -> None:
    """Verify MPCPolicy achieves lower operational cost than GridOnlyPolicy on a validation week."""
    # Val week 8 has low exceedance but active battery price arbitrage potential
    start_idx = 8 * 168

    # GridOnlyPolicy evaluation
    p_grid = GridOnlyPolicy()
    obs, _ = val_env.reset(options={"start_idx": start_idx})
    grid_cost = 0.0
    done = False
    while not done:
        obs, _, term, trunc, info = val_env.step(p_grid.get_action(obs))
        grid_cost += (
            info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
        )
        done = term or trunc

    # MPCPolicy evaluation (perfect forecast mode)
    p_mpc = MPCPolicy(env=val_env, horizon=24, forecast_mode="perfect")
    obs, info = val_env.reset(options={"start_idx": start_idx})
    p_mpc.reset(val_env)
    mpc_cost = 0.0
    max_exc = 0.0
    done = False
    while not done:
        action = p_mpc.get_action(obs, info)
        obs, _, term, trunc, info = val_env.step(action)
        mpc_cost += (
            info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
        )
        max_exc = max(max_exc, float(info["grid_exceedance_kw"]))
        done = term or trunc

    assert max_exc <= val_env.violation_tolerance_kw
    assert mpc_cost < grid_cost, f"MPC cost {mpc_cost:.2f} >= GridOnly {grid_cost:.2f}"

    # Also check persistence mode achieves lower cost than GridOnly on this week
    p_persist = MPCPolicy(env=val_env, horizon=24, forecast_mode="persistence")
    obs, info = val_env.reset(options={"start_idx": start_idx})
    p_persist.reset(val_env)
    persist_cost = 0.0
    done = False
    while not done:
        action = p_persist.get_action(obs, info)
        obs, _, term, trunc, info = val_env.step(action)
        persist_cost += (
            info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
        )
        done = term or trunc

    assert persist_cost < grid_cost, (
        f"Persistence MPC cost {persist_cost:.2f} >= GridOnly {grid_cost:.2f}"
    )


def test_mpc_lower_cost_than_grid_only_test_week(test_env: SmartGridEnv) -> None:
    """Verify MPCPolicy achieves lower operational cost than GridOnlyPolicy on a test week."""
    start_idx = 0  # Test week 0

    p_grid = GridOnlyPolicy()
    obs, _ = test_env.reset(options={"start_idx": start_idx})
    grid_cost = 0.0
    done = False
    while not done:
        obs, _, term, trunc, info = test_env.step(p_grid.get_action(obs))
        grid_cost += (
            info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
        )
        done = term or trunc

    p_mpc = MPCPolicy(env=test_env, horizon=24, forecast_mode="perfect")
    obs, info = test_env.reset(options={"start_idx": start_idx})
    p_mpc.reset(test_env)
    mpc_cost = 0.0
    done = False
    while not done:
        action = p_mpc.get_action(obs, info)
        obs, _, term, trunc, info = test_env.step(action)
        mpc_cost += (
            info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
        )
        done = term or trunc

    assert mpc_cost < grid_cost, f"MPC cost {mpc_cost:.2f} >= GridOnly {grid_cost:.2f}"


def test_mpc_milp_mode(val_env: SmartGridEnv) -> None:
    """Verify MILP mode (use_milp=True) runs successfully and produces valid actions."""
    policy = MPCPolicy(env=val_env, horizon=12, use_milp=True)
    obs, info = val_env.reset(options={"start_idx": 0})
    policy.reset(val_env)

    for _ in range(5):
        action = policy.get_action(obs, info)
        assert action.shape == (6,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        obs, _, _, _, info = val_env.step(action)

    assert len(policy.solve_times_ms) == 5
    assert policy.mean_solve_latency_ms > 0.0


def test_mpc_dynamic_parameters_no_hardcoding() -> None:
    """Verify physical and cost parameters are pulled dynamically without hardcoded constants."""
    custom_cfg = default_config()
    custom_cfg["battery_capacity_kwh"] = 800.0
    custom_cfg["battery_max_power_kw"] = 400.0
    custom_cfg["grid_import_cap_kw"] = 250.0
    custom_cfg["cost_unserved_eur_kwh"] = 2.5
    custom_cfg["cost_wear_eur_kwh"] = 0.05
    custom_cfg["battery_eta_charge"] = 0.90
    custom_cfg["battery_eta_discharge"] = 0.92

    custom_env = SmartGridEnv(custom_cfg)
    policy = MPCPolicy(env=custom_env)

    assert policy.c_bat == 800.0
    assert policy.p_max == 400.0
    assert policy.cap == 250.0
    assert policy.w_uns == 2.5
    assert policy.w_wear == 0.05
    assert policy.eta_c == 0.90
    assert policy.eta_d == 0.92

    # Factory helper
    prime_pol = mpc_prime_policy(env=custom_env, horizon=12)
    assert isinstance(prime_pol, MPCPolicy)
    assert prime_pol.horizon == 12
    assert prime_pol.cap == 250.0
