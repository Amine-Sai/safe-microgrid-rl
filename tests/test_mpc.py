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


def test_mpc_regime2_peak_demand_epigraph_parity() -> None:
    """Verify MPCPolicy optimizes epigraph P_peak >= env._week_peak_grid_kw cleanly."""
    ep_len = 6
    env = SmartGridEnv(
        {
            "split": "val",
            "charge_clamp": True,
            "episode_length_hours": ep_len,
            "battery_capacity_kwh": 200.0,
            "battery_max_power_kw": 60.0,
            "battery_soc_init": 0.50,
            "grid_import_cap_kw": 500.0,
            "cost_wear_eur_kwh": 0.02,
            "zone_voll_eur_kwh": {"residential": 0.5, "commercial": 5.0, "stadium": 5.0, "street": 5.0, "industrial": 5.0},
            "demand_charge_eur_per_kw_week": 10.0,
        }
    )
    env._zone_loads[0:ep_len, :] = 0.0
    env._zone_loads[0:ep_len, 0] = 100.0
    env._zone_loads[1, 0] = 200.0  # Spike that MPC should shave using battery + cheap VoLL if needed
    env._pv_kw_series[0:ep_len] = 0.0
    env._price[0:ep_len] = 10.0  # Below wear cost (0.02 EUR/kWh) so arbitrage alone won't discharge

    # Part A: When initial_peak is low (0.0), MPC shaves at the spike step.
    # When env._week_peak_grid_kw is already 200.0, MPC does NOT shed/discharge at a 120 kW step.
    for use_milp in (False, True):
        obs, info = env.reset(options={"start_idx": 0})
        env._zone_loads[0, 0] = 120.0
        obs = env._get_obs(0)
        pol = MPCPolicy(env=env, horizon=ep_len, forecast_mode="perfect", use_milp=use_milp)
        env._week_peak_grid_kw = 200.0
        act = pol.get_action(obs, info)
        _, _, _, _, step_info = env.step(act)
        # Because ratchet is already 200 kW > 120 kW, no shedding or discharge is warranted at t=0
        assert step_info["total_unserved_kw"] == pytest.approx(0.0, abs=1e-4)
        assert step_info["battery_power_kw"] >= -1e-4

    # Part B: Full-episode peak-shaving rollout reduces peak below 200 kW
    obs, info = env.reset(options={"start_idx": 0})
    pol_milp = MPCPolicy(env=env, horizon=ep_len, forecast_mode="perfect", use_milp=True)
    pol_milp.reset(env)
    done = False
    while not done:
        act = pol_milp.get_action(obs, info)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
    assert info["week_peak_grid_kw"] < 150.0


def test_mpc_regime2_zone_voll_sheds_lowest_voll_first() -> None:
    """Verify MPCPolicy sheds the lowest-VoLL zone first even when it differs from SHED_ORDER."""
    voll_cfg = {"residential": 0.15, "commercial": 2.0, "stadium": 3.0, "street": 4.0, "industrial": 5.0}
    env = SmartGridEnv(
        {
            "split": "val",
            "charge_clamp": True,
            "episode_length_hours": 4,
            "battery_max_power_kw": 0.0,
            "grid_import_cap_kw": 450.0,
            "zone_voll_eur_kwh": voll_cfg,
        }
    )
    env._zone_loads[0:4, :] = 100.0  # 500 kW total -> 50 kW deficit
    env._pv_kw_series[0:4] = 0.0
    env._price[0:4] = 50.0

    for use_milp in (False, True):
        obs, info = env.reset(options={"start_idx": 0})
        pol = MPCPolicy(env=env, horizon=4, forecast_mode="perfect", use_milp=use_milp)
        act = pol.get_action(obs, info)
        _, _, _, _, step_info = env.step(act)
        shed = step_info["shed_fractions"]
        assert step_info["grid_exceedance_kw"] <= env.violation_tolerance_kw
        assert shed[0] == pytest.approx(0.50, abs=1e-4)  # residential (VoLL=0.15) shed first
        assert all(s == pytest.approx(0.0, abs=1e-6) for s in shed[1:])


def test_mpc_regime2_industrial_trip_penalty_aware() -> None:
    """Verify MILP-MPC accounts for industrial_trip_penalty_eur and env._industrial_tripped."""
    voll_cfg = {"residential": 5.0, "commercial": 5.0, "stadium": 5.0, "street": 5.0, "industrial": 1.0}
    base_cfg = {
        "split": "val",
        "charge_clamp": True,
        "episode_length_hours": 4,
        "battery_max_power_kw": 0.0,
        "grid_import_cap_kw": 200.0,
        "zone_voll_eur_kwh": voll_cfg,
    }

    env_no_pen = SmartGridEnv({**base_cfg, "industrial_trip_penalty_eur": 0.0})
    env_pen = SmartGridEnv({**base_cfg, "industrial_trip_penalty_eur": 500.0})
    for e in (env_no_pen, env_pen):
        e._zone_loads[0:4, :] = 0.0
        e._zone_loads[0:4, 0] = 150.0  # residential
        e._zone_loads[0:4, 4] = 100.0  # industrial
        e._pv_kw_series[0:4] = 0.0
        e._price[0:4] = 50.0

    # Without trip penalty, MILP-MPC trips industrial (100 EUR < 130 EUR)
    obs, info = env_no_pen.reset(options={"start_idx": 0})
    pol_no_pen = MPCPolicy(env=env_no_pen, horizon=4, forecast_mode="perfect", use_milp=True)
    _, _, _, _, info_no_pen = env_no_pen.step(pol_no_pen.get_action(obs, info))
    assert info_no_pen["industrial_tripped"] is True

    # With 500 EUR restart penalty from non-tripped state, MILP-MPC avoids tripping
    obs, info = env_pen.reset(options={"start_idx": 0})
    pol_pen = MPCPolicy(env=env_pen, horizon=4, forecast_mode="perfect", use_milp=True)
    _, _, _, _, info_pen = env_pen.step(pol_pen.get_action(obs, info))
    assert info_pen["industrial_tripped"] is False
    assert info_pen["cost_industrial_trip_eur"] == 0.0

    # If already tripped, no restart fee applies, so MILP-MPC keeps industrial tripped cleanly
    obs, info = env_pen.reset(options={"start_idx": 0})
    env_pen._industrial_tripped = True
    _, _, _, _, info_cont = env_pen.step(pol_pen.get_action(obs, info))
    assert info_cont["industrial_tripped"] is True
    assert info_cont["cost_industrial_trip_eur"] == 0.0


def test_mpc_causal_noisy_and_persistence_horizon_integration() -> None:
    """Verify MPCPolicy uses env.get_horizon_data for causal_noisy and matching persistence modes."""
    K = 6
    env_noisy = SmartGridEnv(
        {
            "split": "val",
            "seed": 123,
            "forecast_horizon_hours": K,
            "forecast_mode": "causal_noisy",
            "forecast_noise_rel_pv": 0.25,
            "forecast_noise_rel_load": 0.15,
        }
    )
    obs, info = env_noisy.reset(options={"start_idx": 0})
    pol_noisy = MPCPolicy(env=env_noisy, horizon=K, forecast_mode="causal_noisy")

    pv_f, loads_f, price_f = pol_noisy._get_forecast(obs, info=info, env=env_noisy, K=K)
    assert pv_f.shape == (K,)
    assert loads_f.shape == (K, 5)
    assert price_f.shape == (K,)
    assert np.all(pv_f >= 0.0) and np.all(loads_f >= 0.0)
    # Exact D-1 spot prices across the entire horizon
    expected_price = np.maximum(0.0, env_noisy._price[0:K]) / 1000.0
    assert np.allclose(price_f, expected_price, atol=1e-6)
    # Current step t=0 matches current observation exactly
    assert pv_f[0] == pytest.approx(float(obs[1]), abs=1e-5)
    assert np.allclose(loads_f[0], obs[2:7], atol=1e-5)
    # Future loads carry causal noise (differ from oracle ground truth)
    assert not np.allclose(loads_f[1:], env_noisy._zone_loads[1:K])

    # Multi-step execution remains cap-safe
    for _ in range(12):
        act = pol_noisy.get_action(obs, info)
        assert act.shape == (6,)
        obs, _, _, _, info = env_noisy.step(act)
        assert info["grid_exceedance_kw"] <= env_noisy.violation_tolerance_kw

    # Matching persistence mode uses env.get_horizon_data: exact D-1 prices + persisted PV/loads
    env_persist = SmartGridEnv(
        {
            "split": "val",
            "seed": 123,
            "forecast_horizon_hours": K,
            "forecast_mode": "persistence",
        }
    )
    obs_p, info_p = env_persist.reset(options={"start_idx": 10})
    pol_persist = MPCPolicy(env=env_persist, horizon=K, forecast_mode="persistence")
    pv_p, loads_p, price_p = pol_persist._get_forecast(obs_p, info=info_p, env=env_persist, K=K)
    assert np.allclose(price_p, np.maximum(0.0, env_persist._price[10 : 10 + K]) / 1000.0, atol=1e-6)
    assert np.allclose(pv_p, float(obs_p[1]), atol=1e-5)
    assert np.allclose(loads_p, np.tile(obs_p[2:7], (K, 1)), atol=1e-5)
