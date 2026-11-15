import pytest

from agents.mpc import MPCPolicy
from env.smartgrid_env import SmartGridEnv
from tools.report.lp_oracle import compute_split_lp, solve_week_lp, solve_week_milp

LP_TOLERANCE_EUR = 5.0
VAL_WEEKS = 26
TEST_WEEKS = 24


@pytest.mark.needs_real_data
def test_lp_oracle_single_week():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    res = solve_week_lp(env, 0)
    assert res["cost_eur"] > 0
    assert res["mean_exceedance_kw"] == 0.0
    assert res["violation_hours"] == 0
    assert res["battery_throughput_mwh"] > 0.0
    assert res["active_soc_pct"] > 0.0


@pytest.mark.needs_real_data
def test_milp_oracle_single_week_bound():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    res_lp = solve_week_lp(env, 0)
    res_milp = solve_week_milp(env, 0, mip_rel_gap=0.001)

    assert res_milp["cost_eur"] >= res_lp["cost_eur"] - 1e-2
    assert res_milp["mean_exceedance_kw"] == 0.0
    assert res_milp["violation_hours"] == 0
    assert res_milp["battery_throughput_mwh"] > 0.0
    assert res_milp["active_soc_pct"] > 0.0
    assert res_milp["tripped_hours"] >= 0


@pytest.mark.needs_real_data
def test_milp_oracle_stress_week_w49():
    # W49 is the cap-binding test stress week at start_idx = 3360
    env = SmartGridEnv({"split": "test", "charge_clamp": True})
    w49_start_idx = 3360
    res_lp = solve_week_lp(env, w49_start_idx)
    res_milp = solve_week_milp(env, w49_start_idx, mip_rel_gap=0.01, time_limit=30.0)

    assert res_milp["cost_eur"] >= res_lp["cost_eur"] - 1e-2
    assert res_milp["mean_exceedance_kw"] == 0.0
    assert res_milp["violation_hours"] == 0
    assert res_milp["battery_throughput_mwh"] > 0.0


def test_milp_nonconvex_industrial_trip_constraint():
    # Construct an episode where continuous shedding of 40% is needed to satisfy cap.
    # LP relaxes the motor constraint and sheds 40% continuously (200 kW).
    # MILP enforces s_ind in [0.0, 0.30] U {1.0}: since 30% is insufficient,
    # the motor must trip to 100% shutdown (z_ind = 1), strictly increasing cost.
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    ep_len = env.episode_length
    env._zone_loads[0:ep_len, :4] = 0.0
    env._zone_loads[0:ep_len, 4] = 500.0  # Industrial demand = 500 kW
    env.grid_import_cap_kw = 300.0        # Cap = 300 kW
    env.bat_max_power_kw = 0.0            # No battery dispatch
    env._pv_kw_series[0:ep_len] = 0.0     # No PV

    res_lp = solve_week_lp(env, 0)
    res_milp = solve_week_milp(env, 0)

    assert res_milp["tripped_hours"] == ep_len
    assert res_milp["cost_eur"] > res_lp["cost_eur"]
    assert res_milp["unserved_cost_eur"] == ep_len * 500.0 * env.cost_unserved_eur_kwh


@pytest.mark.needs_real_data
def test_lp_oracle_val_split(project_constants):
    res = compute_split_lp("val")
    assert res["n_weeks"] == VAL_WEEKS
    # Matches frozen theoretical value within 5 EUR
    expected_val_eur = project_constants["lp_prime_val_eur"]
    assert abs(res["cost_eur"] - expected_val_eur) < LP_TOLERANCE_EUR
    assert res["mean_exceedance_kw"] == 0.0


@pytest.mark.needs_real_data
def test_lp_oracle_test_split(project_constants):
    res = compute_split_lp("test")
    assert res["n_weeks"] == TEST_WEEKS
    # Matches theoretical Test optimum within 5 EUR
    expected_test_eur = project_constants["lp_prime_test_eur"]
    assert abs(res["cost_eur"] - expected_test_eur) < LP_TOLERANCE_EUR
    assert res["mean_exceedance_kw"] == 0.0


def test_lp_and_milp_demand_charge_epigraph_parity_with_env():
    ep_len = 6
    base_cfg = {
        "split": "val",
        "charge_clamp": True,
        "episode_length_hours": ep_len,
        "battery_capacity_kwh": 200.0,
        "battery_max_power_kw": 50.0,
        "battery_soc_init": 0.50,
        "grid_import_cap_kw": 500.0,
        "cost_unserved_eur_kwh": 50.0,
        "cost_wear_eur_kwh": 0.02,
    }
    env_no_dc = SmartGridEnv({**base_cfg, "demand_charge_eur_per_kw_week": 0.0})
    env_dc = SmartGridEnv({**base_cfg, "demand_charge_eur_per_kw_week": 10.0})

    for e in (env_no_dc, env_dc):
        e._zone_loads[0:ep_len, :] = 0.0
        e._zone_loads[0:ep_len, 0] = 100.0
        e._zone_loads[2, 0] = 200.0  # Single peak spike hour
        e._pv_kw_series[0:ep_len] = 0.0
        e._price[0:ep_len] = 50.0  # Flat price: only demand charge incentivizes peak shaving

    res_lp_no_dc = solve_week_lp(env_no_dc, 0)
    res_lp_dc = solve_week_lp(env_dc, 0)
    res_milp_dc = solve_week_milp(env_dc, 0)

    assert res_lp_dc["week_peak_grid_kw"] < res_lp_no_dc["week_peak_grid_kw"]
    assert res_lp_dc["week_peak_grid_kw"] == pytest.approx(150.0, abs=1e-3)
    assert res_lp_dc["demand_charge_cost_eur"] == pytest.approx(10.0 * 150.0, abs=0.05)
    assert res_milp_dc["cost_eur"] == pytest.approx(res_lp_dc["cost_eur"], abs=0.05)

    # Full-horizon MILP-MPC rollout must match MILP' and SmartGridEnv accounting identically
    mpc = MPCPolicy(env=env_dc, horizon=ep_len, forecast_mode="perfect", use_milp=True)
    obs, info = env_dc.reset(options={"start_idx": 0})
    mpc.reset(env_dc)
    env_cost = 0.0
    env_dc_cost = 0.0
    done = False
    while not done:
        action = mpc.get_action(obs, info)
        obs, reward, term, trunc, info = env_dc.step(action)
        env_cost += -reward
        env_dc_cost += info["cost_demand_charge_eur"]
        done = term or trunc

    assert env_dc_cost == pytest.approx(res_milp_dc["demand_charge_cost_eur"], abs=0.05)
    assert env_cost == pytest.approx(res_milp_dc["cost_eur"], abs=0.05)
    assert info["week_peak_grid_kw"] == pytest.approx(res_milp_dc["week_peak_grid_kw"], abs=0.05)


def test_lp_and_milp_zone_voll_preference():
    ep_len = 4
    voll_cfg = {"residential": 0.1, "commercial": 2.0, "stadium": 3.0, "street": 4.0, "industrial": 5.0}
    env = SmartGridEnv(
        {
            "split": "val",
            "charge_clamp": True,
            "episode_length_hours": ep_len,
            "battery_max_power_kw": 0.0,
            "grid_import_cap_kw": 400.0,
            "zone_voll_eur_kwh": voll_cfg,
        }
    )
    env._zone_loads[0:ep_len, :] = 100.0  # 500 kW total -> must shed 100 kW/h
    env._pv_kw_series[0:ep_len] = 0.0
    env._price[0:ep_len] = 50.0

    res_lp = solve_week_lp(env, 0)
    res_milp = solve_week_milp(env, 0)

    expected_unserved = ep_len * 100.0 * 0.1  # Sheds residential (lowest VoLL) exclusively
    assert res_lp["unserved_cost_eur"] == pytest.approx(expected_unserved, abs=1e-2)
    assert res_milp["unserved_cost_eur"] == pytest.approx(expected_unserved, abs=1e-2)

    # Uniform zone_voll matches scalar cost_unserved_eur_kwh identically
    env_uniform = SmartGridEnv(
        {
            "split": "val",
            "charge_clamp": True,
            "episode_length_hours": ep_len,
            "zone_voll_eur_kwh": [1.0, 1.0, 1.0, 1.0, 1.0],
            "cost_unserved_eur_kwh": 1.0,
        }
    )
    env_scalar = SmartGridEnv(
        {
            "split": "val",
            "charge_clamp": True,
            "episode_length_hours": ep_len,
            "zone_voll_eur_kwh": None,
            "cost_unserved_eur_kwh": 1.0,
        }
    )
    assert solve_week_lp(env_uniform, 0)["cost_eur"] == pytest.approx(
        solve_week_lp(env_scalar, 0)["cost_eur"], abs=1e-2
    )
    assert solve_week_milp(env_uniform, 0)["cost_eur"] == pytest.approx(
        solve_week_milp(env_scalar, 0)["cost_eur"], abs=1e-2
    )


def test_milp_industrial_trip_switching_penalty():
    ep_len = 4
    voll_cfg = {"residential": 5.0, "commercial": 5.0, "stadium": 5.0, "street": 5.0, "industrial": 1.0}
    common_cfg = {
        "split": "val",
        "charge_clamp": True,
        "episode_length_hours": ep_len,
        "battery_max_power_kw": 0.0,
        "grid_import_cap_kw": 200.0,
        "zone_voll_eur_kwh": voll_cfg,
    }

    # Need 50 kW shed/h: without trip penalty, 100 kW industrial trip (100 EUR/h) beats
    # 30 kW partial industrial + 20 kW residential (30*1 + 20*5 = 130 EUR/h).
    # With 500 EUR trip penalty, MILP keeps motor running and sheds residential instead.
    env_no_pen = SmartGridEnv({**common_cfg, "industrial_trip_penalty_eur": 0.0})
    env_pen = SmartGridEnv({**common_cfg, "industrial_trip_penalty_eur": 500.0})
    for e in (env_no_pen, env_pen):
        e._zone_loads[0:ep_len, :] = 0.0
        e._zone_loads[0:ep_len, 0] = 150.0  # residential
        e._zone_loads[0:ep_len, 4] = 100.0  # industrial (max partial = 30 kW)
        e._pv_kw_series[0:ep_len] = 0.0
        e._price[0:ep_len] = 50.0

    res_no_pen = solve_week_milp(env_no_pen, 0)
    res_pen = solve_week_milp(env_pen, 0)
    assert res_no_pen["tripped_hours"] == ep_len
    assert res_pen["tripped_hours"] == 0
    assert res_pen["trip_transitions"] == 0
    assert res_pen["trip_penalty_cost_eur"] == 0.0

    # Forced trip over consecutive hours pays restart penalty only ONCE (0 -> 1 transition)
    env_forced = SmartGridEnv({**common_cfg, "industrial_trip_penalty_eur": 500.0})
    env_forced._zone_loads[0:ep_len, :] = 0.0
    env_forced._zone_loads[0:ep_len, 4] = 300.0
    env_forced._pv_kw_series[0:ep_len] = 0.0
    env_forced._price[0:ep_len] = 50.0
    res_forced = solve_week_milp(env_forced, 0)
    assert res_forced["tripped_hours"] == ep_len
    assert res_forced["trip_transitions"] == 1
    assert res_forced["trip_penalty_cost_eur"] == pytest.approx(500.0, abs=1e-2)
