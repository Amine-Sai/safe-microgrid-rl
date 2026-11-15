import pytest

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
