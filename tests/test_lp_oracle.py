import pytest
from tools.report.lp_oracle import solve_week_lp, compute_split_lp
from env.smartgrid_env import SmartGridEnv


@pytest.mark.needs_real_data
def test_lp_oracle_single_week():
    env = SmartGridEnv({"split": "val", "charge_clamp": True})
    res = solve_week_lp(env, 0)
    assert res["cost_eur"] > 0
    assert res["mean_exceedance_kw"] == 0.0
    assert res["violation_hours"] == 0


@pytest.mark.needs_real_data
def test_lp_oracle_val_split():
    res = compute_split_lp("val")
    assert res["n_weeks"] == 26
    # Matches frozen theoretical value within 5 EUR
    assert abs(res["cost_eur"] - 53290.51) < 5.0
    assert res["mean_exceedance_kw"] == 0.0


@pytest.mark.needs_real_data
def test_lp_oracle_test_split():
    res = compute_split_lp("test")
    assert res["n_weeks"] == 24
    # Matches theoretical Test optimum within 5 EUR
    assert abs(res["cost_eur"] - 36991.67) < 5.0
    assert res["mean_exceedance_kw"] == 0.0
