import pytest

from env.smartgrid_env import SmartGridEnv
from tools.report.lp_oracle import compute_split_lp, solve_week_lp

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
