"""Tests for dynamic results loading in tools/report/load_results.py."""

import pytest
from tools.report.load_results import (
    load_project_constants,
    load_baselines,
    load_per_seed_table,
    load_test_stress_weeks_data,
    extract_metric_row,
)


def test_project_constants_loaded_dynamically():
    constants = load_project_constants()
    assert "g1_exceedance_kw" in constants
    assert "g2_cost_eur" in constants
    assert "contract_cap_kw" in constants
    assert constants["g1_exceedance_kw"] > 0.0
    assert constants["g2_cost_eur"] > 40000.0
    assert constants["contract_cap_kw"] > 300.0


def test_frozen_baselines_loaded_dynamically():
    baselines = load_baselines()
    expected_baselines = ["Grid-Only", "PSM'", "LP'", "Shield + Idle", "Shield + PSM'"]
    for b in expected_baselines:
        assert b in baselines, f"Baseline {b} not found in loaded baselines"
        data = baselines[b]
        assert "cost_eur" in data
        assert "exceedance_kw" in data
        assert "battery_throughput_mwh" in data
        assert data["cost_eur"] > 30000.0


def test_test_stress_weeks_data_loaded_dynamically():
    data = load_test_stress_weeks_data()
    assert "panel_a" in data
    assert "w49_by_prefix" in data
    assert len(data["panel_a"]) > 0
    w49 = data["w49_by_prefix"]
    assert "Grid-Only" in w49
    assert "clamped_mb_ppo" in w49
    assert w49["Grid-Only"][0]["violation_hours"] > 0


def test_extract_metric_row_no_hardcoding():
    baselines = load_baselines()
    per_seed = load_per_seed_table()

    # Baseline extraction
    purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, _, _, soc, s_soc = extract_metric_row(
        "PSM'", "Baseline", baselines=baselines, per_seed_table=per_seed
    )
    assert tot == baselines["PSM'"]["cost_eur"]
    assert tp == baselines["PSM'"]["battery_throughput_mwh"]
    assert exc == baselines["PSM'"]["exceedance_kw"]

    # Safe RL extraction
    purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, _, _, soc, s_soc = extract_metric_row(
        "PPO-Diff-Shield", "In-The-Loop", baselines=baselines, per_seed_table=per_seed
    )
    assert tot > 50000.0
    assert tp > 50.0  # Active battery dispatch
    assert exc < 0.001  # Zero cap exceedance
