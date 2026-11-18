"""Tests for dynamic results loading and plotting helpers."""

import math
import os
import pytest
from tools.report.load_results import (
    DEFAULT_G2_COST_MULTIPLIER,
    compute_g2_cost_gate,
    load_project_constants,
    load_baselines,
    load_per_seed_table,
    load_regime_benchmarks,
    load_test_evaluation_summary,
    load_test_stress_weeks_data,
    load_training_curves,
    extract_metric_row,
)
from tools.report.plots import fig_w49_dispatch_trajectory


def test_project_constants_loaded_dynamically():
    constants = load_project_constants()
    assert "g1_exceedance_kw" in constants
    assert "g2_cost_eur" in constants
    assert "contract_cap_kw" in constants
    assert constants["g1_exceedance_kw"] > 0.0
    assert constants["g2_cost_eur"] > 40000.0
    assert constants["contract_cap_kw"] > 300.0
    assert constants["g2_cost_eur"] == compute_g2_cost_gate()
    assert constants["g2_cost_multiplier"] == DEFAULT_G2_COST_MULTIPLIER


def test_canonical_g2_gate_computation():
    # PSM' Val cost is 55,530.22 EUR -> 1.05x multiplier gives 58,306.73 EUR
    psm_val = 55530.22
    expected_g2 = round(1.05 * psm_val, 2)
    assert expected_g2 == 58306.73
    assert compute_g2_cost_gate(psm_val, multiplier=1.05) == 58306.73
    assert compute_g2_cost_gate() == 58306.73


def test_fig_w49_dispatch_trajectory_generation(tmp_path):
    out_dir = str(tmp_path)
    fig_w49_dispatch_trajectory(out_dir)
    trajectory_png = os.path.join(out_dir, "w49_dispatch_trajectory.png")
    assert os.path.exists(trajectory_png)
    assert os.path.getsize(trajectory_png) > 1000


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
    assert baselines["LP'"]["battery_throughput_mwh"] == pytest.approx(21.837, abs=1e-2)


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


def test_load_regime_benchmarks_and_training_curves():
    """Verify regime benchmark, training curve, and read-only test summary loaders."""
    regime_rows = load_regime_benchmarks()
    assert len(regime_rows) == 121
    suites = {row["suite"] for row in regime_rows}
    assert {"regime1_val", "regime2_val", "ood_val_2019-W04"}.issubset(suites)
    regime2_methods = {row["method"] for row in regime_rows if row["suite"] == "regime2_val"}
    assert {"Shield + Idle", "Shield + PSM'"}.issubset(regime2_methods)
    ood_rows = [row for row in regime_rows if row["suite"] == "ood_val_2019-W04"]
    assert any(row["method"] == "Shield + MPCPolicy(causal_noisy, H=24)" for row in ood_rows)
    assert any(
        row["method"] == "clamped_ppo_diff_shield" and row["row_type"] == "mean"
        for row in ood_rows
    )
    assert all(
        isinstance(row["n_weeks"], int)
        and isinstance(row["cost_eur"], float)
        and math.isfinite(row["cost_eur"])
        for row in regime_rows
    )

    prefixes = ["clamped_ppo_diff_shield", "clamped_mb_ppo_lagrangian"]
    curves = load_training_curves(prefixes)
    assert list(curves.keys()) == prefixes
    expected_steps = [50000, 100000, 150000, 200000, 250000, 300000]
    metric_keys = (
        "total_cost_eur_mean",
        "total_cost_eur_std",
        "violation_rate_pct_mean",
        "violation_rate_pct_std",
        "active_soc_pct_mean",
        "active_soc_pct_std",
        "unserved_penalty_eur_mean",
        "unserved_penalty_eur_std",
    )
    for prefix in prefixes:
        series = curves[prefix]
        assert [pt["step"] for pt in series] == expected_steps
        for pt in series:
            assert pt["n_seeds"] == 5
            for key in metric_keys:
                assert isinstance(pt[key], float) and math.isfinite(pt[key])
            if prefix == "clamped_mb_ppo_lagrangian":
                assert isinstance(pt.get("lambda_mean"), float) and math.isfinite(pt["lambda_mean"])
                assert isinstance(pt.get("lambda_std"), float) and math.isfinite(pt["lambda_std"])
            else:
                assert "lambda_mean" not in pt

    test_summary = load_test_evaluation_summary()
    summary_float_keys = (
        "cost_mean",
        "cost_std",
        "violation_rate_mean",
        "mean_exceedance_mean",
        "unserved_pct_mean",
    )
    for key in ("Grid-Only", "PSM'", "clamped_ppo_diff_shield"):
        assert key in test_summary
        entry = test_summary[key]
        assert entry["n"] >= 1
        for field in summary_float_keys:
            assert isinstance(entry[field], float) and math.isfinite(entry[field])
    assert test_summary["clamped_ppo_diff_shield"]["n"] == 5


def test_presentation_report_modules_import_cleanly():
    """T5: Verify reconstructed style and presentation_plots modules import cleanly."""
    import tools.report.presentation_plots as presentation_plots
    import tools.report.style as style

    assert callable(getattr(presentation_plots, "plot_presentation_all", None))
    assert callable(getattr(presentation_plots, "main", None))
    assert callable(getattr(style, "apply_presentation_rc", None))
    assert callable(getattr(style, "save_slide_fig", None))

