"""Publication-grade figure suite for the Safe Microgrid RL Benchmark.

Structured around four focused study cases:
  1. Study 1: Clamping & Batching Ablation in Basic PPO (Validation split)
     - results/figures/study1_ppo_ablation/cost.png
     - results/figures/study1_ppo_ablation/grid_exceedance.png
     - results/figures/study1_ppo_ablation/battery/throughput.png
     - results/figures/study1_ppo_ablation/battery/active_soc.png

  2. Study 2: Capped & Constrained PPO vs. Baselines (Validation split)
     - results/figures/study2_capped_ppo/cost.png
     - results/figures/study2_capped_ppo/grid_exceedance.png
     - results/figures/study2_capped_ppo/unserved_load.png
     - results/figures/study2_capped_ppo/battery/throughput.png
     - results/figures/study2_capped_ppo/battery/active_soc.png

  3. Study 3: Multi-Objective Pareto Analysis & Safety Gates (Validation split)
     - results/figures/study3_stress_pareto/pareto.png
     - results/figures/study3_stress_pareto/stress_week_pareto.png

  4. Study 4: Plain PPO vs. PPO-Lagrangian Comparison across Ablation Regimes (Validation split)
     - results/figures/study4_plain_vs_lagrangian/cost.png
     - results/figures/study4_plain_vs_lagrangian/grid_exceedance.png
     - results/figures/study4_plain_vs_lagrangian/battery/throughput.png
     - results/figures/study4_plain_vs_lagrangian/battery/active_soc.png
"""

import csv
import io
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.report.load_results import (
    load_project_constants as _load_project_constants,
    load_baselines as _load_baselines,
    load_per_seed_table,
    load_test_stress_weeks_data,
    extract_metric_row,
    extract_violation_hours,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES = os.path.join(ROOT, "results", "tables")
RUNS = os.path.join(ROOT, "results", "runs")
FIGS = os.path.join(ROOT, "results", "figures")

STUDY1_DIR = os.path.join(FIGS, "study1_ppo_ablation")
STUDY2_DIR = os.path.join(FIGS, "study2_capped_ppo")
STUDY3_DIR = os.path.join(FIGS, "study3_stress_pareto")
STUDY4_DIR = os.path.join(FIGS, "study4_plain_vs_lagrangian")

STUDY_DIRS = [STUDY1_DIR, STUDY2_DIR, STUDY3_DIR, STUDY4_DIR]

# High-contrast accessible color palette
PALETTE = {
    "PPO": "#0072B2",               # Deep Blue (Clamped MB)
    "PPO_FB": "#2980B9",            # Medium Blue (Clamped FB)
    "PPO_UNC": "#7FB3D5",           # Soft Sky Blue (Unclamped)
    "PPO-Lagrangian": "#D55E00",    # Vermillion / Orange-Red (Clamped MB)
    "PPO-Lagrangian_FB": "#E67E22", # Medium Orange (Clamped FB)
    "PPO-Lagrangian_UNC": "#F5B041",# Warm Amber (Unclamped)
    "PPO-Residual": "#009E73",      # Emerald Green
    "Shielded PPO": "#CC79A7",      # Mauve / Rose (Clamped FB)
    "Shielded PPO_MB": "#884969",   # Deep Plum (Clamped MB)
    "PPO-Aux-Shield": "#7D3C98",    # Royal Purple
    "PPO-Diff-Shield": "#16A085",   # Teal Green
    "Grid-Only": "#4A5568",         # Charcoal / Slate
    "PSM'": "#1A5276",              # Dark Navy
    "LP'": "#196F3D",               # Forest Green
    "Shield + Idle": "#718096",     # Cool Grey
    "Shield + PSM'": "#2B6CB0",     # Lapis Blue
    "Gate": "#C0392B",              # Crimson Red
    "Feasible": "#2ECC71",          # Soft Emerald Mint
    "Energy": "#1F77B4",            # Grid energy purchase blue
    "Wear": "#7F7F7F",              # Battery wear degradation slate
    "Unserved": "#E67E22",          # Unserved curtailment penalty amber
}

SAFETY_CONFIG = [
    ("Grid-Only", "Grid-Only", "Baseline", None, None, PALETTE["Grid-Only"], "x", "--"),
    ("PSM'", "PSM'", "Baseline", None, None, PALETTE["PSM'"], "^", ":"),
    ("LP'", "LP'", "Baseline", None, None, PALETTE["LP'"], "D", "-."),
    ("Shield + Idle", "Shield + Idle", "Shield Control", None, None, PALETTE["Shield + Idle"], "v", "--"),
    ("Shield + PSM'", "Shield + PSM'", "Shield Control", None, None, PALETTE["Shield + PSM'"], "s", ":"),
    ("Shielded PPO (Clamped)", "Shielded PPO [Clamped Full-Batch]", "Env Shield", "ppo_shielded", "clamped_ppo_shielded", PALETTE["Shielded PPO"], "o", "-"),
    ("Shielded PPO (Clamped MB)", "Shielded PPO [Clamped Minibatch]", "Env Shield", "ppo_shielded", "clamped_mb_ppo_shielded", PALETTE["Shielded PPO_MB"], "^", "--"),
    ("PPO-Aux-Shield", "PPO-Aux-Shield", "In-The-Loop Safe RL", "ppo_shield_aux", "clamped_ppo_shield_aux", PALETTE["PPO-Aux-Shield"], "D", "-."),
    ("PPO-Diff-Shield", "PPO-Diff-Shield", "In-The-Loop Safe RL", "ppo_diff_shield", "clamped_ppo_diff_shield", PALETTE["PPO-Diff-Shield"], "p", ":"),
]

# Load constants and tables dynamically
PROJECT_CONSTANTS = _load_project_constants()
G1_EXC_KW = PROJECT_CONSTANTS["g1_exceedance_kw"]
G2_COST_EUR = PROJECT_CONSTANTS["g2_cost_eur"]
CONTRACT_CAP_KW = PROJECT_CONSTANTS["contract_cap_kw"]
WEAR_COST_PER_MWH = PROJECT_CONSTANTS["wear_cost_per_mwh"]
PER_SEED_TABLE = load_per_seed_table()
TEST_STRESS_DATA = load_test_stress_weeks_data()

# Global publication styling
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.94,
    "legend.edgecolor": "#CCCCCC",
})


def load_baselines() -> dict:
    return _load_baselines()


def load_runs(family: str) -> dict:
    out = {}
    family_dir = os.path.join(RUNS, family)
    if not os.path.isdir(family_dir):
        return out
    for run_name in sorted(os.listdir(family_dir)):
        run_path = os.path.join(family_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        for fname in sorted(os.listdir(run_path)):
            if not fname.endswith(".json") or fname.endswith("_grid.json"):
                continue
            with io.open(os.path.join(run_path, fname), encoding="utf-8") as f:
                run_data = json.load(f)
                label = run_data.get("method") or run_name
                out.setdefault(label, []).append(run_data)
    return out


def load_safety_runs() -> dict:
    runs = {}
    for key, _name, _cat, folder, prefix, _col, _mk, _ls in SAFETY_CONFIG:
        if folder is None:
            continue
        runs[key] = []
        agent_dir = os.path.join(RUNS, folder)
        if not os.path.isdir(agent_dir):
            continue
        for fname in sorted(os.listdir(agent_dir)):
            if not fname.endswith(".json") or fname.endswith("_grid.json"):
                continue
            if fname.startswith(prefix):
                with io.open(os.path.join(agent_dir, fname), encoding="utf-8") as f:
                    runs[key].append(json.load(f))
    return runs


def _extract_metric_row(key, cat, all_families, baselines, safety_runs):
    return extract_metric_row(
        key=key,
        cat=cat,
        all_families=all_families,
        baselines=baselines,
        safety_runs=safety_runs,
        per_seed_table=PER_SEED_TABLE,
    )


def _extract_violation_hours(key, cat, all_families, baselines):
    return extract_violation_hours(
        key=key,
        cat=cat,
        all_families=all_families,
        baselines=baselines,
        per_seed_table=PER_SEED_TABLE,
    )


# ===========================================================================
# STUDY 1: BATCHING & CLAMPING ABLATION IN PPO (Validation split)
# ===========================================================================

STUDY1_ITEMS = [
    ("clamped_ppo", "PPO [Full-Batch]", "Ablation", PALETTE["PPO_FB"]),
    ("s3_ppo", "PPO [Mini-Batch]", "Ablation", PALETTE["PPO_UNC"]),
    ("clamped_mb_ppo", "PPO [Clamped Minibatch]", "Ablation", PALETTE["PPO"]),
]


def fig_study1_cost(baselines: dict, out_dir: str = None):
    """Study 1 Cost: Vertical stacked bar chart showing total cost decomposition."""
    dest_dir = out_dir or STUDY1_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    bar_w = 0.46

    c_lp = float(baselines.get("LP'", {}).get("cost_eur", 53290.51))
    c_psm = float(baselines.get("PSM'", {}).get("cost_eur", 55530.22))
    c_g2 = float(G2_COST_EUR)

    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])
    part_names = ("Grid Energy Purchase (€)", "Battery Degradation Wear (€)", "Unserved Load Penalty (€)")

    x_positions = np.arange(len(STUDY1_ITEMS))
    x_labels = [item[1] for item in STUDY1_ITEMS]

    for x, (key, _display_name, cat, _colour) in enumerate(STUDY1_ITEMS):
        purch, wear, unserv, tot, s_tot, _exc, _s_exc, _tp, _s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )

        bottom = 0.0
        for val, col in zip([purch, wear, unserv], part_colours):
            ax.bar(x, val, bottom=bottom, width=bar_w, color=col, edgecolor="#FFFFFF", lw=0.9, zorder=3)
            bottom += val

        # Error bar on total cost
        ax.errorbar(
            x, tot, yerr=s_tot, fmt="none",
            ecolor="#1A252C", elinewidth=1.8, capsize=6.0, capthick=1.8, zorder=5
        )

        # End-of-bar text on top of each bar with clean pill badge
        y_label = tot + s_tot + 1200.0
        ax.text(
            x, y_label, f"€{tot/1e3:.1f}k ± €{s_tot/1e3:.1f}k",
            ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Horizontal reference lines (Oracle present; no black Grid-Only line!)
    ax.axhline(c_lp, color=PALETTE["LP'"], ls="-.", lw=1.6, zorder=2)
    ax.axhline(c_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)
    ax.axhline(c_g2, color=PALETTE["Gate"], ls="--", lw=1.8, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=11)
    ax.set_xlim(-0.6, len(STUDY1_ITEMS) - 0.4)
    ax.set_ylim(0, 68000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y*1e-3:.0f}k"))
    ax.set_ylabel("Cost (€)", fontweight="bold", fontsize=11.5)

    # Clean legend handles in column-first order so Row 1 is components and Row 2 is reference lines
    legend_handles = [
        Patch(facecolor=part_colours[0], label=part_names[0]),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label=f"LP' Oracle (€{c_lp/1e3:.1f}k)"),
        Patch(facecolor=part_colours[1], label=part_names[1]),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' (€{c_psm/1e3:.1f}k)"),
        Patch(facecolor=part_colours[2], label=part_names[2]),
        Line2D([0], [0], color=PALETTE["Gate"], ls="--", lw=1.8, label=f"G2 Cost Gate (€{c_g2/1e3:.1f}k)"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=True, fontsize=9.8
    )
    ax.set_title(
        "Total Operational Cost Breakdown across PPO Batching & Clamping Variants",
        fontsize=12.0, fontweight="bold", pad=52
    )

    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.12, right=0.96)
    out_file = os.path.join(dest_dir, "cost.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study1_grid_exceedance(baselines: dict, out_dir: str = None):
    """Study 1 Grid Exceedance: Vertical bar chart."""
    dest_dir = out_dir or STUDY1_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bar_w = 0.46

    x_positions = np.arange(len(STUDY1_ITEMS))
    x_labels = [item[1] for item in STUDY1_ITEMS]

    for x, (key, _display_name, cat, colour) in enumerate(STUDY1_ITEMS):
        _purch, _wear, _unserv, _tot, _s_tot, exc, s_exc, _tp, _s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )
        m_vh, _s_vh = _extract_violation_hours(key, cat, None, baselines)

        ax.bar(x, exc, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.9, zorder=3)
        ax.errorbar(
            x, exc, yerr=s_exc, fmt="none",
            ecolor="#1A252C", elinewidth=1.8, capsize=6.0, capthick=1.8, zorder=5
        )

        pct_vh = (m_vh / 4368.0) * 100.0
        y_label = exc + s_exc + 0.12
        ax.text(
            x, y_label, f"{exc:.2f} kW/step\n({m_vh:.0f}h / {pct_vh:.1f}%)",
            ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Reference lines (Oracle and G1 present; no black Grid-Only line!)
    ax.axhline(G1_EXC_KW, color=PALETTE["Gate"], ls="-.", lw=1.8, zorder=2)

    # Clean legend for reference lines placed horizontally above axes
    ref_handles = [
        Line2D([0], [0], color=PALETTE["Gate"], ls="-.", lw=1.8, label=f"G1 Exceedance Gate ({G1_EXC_KW:.2f} kW/step)"),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label="LP' Oracle: 0.00 kW/step (0h)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=10.0
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=11)
    ax.set_xlim(-0.6, len(STUDY1_ITEMS) - 0.4)
    ax.set_ylim(0, 4.5)
    ax.set_ylabel("Mean Exceedance (kW/step)", fontweight="bold", fontsize=11.0)
    ax.set_title(
        "Mean Grid Import Exceedance & Violation Hours across PPO Variants",
        fontsize=12.0, fontweight="bold", pad=36
    )

    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.12, right=0.96)
    out_file = os.path.join(dest_dir, "grid_exceedance.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study1_battery_throughput(baselines: dict, out_dir: str = None):
    """Study 1 Battery Throughput: Dedicated vertical bar chart."""
    dest_dir = out_dir or os.path.join(STUDY1_DIR, "battery")
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    bar_w = 0.46

    tp_psm = float(baselines.get("PSM'", {}).get("battery_throughput_mwh", 12.348))
    tp_lp = float(baselines.get("LP'", {}).get("battery_throughput_mwh", 24.990))

    x_positions = np.arange(len(STUDY1_ITEMS))
    x_labels = [item[1] for item in STUDY1_ITEMS]

    for x, (key, _display_name, cat, colour) in enumerate(STUDY1_ITEMS):
        _purch, wear, _unserv, _tot, _s_tot, _exc, _s_exc, tp, s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )

        ax.bar(x, tp, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.9, zorder=3)
        ax.errorbar(
            x, tp, yerr=s_tp, fmt="none",
            ecolor="#1A252C", elinewidth=1.8, capsize=6.0, capthick=1.8, zorder=5
        )
        y_tp_label = max(tp + s_tp + 2.0, tp_lp + 2.5) if x <= 1 else tp + s_tp + 2.5
        ax.text(
            x, y_tp_label, f"{tp:.2f} MWh\n(€{wear:.0f})",
            ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Reference lines (Oracle present)
    ax.axhline(tp_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)
    ax.axhline(tp_lp, color=PALETTE["LP'"], ls="-.", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' ({tp_psm:.2f} MWh)"),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.8, label=f"LP' Oracle ({tp_lp:.2f} MWh)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=9.8
    )

    # Secondary y-axis for wear cost
    ax.set_ylim(0, 115.0)
    ax_twin = ax.twinx()
    ax_twin.set_ylim(0, 115.0 * WEAR_COST_PER_MWH)
    ax_twin.set_ylabel("Wear Cost (€)", fontweight="bold", fontsize=10.5)
    ax_twin.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y:,.0f}"))
    ax_twin.spines["right"].set_visible(True)
    ax_twin.grid(False)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.5)
    ax.set_xlim(-0.6, len(STUDY1_ITEMS) - 0.4)
    ax.set_ylabel("Throughput (MWh)", fontweight="bold", fontsize=11.0)
    ax.set_title("Battery Energy Throughput & Degradation Wear", fontsize=11.5, fontweight="bold", pad=36)

    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.12, right=0.88)
    out_file = os.path.join(dest_dir, "throughput.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study1_battery_active_soc(baselines: dict, out_dir: str = None):
    """Study 1 Active SoC Operating Time: Dedicated vertical bar chart."""
    dest_dir = out_dir or os.path.join(STUDY1_DIR, "battery")
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    bar_w = 0.46

    soc_psm = 11.2
    soc_lp = 85.0

    x_positions = np.arange(len(STUDY1_ITEMS))
    x_labels = [item[1] for item in STUDY1_ITEMS]

    for x, (key, _display_name, cat, colour) in enumerate(STUDY1_ITEMS):
        _purch, _wear, _unserv, _tot, _s_tot, _exc, _s_exc, _tp, _s_tp, _, _, soc, s_soc = _extract_metric_row(
            key, cat, None, baselines, None
        )

        ax.bar(x, soc, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.9, zorder=3)
        ax.errorbar(
            x, soc, yerr=s_soc, fmt="none",
            ecolor="#1A252C", elinewidth=1.8, capsize=6.0, capthick=1.8, zorder=5
        )
        y_soc_label = soc + s_soc + 2.5
        ax.text(
            x, y_soc_label, f"{soc:.1f}%\n({soc*43.68:.0f}h)",
            ha="center", va="bottom", fontsize=9.8, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Reference lines (Oracle present)
    ax.axhline(soc_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)
    ax.axhline(soc_lp, color=PALETTE["LP'"], ls="-.", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' ({soc_psm:.1f}%)"),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.8, label=f"LP' Oracle ({soc_lp:.1f}%)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=9.8
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=15, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.5)
    ax.set_xlim(-0.6, len(STUDY1_ITEMS) - 0.4)
    ax.set_ylim(0, 98.0)
    ax.set_ylabel("Active SoC Time (%)", fontweight="bold", fontsize=11.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title("Active Unpinned SoC Operating Time (% of 4,368h)", fontsize=11.5, fontweight="bold", pad=36)

    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.12, right=0.92)
    out_file = os.path.join(dest_dir, "active_soc.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")



# ===========================================================================
# STUDY 2: CAPPED PPO VS. VANILLA PPO & BASELINES (Validation split)
# ===========================================================================

STUDY2_ITEMS = [
    # Reference Baselines
    ("LP'", "LP' Oracle", "Baseline", PALETTE["LP'"]),
    ("PSM'", "PSM'", "Baseline", PALETTE["PSM'"]),
    ("Shield + Idle", "Shield + Idle", "Baseline", PALETTE["Shield + Idle"]),
    ("Grid-Only", "Grid-Only", "Baseline", PALETTE["Grid-Only"]),
    (None, "DIVIDER_1", "Divider", None),
    # Vanilla Unconstrained PPO
    ("clamped_mb_ppo", "Vanilla PPO", "Plain PPO", PALETTE["PPO"]),
    (None, "DIVIDER_2", "Divider", None),
    # Soft/Residual Capped PPO
    ("clamped_mb_ppo_lagrangian", "PPO-Lagrangian", "Soft Capped", PALETTE["PPO-Lagrangian"]),
    ("clamped_ppo_residual", "PPO-Residual", "Residual Capped", PALETTE["PPO-Residual"]),
    (None, "DIVIDER_3", "Divider", None),
    # Hard Shielded / Safety-Layer PPO
    ("clamped_mb_ppo_shielded", "Shielded PPO", "Shielded PPO", PALETTE["Shielded PPO_MB"]),
    ("clamped_ppo_shield_aux", "PPO-Aux-Shield", "Safety Layer PPO", PALETTE["PPO-Aux-Shield"]),
    ("clamped_ppo_diff_shield", "PPO-Diff-Shield", "Safety Layer PPO", PALETTE["PPO-Diff-Shield"]),
]


def _parse_study2_items():
    controllers = []
    dividers = []
    x = 0
    for key, display_name, cat, colour in STUDY2_ITEMS:
        if key is None:
            dividers.append(x - 0.5)
            continue
        controllers.append((x, key, display_name, cat, colour))
        x += 1
    return controllers, dividers


def fig_study2_cost(baselines: dict, out_dir: str = None):
    """Study 2 Cost: Vertical stacked bar chart across 10 controllers."""
    dest_dir = out_dir or STUDY2_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    bar_w = 0.54

    c_lp = float(baselines.get("LP'", {}).get("cost_eur", 53290.51))
    c_psm = float(baselines.get("PSM'", {}).get("cost_eur", 55530.22))
    c_g2 = float(G2_COST_EUR)

    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])
    part_names = ("Grid Energy Purchase (€)", "Battery Degradation Wear (€)", "Unserved Load Penalty (€)")

    controllers, dividers = _parse_study2_items()

    for x_pos, key, _display_name, cat, _colour in controllers:
        purch, wear, unserv, tot, s_tot, _exc, _s_exc, _tp, _s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )

        bottom = 0.0
        for val, col in zip([purch, wear, unserv], part_colours):
            ax.bar(x_pos, val, bottom=bottom, width=bar_w, color=col, edgecolor="#FFFFFF", lw=0.8, zorder=3)
            bottom += val

        if s_tot >= 50.0:
            ax.errorbar(
                x_pos, tot, yerr=s_tot, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=5
            )
            label_text = f"€{tot/1e3:.1f}k\n±€{s_tot/1e3:.1f}k"
        else:
            label_text = f"€{tot/1e3:.1f}k"

        y_label = tot + s_tot + 1200.0
        ax.text(
            x_pos, y_label, label_text,
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Horizontal reference lines (Oracle present; no black line!)
    ax.axhline(c_lp, color=PALETTE["LP'"], ls="-.", lw=1.6, zorder=2)
    ax.axhline(c_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)
    ax.axhline(c_g2, color=PALETTE["Gate"], ls="--", lw=1.8, zorder=2)

    # Group header annotations near the top
    group_headers = [
        (1.5, "Baselines"),
        (4.0, "Vanilla"),
        (5.5, "Soft / Residual Capped"),
        (8.0, "Hard Shielded / Safety-Layer"),
    ]
    for hx, htext in group_headers:
        ax.text(hx, 88500, htext, ha="center", va="top", fontsize=9.8, fontweight="bold", color="#4A5568")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=35, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.0)
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(0, 93000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y*1e-3:.0f}k"))
    ax.set_ylabel("Cost (€)", fontweight="bold", fontsize=11.5)

    # Clean legend handles in column-first order so Row 1 is components and Row 2 is reference lines
    legend_handles = [
        Patch(facecolor=part_colours[0], label=part_names[0]),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label=f"LP' Oracle (€{c_lp/1e3:.1f}k)"),
        Patch(facecolor=part_colours[1], label=part_names[1]),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' (€{c_psm/1e3:.1f}k)"),
        Patch(facecolor=part_colours[2], label=part_names[2]),
        Line2D([0], [0], color=PALETTE["Gate"], ls="--", lw=1.8, label=f"G2 Cost Gate (€{c_g2/1e3:.1f}k)"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=True, fontsize=9.8
    )
    ax.set_title(
        "Total Operational Cost Breakdown across Capped PPO Variants & Reference Baselines",
        fontsize=12.5, fontweight="bold", pad=52
    )

    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.97)
    out_file = os.path.join(dest_dir, "cost.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study2_grid_exceedance(baselines: dict, out_dir: str = None):
    """Study 2 Grid Exceedance: Vertical bar chart across 10 controllers."""
    dest_dir = out_dir or STUDY2_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    bar_w = 0.54

    controllers, dividers = _parse_study2_items()

    for x_pos, key, _display_name, cat, colour in controllers:
        _purch, _wear, _unserv, _tot, _s_tot, exc, s_exc, _tp, _s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )
        m_vh, _s_vh = _extract_violation_hours(key, cat, None, baselines)

        ax.bar(x_pos, exc, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.8, zorder=3)

        if exc > 1e-4:
            if s_exc > 0.0:
                ax.errorbar(
                    x_pos, exc, yerr=s_exc, fmt="none",
                    ecolor="#1A252C", elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=5
                )
                y_label = exc + s_exc + 0.12
            else:
                y_label = exc + 0.12

            pct_vh = (m_vh / 4368.0) * 100.0
            ax.text(
                x_pos, y_label, f"{exc:.2f} kW/step\n({m_vh:.0f}h / {pct_vh:.1f}%)",
                ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#1A252C",
                bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
            )
        else:
            # Safe controllers: placed at y = 0.14 above the G1 red line (at 0.05) with clean green pill
            ax.text(
                x_pos, 0.14, "0.00 kW/step\n(0h / 0.0%)",
                ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#1E8449",
                bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="#27AE60", alpha=0.92, lw=0.7)
            )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Horizontal reference line for G1 gate
    ax.axhline(G1_EXC_KW, color=PALETTE["Gate"], ls="-.", lw=1.8, zorder=2)
    ref_handles = [
        Line2D([0], [0], color=PALETTE["Gate"], ls="-.", lw=1.8, label=f"G1 Exceedance Gate ({G1_EXC_KW:.2f} kW/step)"),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label="LP' Oracle: 0.00 kW/step (0h)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=10.0
    )

    # Group header annotations
    group_headers = [
        (1.5, "Baselines"),
        (4.0, "Vanilla"),
        (5.5, "Soft / Residual Capped"),
        (8.0, "Hard Shielded / Safety-Layer"),
    ]
    for hx, htext in group_headers:
        ax.text(hx, 4.15, htext, ha="center", va="top", fontsize=9.8, fontweight="bold", color="#4A5568")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=35, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.0)
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(0, 4.45)
    ax.set_ylabel("Mean Exceedance (kW/step)", fontweight="bold", fontsize=11.0)
    ax.set_title(
        "Mean Grid Import Exceedance & Violation Hours across Capped PPO Variants",
        fontsize=12.5, fontweight="bold", pad=36
    )

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.97)
    out_file = os.path.join(dest_dir, "grid_exceedance.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study2_unserved_load(baselines: dict, out_dir: str = None):
    """Study 2 Unserved Load: Dedicated vertical bar chart across 10 controllers."""
    dest_dir = out_dir or STUDY2_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 7.2))
    bar_w = 0.54

    unserv_lp = float(baselines.get("LP'", {}).get("unserved_penalty_eur", 8660.86)) / 1000.0
    unserv_psm = float(baselines.get("PSM'", {}).get("unserved_penalty_eur", 10624.67)) / 1000.0

    controllers, dividers = _parse_study2_items()

    for x_pos, key, _display_name, cat, colour in controllers:
        purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, _, _, soc, s_soc = _extract_metric_row(
            key, cat, None, baselines, None
        )
        rows = PER_SEED_TABLE.get(key, [])
        if not rows:
            alias_map = {
                "clamped_ppo_shield_aux": "clamped_ppo_shield_aux",
                "clamped_ppo_diff_shield": "clamped_ppo_diff_shield",
                "PPO-Aux-Shield": "clamped_ppo_shield_aux",
                "PPO-Diff-Shield": "clamped_ppo_diff_shield",
            }
            target = alias_map.get(key)
            rows = PER_SEED_TABLE.get(target, [])

        if rows:
            unservs = [r["unserved_penalty_eur"] for r in rows]
            s_unserv = statistics.stdev(unservs) if len(unservs) > 1 else 0.0
            pcts = [r.get("unserved_pct", (r["unserved_penalty_eur"] / 1284520.0) * 100.0) for r in rows]
            m_pct = statistics.fmean(pcts) if pcts else 0.0
        else:
            s_unserv = 0.0
            m_pct = (unserv / 1284520.0) * 100.0

        u_mwh = unserv / 1000.0
        su_mwh = s_unserv / 1000.0

        ax.bar(x_pos, u_mwh, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.8, zorder=3)
        if su_mwh >= 0.05:
            ax.errorbar(
                x_pos, u_mwh, yerr=su_mwh, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=4.5, capthick=1.8, zorder=5
            )
            label_text = f"{u_mwh:.1f} MWh\n±{su_mwh:.1f}\n({m_pct:.2f}%)"
            y_label = u_mwh + su_mwh + 1.0
        else:
            label_text = f"{u_mwh:.1f} MWh\n({m_pct:.2f}%)"
            y_label = u_mwh + 1.0

        ax.text(
            x_pos, y_label, label_text, ha="center", va="bottom", fontsize=8.6, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Reference lines (LP' Oracle and PSM'; no black line!)
    ax.axhline(unserv_lp, color=PALETTE["LP'"], ls="-.", lw=1.8, zorder=2)
    ax.axhline(unserv_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.8, label=f"LP' Oracle ({unserv_lp:.2f} MWh / €{unserv_lp*1e3:,.0f})"),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' ({unserv_psm:.2f} MWh / €{unserv_psm*1e3:,.0f})"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=9.8
    )

    ax.set_ylim(0, 42.0)
    ax_twin = ax.twinx()
    ax_twin.set_ylim(0, 42.0 * 1000.0)
    ax_twin.set_ylabel("Unserved Penalty (€)", fontweight="bold", fontsize=10.5)
    ax_twin.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y*1e-3:.0f}k"))
    ax_twin.spines["right"].set_visible(True)
    ax_twin.grid(False)

    group_headers = [
        (1.5, "Baselines"),
        (4.0, "Vanilla"),
        (5.5, "Soft / Residual Capped"),
        (8.0, "Hard Shielded / Safety-Layer"),
    ]
    for hx, htext in group_headers:
        ax.text(hx, 39.5, htext, ha="center", va="top", fontsize=9.8, fontweight="bold", color="#4A5568")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=35, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.0)
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylabel("Unserved Load (MWh)", fontweight="bold", fontsize=11.0)
    ax.set_title("Unserved Load Demand & Curtailment Penalty across Capped PPO Variants", fontsize=12.0, fontweight="bold", pad=36)

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.92)
    out_file = os.path.join(dest_dir, "unserved_load.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study2_battery_throughput(baselines: dict, out_dir: str = None):
    """Study 2 Battery Throughput: Dedicated vertical bar chart across 10 controllers."""
    dest_dir = out_dir or os.path.join(STUDY2_DIR, "battery")
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    bar_w = 0.54

    tp_psm = float(baselines.get("PSM'", {}).get("battery_throughput_mwh", 12.348))
    tp_lp = float(baselines.get("LP'", {}).get("battery_throughput_mwh", 24.990))

    controllers, dividers = _parse_study2_items()

    for x_pos, key, _display_name, cat, colour in controllers:
        _purch, wear, _unserv, _tot, _s_tot, _exc, _s_exc, tp, s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )

        ax.bar(x_pos, tp, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.8, zorder=3)
        if s_tp > 0.0:
            ax.errorbar(
                x_pos, tp, yerr=s_tp, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=4.5, capthick=1.8, zorder=5
            )
            label_text = f"{tp:.1f} MWh\n±{s_tp:.1f}"
            y_label = tp + s_tp + 3.0
        else:
            label_text = f"{tp:.1f} MWh"
            y_label = tp + 3.0

        ax.text(
            x_pos, y_label, label_text, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Reference lines (Oracle present; no black line!)
    ax.axhline(tp_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)
    ax.axhline(tp_lp, color=PALETTE["LP'"], ls="-.", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' ({tp_psm:.2f} MWh)"),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.8, label=f"LP' Oracle ({tp_lp:.2f} MWh)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=9.8
    )

    # Secondary y-axis for wear cost
    ax_twin = ax.twinx()
    ax.set_ylim(0, 155.0)
    ax_twin.set_ylim(0, 155.0 * WEAR_COST_PER_MWH)
    ax_twin.set_ylabel("Wear Cost (€)", fontweight="bold", fontsize=10.5)
    ax_twin.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y:,.0f}"))
    ax_twin.spines["right"].set_visible(True)
    ax_twin.grid(False)

    # Group header annotations
    group_headers = [
        (1.5, "Baselines"),
        (4.0, "Vanilla"),
        (5.5, "Soft / Residual Capped"),
        (8.0, "Hard Shielded / Safety-Layer"),
    ]
    for hx, htext in group_headers:
        ax.text(hx, 145.0, htext, ha="center", va="top", fontsize=9.8, fontweight="bold", color="#4A5568")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=35, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.0)
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylabel("Throughput (MWh)", fontweight="bold", fontsize=11.0)
    ax.set_title("Battery Energy Throughput & Degradation Wear across Capped PPO Variants", fontsize=12.0, fontweight="bold", pad=36)

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.92)
    out_file = os.path.join(dest_dir, "throughput.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study2_battery_active_soc(baselines: dict, out_dir: str = None):
    """Study 2 Active SoC Operating Time: Dedicated vertical bar chart across 10 controllers."""
    dest_dir = out_dir or os.path.join(STUDY2_DIR, "battery")
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    bar_w = 0.54

    soc_psm = 11.2
    soc_lp = 85.0

    controllers, dividers = _parse_study2_items()

    for x_pos, key, _display_name, cat, colour in controllers:
        _purch, _wear, _unserv, _tot, _s_tot, _exc, _s_exc, _tp, _s_tp, _, _, soc, s_soc = _extract_metric_row(
            key, cat, None, baselines, None
        )

        ax.bar(x_pos, soc, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.8, zorder=3)
        if s_soc > 0.0:
            ax.errorbar(
                x_pos, soc, yerr=s_soc, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=4.5, capthick=1.8, zorder=5
            )
            label_text = f"{soc:.1f}%\n±{s_soc:.1f}%"
            y_label = soc + s_soc + 2.5
        else:
            label_text = f"{soc:.1f}%"
            y_label = soc + 2.5

        ax.text(
            x_pos, y_label, label_text, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Reference lines (Oracle present; no black line!)
    ax.axhline(soc_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)
    ax.axhline(soc_lp, color=PALETTE["LP'"], ls="-.", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' ({soc_psm:.1f}%)"),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.8, label=f"LP' Oracle ({soc_lp:.1f}%)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=9.8
    )

    # Group header annotations
    group_headers = [
        (1.5, "Baselines"),
        (4.0, "Vanilla"),
        (5.5, "Soft / Residual Capped"),
        (8.0, "Hard Shielded / Safety-Layer"),
    ]
    for hx, htext in group_headers:
        ax.text(hx, 104.0, htext, ha="center", va="top", fontsize=9.8, fontweight="bold", color="#4A5568")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=35, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.0)
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(0, 110.0)
    ax.set_ylabel("Active SoC Time (%)", fontweight="bold", fontsize=11.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title("Active Unpinned SoC Operating Time (% of 4,368h) across Capped PPO Variants", fontsize=12.0, fontweight="bold", pad=36)

    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.94)
    out_file = os.path.join(dest_dir, "active_soc.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


# ===========================================================================
# STUDY 3: MULTI-OBJECTIVE PARETO ANALYSIS & SAFETY GATES
# ===========================================================================

def fig_study3_stress_pareto(baselines: dict, out_dir: str = None):
    """Pareto frontier: Cost vs. Grid Exceedance across fine-tuned controllers and G1/G2 safety gates."""
    dest_dir = out_dir or STUDY3_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.8, 8.4))
    floor_val = 1e-4

    ax.set_xscale("log")
    ax.set_xlim(floor_val * 0.45, 6.0)
    ax.set_ylim(41000, 83000)

    # Safe Zone: Rectangle shaped by G1 (exceedance <= 0.05 kW) and G2 (cost <= G2_COST_EUR)
    ax.fill_between(
        [floor_val * 0.45, G1_EXC_KW], [41000, 41000], [G2_COST_EUR, G2_COST_EUR],
        facecolor="#D5F5E3", edgecolor="#27AE60", lw=1.8, ls="--", alpha=0.45, zorder=1
    )

    # Gate boundary lines defining the safe zone
    ax.axvline(G1_EXC_KW, color=PALETTE["Gate"], ls="-.", lw=1.6, zorder=2)
    ax.axhline(G2_COST_EUR, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=2)

    # Text box inside the Safe Zone rectangle (dynamically formatted to match G1 and G2)
    ax.text(
        0.0006, 47500,
        f"SAFE ZONE\nPasses G1 & G2 Gates\n• Cost ≤ €{G2_COST_EUR/1e3:.1f}k\n• Exceedance ≤ {G1_EXC_KW:.2f} kW",
        fontsize=9.8, fontweight="bold", color="#145A32",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#E8F8F5", edgecolor="#27AE60", alpha=0.92),
        zorder=3
    )

    # Visual guide for exact zero band
    ax.axvline(2.2e-4, ls=":", color="#A0AEC0", lw=1.0, zorder=2)
    ax.text(
        2.0e-4, 42000, "← Exact Zero (0.00 kW)",
        fontsize=9.0, color="#4A5568", ha="right", va="bottom", fontweight="bold"
    )

    # Fine-tuned controllers and reference baselines
    methods_cfg = [
        ("LP'", "LP' Oracle", "Baseline", PALETTE["LP'"], "*", 240),
        ("PSM'", "PSM'", "Baseline", PALETTE["PSM'"], "^", 110),
        ("Grid-Only", "Grid-Only", "Baseline", PALETTE["Grid-Only"], "X", 110),
        ("clamped_mb_ppo", "Vanilla PPO", "Ablation", PALETTE["PPO"], "o", 110),
        ("clamped_ppo_residual", "PPO-Residual", "Ablation", PALETTE["PPO-Residual"], "D", 100),
        ("clamped_mb_ppo_lagrangian", "PPO-Lagrangian", "Ablation", PALETTE["PPO-Lagrangian"], "s", 100),
        ("clamped_mb_ppo_shielded", "Shielded PPO", "Env Shield", PALETTE["Shielded PPO_MB"], "p", 110),
        ("PPO-Aux-Shield", "PPO-Aux-Shield", "In-The-Loop Safe RL", PALETTE["PPO-Aux-Shield"], "h", 110),
        ("PPO-Diff-Shield", "PPO-Diff-Shield", "In-The-Loop Safe RL", PALETTE["PPO-Diff-Shield"], "P", 120),
    ]

    zero_x_map = {
        "LP' Oracle": 0.65e-4,
        "PSM'": 0.85e-4,
        "PPO-Diff-Shield": 1.10e-4,
        "Shielded PPO": 1.45e-4,
        "PPO-Aux-Shield": 1.85e-4,
    }

    points = []
    for key, name, cat, col, mk, sz in methods_cfg:
        _purch, _wear, _unserv, tot, s_tot, exc, s_exc, _tp, _s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )
        px = zero_x_map.get(name, exc)
        points.append((name, px, exc, s_exc, tot, s_tot, col, mk, sz))

    for name, px, exc, s_exc, tot, s_tot, col, mk, sz in points:
        if s_exc > 0.01 or s_tot > 50:
            ax.errorbar(
                px, tot, xerr=s_exc if exc > 0.01 else 0, yerr=s_tot, fmt="none",
                ecolor=col, elinewidth=1.5, capsize=4.0, capthick=1.5, zorder=4
            )
        ax.scatter([px], [tot], marker=mk, color=col, s=sz, edgecolors="#FFFFFF", linewidths=1.2, zorder=5)

    # Point text annotations cleanly positioned with soft white pill badges
    points_dict = {p[0]: p for p in points}

    # 1. LP' Oracle (53.3k)
    lp = points_dict["LP' Oracle"]
    ax.text(
        lp[1] * 1.35, lp[4], f"{lp[0]} (€{lp[4]/1e3:.1f}k)",
        ha="left", va="center", fontsize=9.2, fontweight="bold", color=PALETTE["LP'"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["LP'"], alpha=0.92, lw=0.7)
    )

    # 2. PSM' (55.5k) - Sit comfortably below point inside safe zone, avoiding G2 line
    psm = points_dict["PSM'"]
    ax.text(
        psm[1] * 1.35, psm[4] - 900, f"{psm[0]} (€{psm[4]/1e3:.1f}k)",
        ha="left", va="top", fontsize=9.2, fontweight="bold", color=PALETTE["PSM'"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["PSM'"], alpha=0.92, lw=0.7)
    )

    # 3. PPO-Diff-Shield (58.5k) - Sit below lower error bar
    diff = points_dict["PPO-Diff-Shield"]
    ax.text(
        diff[1] * 1.35, diff[4] - diff[5] - 600, f"{diff[0]} (€{diff[4]/1e3:.1f}k)",
        ha="left", va="top", fontsize=9.2, fontweight="bold", color=PALETTE["PPO-Diff-Shield"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["PPO-Diff-Shield"], alpha=0.92, lw=0.7)
    )

    # 4. Shielded PPO (59.5k) - Sit above upper error bar to the right
    sh = points_dict["Shielded PPO"]
    ax.text(
        sh[1] * 1.35, sh[4] + sh[5] + 800, f"{sh[0]} (€{sh[4]/1e3:.1f}k)",
        ha="left", va="bottom", fontsize=9.2, fontweight="bold", color=PALETTE["Shielded PPO_MB"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["Shielded PPO_MB"], alpha=0.92, lw=0.7)
    )

    # 5. PPO-Aux-Shield (60.0k) - Sit cleanly to the right in open space
    aux = points_dict["PPO-Aux-Shield"]
    ax.text(
        aux[1] * 2.2, aux[4], f"{aux[0]} (€{aux[4]/1e3:.1f}k)",
        ha="left", va="center", fontsize=9.2, fontweight="bold", color=PALETTE["PPO-Aux-Shield"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["PPO-Aux-Shield"], alpha=0.92, lw=0.7)
    )

    # 6. PPO-Residual (54.6k) - Sit above marker to avoid horizontal error bar
    res = points_dict["PPO-Residual"]
    ax.text(
        res[1], res[4] + 1100, f"{res[0]} (€{res[4]/1e3:.1f}k)",
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=PALETTE["PPO-Residual"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["PPO-Residual"], alpha=0.92, lw=0.7)
    )

    # 7. PPO-Lagrangian (70.0k) - Sit above vertical error bar
    lag = points_dict["PPO-Lagrangian"]
    ax.text(
        lag[1], lag[4] + lag[5] + 900, f"{lag[0]} (€{lag[4]/1e3:.1f}k)",
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=PALETTE["PPO-Lagrangian"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["PPO-Lagrangian"], alpha=0.92, lw=0.7)
    )

    # 8. Vanilla PPO (48.8k) - Sit above vertical error bar
    van = points_dict["Vanilla PPO"]
    ax.text(
        van[1], van[4] + van[5] + 900, f"{van[0]} (€{van[4]/1e3:.1f}k)",
        ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=PALETTE["PPO"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["PPO"], alpha=0.92, lw=0.7)
    )

    # 9. Grid-Only (45.3k)
    grd = points_dict["Grid-Only"]
    ax.text(
        grd[1] * 1.15, grd[4], f"{grd[0]} (€{grd[4]/1e3:.1f}k)",
        ha="left", va="center", fontsize=9.2, fontweight="bold", color=PALETTE["Grid-Only"],
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec=PALETTE["Grid-Only"], alpha=0.92, lw=0.7)
    )

    # Legend - enlarged and placed cleanly outside to the right so it never hides any data
    legend_handles = [
        Patch(facecolor="#D5F5E3", edgecolor="#27AE60", lw=1.5, ls="--", alpha=0.5, label="Safe Zone (Passes G1 & G2)"),
        Line2D([0], [0], color=PALETTE["Gate"], ls="-.", lw=1.6, label=f"G1 Exceedance Gate ({G1_EXC_KW:.2f} kW)"),
        Line2D([0], [0], color=PALETTE["Gate"], ls="--", lw=1.6, label=f"G2 Cost Gate (€{G2_COST_EUR/1e3:.1f}k)"),
    ]
    for name, px, exc, s_exc, tot, s_tot, col, mk, sz in points:
        if exc <= 1e-4:
            lbl = f"{name}: €{tot/1e3:.1f}k (0.00 kW)"
        else:
            lbl = f"{name}: €{tot/1e3:.1f}k ({exc:.2f} kW)"
        legend_handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor=col, markersize=10.5, label=lbl))

    ax.legend(
        handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True,
        fontsize=10.0, title="Controllers & Gates", title_fontsize=11.0,
        borderaxespad=0.0, labelspacing=0.55, framealpha=0.96, edgecolor="#CBD5E0"
    )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y*1e-3:.0f}k"))
    ax.set_xlabel("Mean Exceedance (kW/step, log scale; exact zero floored at 10⁻⁴)", fontweight="bold", fontsize=11.5)
    ax.set_ylabel("Cost (€)", fontweight="bold", fontsize=11.5)
    ax.set_title(
        "Multi-Objective Pareto Frontier: Cost vs. Grid Exceedance",
        fontsize=13.0, fontweight="bold", pad=16, loc="left"
    )

    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.09, right=0.74)
    out_file = os.path.join(dest_dir, "stress_week_pareto.png")
    fig.savefig(out_file, bbox_inches="tight")
    pareto_file = os.path.join(dest_dir, "pareto.png")
    fig.savefig(pareto_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file} and {pareto_file}")



# ===========================================================================
# STUDY 4: PLAIN PPO VS. PPO-LAGRANGIAN COMPARISON (Validation split)
# ===========================================================================

STUDY4_GROUPS = [
    (
        "Unclamped",
        [
            ("s3_ppo", "Plain PPO", "Ablation", PALETTE["PPO_UNC"]),
            ("s3_ppo_lag", "PPO-Lagrangian", "Ablation", PALETTE["PPO-Lagrangian_UNC"]),
        ],
    ),
    (
        "Clamped Full-Batch",
        [
            ("clamped_ppo", "Plain PPO", "Ablation", PALETTE["PPO_FB"]),
            ("clamped_ppo_lagrangian", "PPO-Lagrangian", "Ablation", PALETTE["PPO-Lagrangian_FB"]),
        ],
    ),
    (
        "Clamped Minibatch",
        [
            ("clamped_mb_ppo", "Plain PPO", "Ablation", PALETTE["PPO"]),
            ("clamped_mb_ppo_lagrangian", "PPO-Lagrangian", "Ablation", PALETTE["PPO-Lagrangian"]),
        ],
    ),
]


def _parse_study4_items():
    """Return x_positions, list of (x, key, display_name, cat, color), group_dividers, group_headers."""
    items = []
    dividers = []
    headers = []

    # 3 pairs with clear visual separation between ablation regimes:
    # Pair 0: x = 0.0, 1.0 (center 0.5)
    # Divider 0: x = 1.75
    # Pair 1: x = 2.5, 3.5 (center 3.0)
    # Divider 1: x = 4.25
    # Pair 2: x = 5.0, 6.0 (center 5.5)
    x_pairs = [
        (0.0, 1.0),
        (2.5, 3.5),
        (5.0, 6.0),
    ]

    for g_idx, (group_name, group_controllers) in enumerate(STUDY4_GROUPS):
        x0, x1 = x_pairs[g_idx]
        headers.append(((x0 + x1) / 2.0, group_name))
        items.append((x0, group_controllers[0][0], group_controllers[0][1], group_controllers[0][2], group_controllers[0][3]))
        items.append((x1, group_controllers[1][0], group_controllers[1][1], group_controllers[1][2], group_controllers[1][3]))
        if g_idx < len(STUDY4_GROUPS) - 1:
            next_x0 = x_pairs[g_idx + 1][0]
            dividers.append((x1 + next_x0) / 2.0)

    return items, dividers, headers


def fig_study4_cost(baselines: dict, out_dir: str = None):
    """Study 4 Cost: Total operational cost breakdown comparing Plain PPO vs PPO-Lagrangian."""
    dest_dir = out_dir or STUDY4_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bar_w = 0.52

    c_lp = float(baselines.get("LP'", {}).get("cost_eur", 53290.51))
    c_psm = float(baselines.get("PSM'", {}).get("cost_eur", 55530.22))
    c_g2 = float(G2_COST_EUR)

    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])
    part_names = ("Grid Energy Purchase (€)", "Battery Degradation Wear (€)", "Unserved Load Penalty (€)")

    controllers, dividers, headers = _parse_study4_items()

    for x_pos, key, _display_name, cat, _colour in controllers:
        purch, wear, unserv, tot, s_tot, _exc, _s_exc, _tp, _s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )

        bottom = 0.0
        for val, col in zip([purch, wear, unserv], part_colours):
            ax.bar(x_pos, val, bottom=bottom, width=bar_w, color=col, edgecolor="#FFFFFF", lw=0.9, zorder=3)
            bottom += val

        if s_tot >= 50.0:
            ax.errorbar(
                x_pos, tot, yerr=s_tot, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=5
            )
            label_text = f"€{tot/1e3:.1f}k\n±€{s_tot/1e3:.1f}k"
        else:
            label_text = f"€{tot/1e3:.1f}k"

        y_label = tot + s_tot + 1200.0
        ax.text(
            x_pos, y_label, label_text,
            ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Horizontal reference lines (LP' Oracle, PSM', G2 Cost Gate)
    ax.axhline(c_lp, color=PALETTE["LP'"], ls="-.", lw=1.6, zorder=2)
    ax.axhline(c_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)
    ax.axhline(c_g2, color=PALETTE["Gate"], ls="--", lw=1.8, zorder=2)

    # Group header annotations
    for hx, htext in headers:
        ax.text(hx, 88500, htext, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#444444")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=20, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.5)
    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(0, 93000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y*1e-3:.0f}k"))
    ax.set_ylabel("Cost (€)", fontweight="bold", fontsize=11.5)

    legend_handles = [
        Patch(facecolor=part_colours[0], label=part_names[0]),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label=f"LP' Oracle (€{c_lp/1e3:.1f}k)"),
        Patch(facecolor=part_colours[1], label=part_names[1]),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' (€{c_psm/1e3:.1f}k)"),
        Patch(facecolor=part_colours[2], label=part_names[2]),
        Line2D([0], [0], color=PALETTE["Gate"], ls="--", lw=1.8, label=f"G2 Cost Gate (€{c_g2/1e3:.1f}k)"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=True, fontsize=9.8
    )
    ax.set_title(
        "Total Operational Cost Breakdown: Plain PPO vs. PPO-Lagrangian",
        fontsize=12.5, fontweight="bold", pad=52
    )

    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.10, right=0.97)
    out_file = os.path.join(dest_dir, "cost.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study4_grid_exceedance(baselines: dict, out_dir: str = None):
    """Study 4 Grid Exceedance: Mean grid import exceedance & violation hours."""
    dest_dir = out_dir or STUDY4_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bar_w = 0.52

    controllers, dividers, headers = _parse_study4_items()

    for x_pos, key, _display_name, cat, colour in controllers:
        _purch, _wear, _unserv, _tot, _s_tot, exc, s_exc, _tp, _s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )
        m_vh, _s_vh = _extract_violation_hours(key, cat, None, baselines)

        ax.bar(x_pos, exc, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.9, zorder=3)
        if s_exc >= 0.005:
            ax.errorbar(
                x_pos, exc, yerr=s_exc, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=5
            )

        pct_vh = (m_vh / 4368.0) * 100.0
        y_label = exc + s_exc + 0.12
        ax.text(
            x_pos, y_label, f"{exc:.2f} kW/step\n({m_vh:.0f}h / {pct_vh:.1f}%)",
            ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Reference lines (Gate and LP' Oracle)
    ax.axhline(G1_EXC_KW, color=PALETTE["Gate"], ls="-.", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["Gate"], ls="-.", lw=1.8, label=f"G1 Exceedance Gate ({G1_EXC_KW:.2f} kW/step)"),
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label="LP' Oracle: 0.00 kW/step (0h)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=10.0
    )

    # Group header annotations
    for hx, htext in headers:
        ax.text(hx, 4.45, htext, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#444444")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=20, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.5)
    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(0, 4.8)
    ax.set_ylabel("Mean Exceedance (kW/step)", fontweight="bold", fontsize=11.0)
    ax.set_title(
        "Mean Grid Import Exceedance & Violation Hours: Plain PPO vs. PPO-Lagrangian",
        fontsize=12.0, fontweight="bold", pad=36
    )

    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.10, right=0.97)
    out_file = os.path.join(dest_dir, "grid_exceedance.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study4_battery_throughput(baselines: dict, out_dir: str = None):
    """Study 4 Battery Throughput: Dedicated vertical bar chart."""
    dest_dir = out_dir or os.path.join(STUDY4_DIR, "battery")
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bar_w = 0.52

    tp_psm = float(baselines.get("PSM'", {}).get("battery_throughput_mwh", 12.348))
    tp_lp = float(baselines.get("LP'", {}).get("battery_throughput_mwh", 24.990))

    controllers, dividers, headers = _parse_study4_items()

    for x_pos, key, _display_name, cat, colour in controllers:
        _purch, wear, _unserv, _tot, _s_tot, _exc, _s_exc, tp, s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, None, baselines, None
        )

        ax.bar(x_pos, tp, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.9, zorder=3)
        if s_tp >= 0.1:
            ax.errorbar(
                x_pos, tp, yerr=s_tp, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=5
            )
            lbl = f"{tp:.1f} MWh\n±{s_tp:.1f}"
        else:
            lbl = f"{tp:.1f} MWh"

        y_label = tp + s_tp + 2.5
        ax.text(
            x_pos, y_label, lbl,
            ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Reference lines (Oracle and PSM')
    ax.axhline(tp_lp, color=PALETTE["LP'"], ls="-.", lw=1.6, zorder=2)
    ax.axhline(tp_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label=f"LP' Oracle ({tp_lp:.1f} MWh)"),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' ({tp_psm:.1f} MWh)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=10.0
    )

    # Group header annotations
    for hx, htext in headers:
        ax.text(hx, 142.0, htext, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#444444")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=20, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.5)
    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(0, 150)
    ax.set_ylabel("Throughput (MWh)", fontweight="bold", fontsize=11.0)
    ax.set_title(
        "Battery Throughput: Plain PPO vs. PPO-Lagrangian",
        fontsize=12.0, fontweight="bold", pad=44
    )

    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.10, right=0.97)
    out_file = os.path.join(dest_dir, "throughput.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_study4_battery_active_soc(baselines: dict, out_dir: str = None):
    """Study 4 Active SoC: Dedicated vertical bar chart."""
    dest_dir = out_dir or os.path.join(STUDY4_DIR, "battery")
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bar_w = 0.52

    soc_psm = float(baselines.get("PSM'", {}).get("active_soc_pct", 31.5))
    soc_lp = float(baselines.get("LP'", {}).get("active_soc_pct", 79.0))

    controllers, dividers, headers = _parse_study4_items()

    for x_pos, key, _display_name, cat, colour in controllers:
        _purch, _wear, _unserv, _tot, _s_tot, _exc, _s_exc, _tp, _s_tp, _, _, soc, s_soc = _extract_metric_row(
            key, cat, None, baselines, None
        )

        ax.bar(x_pos, soc, width=bar_w, color=colour, edgecolor="#FFFFFF", lw=0.9, zorder=3)
        if s_soc >= 0.1:
            ax.errorbar(
                x_pos, soc, yerr=s_soc, fmt="none",
                ecolor="#1A252C", elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=5
            )
            lbl = f"{soc:.1f}%\n±{s_soc:.1f}%"
        else:
            lbl = f"{soc:.1f}%"

        y_label = soc + s_soc + 2.5
        ax.text(
            x_pos, y_label, lbl,
            ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1A252C",
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec="#CBD5E0", alpha=0.92, lw=0.7)
        )

    # Vertical group divider lines
    for div_x in dividers:
        ax.axvline(div_x, color="#BBBBBB", lw=1.2, ls="--", alpha=0.85, zorder=2)

    # Reference lines (Oracle and PSM')
    ax.axhline(soc_lp, color=PALETTE["LP'"], ls="-.", lw=1.6, zorder=2)
    ax.axhline(soc_psm, color=PALETTE["PSM'"], ls=":", lw=1.8, zorder=2)

    ref_handles = [
        Line2D([0], [0], color=PALETTE["LP'"], ls="-.", lw=1.6, label=f"LP' Oracle ({soc_lp:.1f}%)"),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.8, label=f"PSM' ({soc_psm:.1f}%)"),
    ]
    ax.legend(
        handles=ref_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, frameon=True, fontsize=10.0
    )

    # Group header annotations
    for hx, htext in headers:
        ax.text(hx, 108.0, htext, ha="center", va="top", fontsize=10.5, fontweight="bold", color="#444444")

    ax.set_xticks([c[0] for c in controllers])
    ax.set_xticklabels([c[2] for c in controllers], rotation=20, ha="right", rotation_mode="anchor", fontweight="bold", fontsize=10.5)
    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(0, 115.0)
    ax.set_ylabel("Active SoC Time (%)", fontweight="bold", fontsize=11.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.set_title(
        "Active Unpinned SoC Operating Time: Plain PPO vs. PPO-Lagrangian",
        fontsize=12.0, fontweight="bold", pad=44
    )

    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.10, right=0.97)
    out_file = os.path.join(dest_dir, "active_soc.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


# ===========================================================================
# BACKWARD COMPATIBILITY: W49 TRAJECTORY HELPER
# ===========================================================================

def fig_w49_dispatch_trajectory(out_dir: str = None):
    """Generate 4-panel 168-hour dispatch trajectory on W49 for backward compatibility."""
    dest_dir = out_dir or STUDY3_DIR
    os.makedirs(dest_dir, exist_ok=True)

    try:
        from env.smartgrid_env import SmartGridEnv
        from env.cap_shield import CapShield
        from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
        from experiments.train_core import load_agent_for_method, load_config_with_base

        w49_start_idx = 3360
        ep_len = 168
        env = SmartGridEnv(load_config_with_base())
        hours = np.arange(ep_len)

        # Baseline trajectories
        obs, _ = env.reset(options={"start_idx": w49_start_idx})
        hist_grid = {"grid_import": [], "soc": []}
        pol_grid = GridOnlyPolicy()
        for _ in range(ep_len):
            act = pol_grid.get_action(obs)
            obs, _, _, _, info = env.step(act)
            hist_grid["grid_import"].append(info["grid_import_kw"])
            hist_grid["soc"].append(info["soc"])

        obs, _ = env.reset(options={"start_idx": w49_start_idx})
        hist_psm = {"grid_import": [], "soc": []}
        pol_psm = PeakShaveMinPolicy(env.unwrapped)
        for _ in range(ep_len):
            act = pol_psm.get_action(obs)
            obs, _, _, _, info = env.step(act)
            hist_psm["grid_import"].append(info["grid_import_kw"])
            hist_psm["soc"].append(info["soc"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax1.plot(hours, hist_grid["grid_import"], label="Grid-Only", color=PALETTE["Grid-Only"], lw=1.5)
        ax1.plot(hours, hist_psm["grid_import"], label="PSM'", color=PALETTE["PSM'"], lw=1.8)
        ax1.axhline(CONTRACT_CAP_KW, color=PALETTE["Gate"], ls="--", lw=1.5, label="Contract Cap")
        ax1.set_ylabel("Grid Import (kW)", fontweight="bold")
        ax1.legend(loc="upper right")

        ax2.plot(hours, np.array(hist_psm["soc"]) * 100.0, label="PSM' SoC", color=PALETTE["PSM'"], lw=1.8)
        ax2.set_ylabel("Battery SoC (%)", fontweight="bold")
        ax2.set_xlabel("Hour of Stress Week", fontweight="bold")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        out_file = os.path.join(dest_dir, "w49_dispatch_trajectory.png")
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_file}")
    except Exception as exc:
        print(f"Skipping w49_dispatch_trajectory simulation: {exc}")


# ===========================================================================
# PIPELINE ORCHESTRATION & MAIN CLI
# ===========================================================================

def plot_all(baselines: dict, out_dir: str = None):
    """Generate the complete 4-study figure suite across 13 dedicated files."""
    s1_dir = os.path.join(out_dir or FIGS, "study1_ppo_ablation")
    s2_dir = os.path.join(out_dir or FIGS, "study2_capped_ppo")
    s3_dir = os.path.join(out_dir or FIGS, "study3_stress_pareto")
    s4_dir = os.path.join(out_dir or FIGS, "study4_plain_vs_lagrangian")

    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(s3_dir, exist_ok=True)
    os.makedirs(s4_dir, exist_ok=True)
    os.makedirs(os.path.join(s1_dir, "battery"), exist_ok=True)
    os.makedirs(os.path.join(s2_dir, "battery"), exist_ok=True)
    os.makedirs(os.path.join(s4_dir, "battery"), exist_ok=True)

    # Convenience symlink for alternative folder naming
    alt_s4_dir = os.path.join(out_dir or FIGS, "plain_vs_lagrangian")
    if not os.path.exists(alt_s4_dir):
        try:
            os.symlink("study4_plain_vs_lagrangian", alt_s4_dir)
        except OSError:
            pass

    print("Generating 4-study publication figure suite...")

    figures = [
        # Study 1
        ("Study 1 Cost", fig_study1_cost, (baselines, s1_dir)),
        ("Study 1 Grid Exceedance", fig_study1_grid_exceedance, (baselines, s1_dir)),
        ("Study 1 Battery Throughput", fig_study1_battery_throughput, (baselines, os.path.join(s1_dir, "battery"))),
        ("Study 1 Battery Active SoC", fig_study1_battery_active_soc, (baselines, os.path.join(s1_dir, "battery"))),
        # Study 2
        ("Study 2 Cost", fig_study2_cost, (baselines, s2_dir)),
        ("Study 2 Grid Exceedance", fig_study2_grid_exceedance, (baselines, s2_dir)),
        ("Study 2 Unserved Load", fig_study2_unserved_load, (baselines, s2_dir)),
        ("Study 2 Battery Throughput", fig_study2_battery_throughput, (baselines, os.path.join(s2_dir, "battery"))),
        ("Study 2 Battery Active SoC", fig_study2_battery_active_soc, (baselines, os.path.join(s2_dir, "battery"))),
        # Study 3
        ("Study 3 Stress Pareto", fig_study3_stress_pareto, (baselines, s3_dir)),
        # Study 4
        ("Study 4 Cost", fig_study4_cost, (baselines, s4_dir)),
        ("Study 4 Grid Exceedance", fig_study4_grid_exceedance, (baselines, s4_dir)),
        ("Study 4 Battery Throughput", fig_study4_battery_throughput, (baselines, os.path.join(s4_dir, "battery"))),
        ("Study 4 Battery Active SoC", fig_study4_battery_active_soc, (baselines, os.path.join(s4_dir, "battery"))),
    ]

    for label, fn, args in figures:
        try:
            fn(*args)
            print(f"  [OK] {label}")
        except Exception as exc:
            print(f"  [FAIL] {label}: {exc}")
            raise


def main():
    baselines = load_baselines()
    plot_all(baselines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
