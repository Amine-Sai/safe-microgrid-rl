import csv
import io
import json
import math
import os
import shutil
import statistics
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.report.load_results import (
    load_project_constants as _load_project_constants,
    load_baselines as _load_baselines,
    load_per_seed_table,
    load_test_stress_weeks_data,
    extract_metric_row,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES = os.path.join(ROOT, "results", "tables")
RUNS = os.path.join(ROOT, "results", "runs")
FIGS = os.path.join(ROOT, "results", "figures")

OVERVIEW_DIR = os.path.join(FIGS, "overview_and_pareto")
BATTERY_DIR = os.path.join(FIGS, "battery_dynamics")
SAFETY_DIR = os.path.join(FIGS, "safety_and_exceedance")
ECONOMIC_DIR = os.path.join(FIGS, "economic_breakdown")
TRAINING_DIR = os.path.join(FIGS, "training_dynamics")

THEMATIC_DIRS = [OVERVIEW_DIR, BATTERY_DIR, SAFETY_DIR, ECONOMIC_DIR, TRAINING_DIR]


def _resolve_family_and_out_dir(family_name, out_dir, default_dir):
    """Safely resolve family name and target directory supporting legacy positional calls."""
    if out_dir is None:
        if family_name and (os.path.isabs(str(family_name)) or "/" in str(family_name) or "\\" in str(family_name)):
            return "clamped", str(family_name)
        return family_name or "clamped", default_dir
    return family_name or "clamped", out_dir

# High-contrast accessible color palette
PALETTE = {
    "PPO": "#0072B2",               # Blue
    "PPO-Lagrangian": "#D55E00",    # Vermillion / Orange-Red
    "PPO-Residual": "#009E73",      # Emerald Green
    "Shielded PPO": "#CC79A7",      # Mauve / Purple-Pink
    "PPO-Aux-Shield": "#7D3C98",    # Deep Purple
    "PPO-Diff-Shield": "#16A085",   # Teal
    "Grid-Only": "#4A5568",         # Charcoal / Slate
    "PSM'": "#1A5276",              # Dark Navy
    "LP'": "#196F3D",               # Forest Green
    "Shield + Idle": "#718096",     # Cool Grey
    "Shield + PSM'": "#2B6CB0",     # Lapis Blue
    "Gate": "#C0392B",              # Crimson Red
    "Feasible": "#2ECC71",          # Soft Emerald Mint
    "Energy": "#1F77B4",            # Energy purchase blue
    "Wear": "#7F7F7F",              # Degradation wear slate
    "Unserved": "#E67E22",          # Unserved penalty amber
}

GROUPS = [
    ("PPO", "ppo", PALETTE["PPO"]),
    ("PPO-Lagrangian", "ppo_lagrangian", PALETTE["PPO-Lagrangian"]),
    ("PPO-Residual", "ppo_residual", PALETTE["PPO-Residual"]),
    ("Shielded PPO", "ppo_shielded", PALETTE["Shielded PPO"]),
]
COLOURS = dict((g[0], g[2]) for g in GROUPS)

SAFETY_CONFIG = [
    ("Grid-Only", "Grid-Only (Uncontrolled)", "Baseline", None, None, PALETTE["Grid-Only"], "x", "--"),
    ("PSM'", "PSM' (Heuristic Shaving)", "Baseline", None, None, PALETTE["PSM'"], "^", ":"),
    ("LP'", "LP' (Hindsight Oracle)", "Baseline", None, None, PALETTE["LP'"], "D", "-."),
    ("Shield + Idle", "Shield + Idle (0-Battery Control)", "Shield Control", None, None, PALETTE["Shield + Idle"], "v", "--"),
    ("Shield + PSM'", "Shield + PSM' (Heuristic + Shield)", "Shield Control", None, None, PALETTE["Shield + PSM'"], "s", ":"),
    ("Shielded PPO (Clamped)", "Shielded PPO (Clamped, Full-Batch)", "Env Shield", "ppo_shielded", "clamped_ppo_shielded", PALETTE["Shielded PPO"], "o", "-"),
    ("Shielded PPO (Clamped MB)", "Shielded PPO (Clamped, Minibatch)", "Env Shield", "ppo_shielded", "clamped_mb_ppo_shielded", "#884969", "^", "--"),
    ("PPO-Aux-Shield", "PPO-Aux-Shield (Aux. Shield Loss)", "In-The-Loop Safe RL", "ppo_shield_aux", "clamped_ppo_shield_aux", PALETTE["PPO-Aux-Shield"], "D", "-."),
    ("PPO-Diff-Shield", "PPO-Diff-Shield (Diff. Safety Layer)", "In-The-Loop Safe RL", "ppo_diff_shield", "clamped_ppo_diff_shield", PALETTE["PPO-Diff-Shield"], "p", ":"),
]

# Load constants and tables dynamically
PROJECT_CONSTANTS = _load_project_constants()
G1_EXC_KW = PROJECT_CONSTANTS["g1_exceedance_kw"]
G2_COST_EUR = PROJECT_CONSTANTS["g2_cost_eur"]
CONTRACT_CAP_KW = PROJECT_CONSTANTS["contract_cap_kw"]
WEAR_COST_PER_MWH = PROJECT_CONSTANTS["wear_cost_per_mwh"]
PER_SEED_TABLE = load_per_seed_table()
TEST_STRESS_DATA = load_test_stress_weeks_data()


def load_project_constants():
    """Load evaluation gates and physical constants dynamically from project constants."""
    return G1_EXC_KW, G2_COST_EUR


def load_baselines():
    """Load baseline results directly from frozen_baselines.csv."""
    return _load_baselines()
FLOOR = 1e-4
GATE = PALETTE["Gate"]
BASE = PALETTE["Grid-Only"]
COST_YLIM = (43500, 61500)
STEP_TICKS = [50, 100, 150, 200, 250, 300]
EXC_TICKS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

# Configure global publication styling
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
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.8,
    "figure.constrained_layout.use": False,
    "axes.prop_cycle": cycler(color=[g[2] for g in GROUPS]),
})


def fmt_num(v, _pos=None):
    if v == FLOOR or v <= 1e-4:
        return "0"
    return "%g" % v


def fmt_k(v, _pos=None):
    return "%dk" % round(v / 1000.0)


def log_axis(ax, which, ticks, formatter=fmt_num):
    target = ax.xaxis if which == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(formatter))
    target.set_minor_locator(FixedLocator([]))
    target.set_minor_formatter(NullFormatter())


def group_of(agent):
    for label, folder, _colour in GROUPS:
        if agent == folder:
            return label
    return None


def load_runs(family: str = "unclamped"):
    out = dict((g[0], []) for g in GROUPS)
    target_clamped = family != "unclamped"
    if family in ("clamped_mb", "mb"):
        prefix_filter = ("clamped_mb_",)
    elif family in ("clamped", "s3v2", "v2"):
        prefix_filter = ("s3v2_", "clamped_ppo")
    else:
        prefix_filter = ("s3_", "unclamped_")

    for agent in sorted(os.listdir(RUNS)):
        agent_dir = os.path.join(RUNS, agent)
        if not os.path.isdir(agent_dir):
            continue
        label = group_of(agent)
        if label is None:
            continue
        for fname in sorted(os.listdir(agent_dir)):
            if not fname.endswith(".json") or fname.endswith("_grid.json"):
                continue
            if not any(fname.startswith(p) for p in prefix_filter):
                continue
            with io.open(os.path.join(agent_dir, fname), encoding="utf-8") as f:
                run_data = json.load(f)
            rc = run_data.get("resolved_config") or {}
            is_clamped = bool(rc.get("charge_clamp", False))
            if is_clamped == target_clamped:
                out[label].append(run_data)
    return out


def load_safety_runs():
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


def load_baselines():
    out = {}
    path = os.path.join(TABLES, "frozen_baselines.csv")
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "split" in row and row.get("split") != "val":
                continue
            name = row.get("agent") or row.get("method")
            if not name:
                continue
            out[name] = {
                "cost_eur": float(row.get("cost_eur") or row.get("val_cost_eur") or 0.0),
                "exceedance_kw": float(row.get("exceedance_kw") or row.get("mean_exceedance_kw") or 0.0),
                "energy_purchase_eur": float(row.get("energy_purchase_eur") or 0.0),
                "battery_wear_eur": float(row.get("battery_wear_eur") or 0.0),
                "unserved_penalty_eur": float(row.get("unserved_penalty_eur") or 0.0),
                "battery_throughput_mwh": float(row.get("battery_throughput_mwh") or 0.0),
                "active_soc_pct": float(row.get("active_soc_pct") or 0.0),
            }
    return out


def final_metrics(run):
    return run.get("final_checkpoint", {}).get("metrics", {})


def eval_curve(runs):
    steps, costs, excs = [], [], []
    for r in runs:
        ec = r.get("eval_checkpoints", {})
        s = sorted(int(k.split("_")[0]) for k in ec)
        if not s:
            continue
        if not steps:
            steps = s
        costs.append([ec["%d_steps" % x]["total_cost_eur"] for x in s])
        excs.append([ec["%d_steps" % x]["mean_exceedance_kw"] for x in s])
    return steps, costs, excs


def mean_sd(rows):
    n = len(rows[0])
    mean = [statistics.fmean([r[i] for r in rows]) for i in range(n)]
    sd = [statistics.stdev([r[i] for r in rows]) if len(rows) > 1 else 0.0
          for i in range(n)]
    return mean, sd


def group_handle(label, extra=""):
    return Line2D([], [], marker="o", ls="", color=COLOURS[label], markersize=7.5,
                  label=label + extra)


# ===========================================================================
# 1. PER-FAMILY FIGURES (Streamlined, Publication-Quality, No Clutter)
# ===========================================================================

def fig_learning_curves(runs, baselines, family_name="clamped", out_dir=None):
    """Fig 1 - Val cost and Val exceedance over training, mean +/- sd of 5 seeds."""
    family_name, dest_dir = _resolve_family_and_out_dir(family_name, out_dir, TRAINING_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.0), gridspec_kw={"wspace": 0.24})

    ax0 = axes[0]
    max_cost_upper = G2_COST_EUR
    for label, _prefix, colour in GROUPS:
        steps, costs, _ = eval_curve(runs[label])
        if not steps:
            continue
        mean, sd = mean_sd(costs)
        x = [s / 1000 for s in steps]
        upper = [m + s for m, s in zip(mean, sd)]
        if upper:
            max_cost_upper = max(max_cost_upper, max(upper))
        ax0.plot(x, mean, color=colour, marker="o", markersize=4.5, lw=2.0, label=label, zorder=3)
        ax0.fill_between(x, [m - s for m, s in zip(mean, sd)], upper, color=colour, alpha=0.16, linewidth=0)

    ax0.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2, label=f"G2 Cost Gate (€{G2_COST_EUR:,.0f})")
    if "Grid-Only" in baselines:
        c_go = float(baselines["Grid-Only"]["cost_eur"])
        ax0.axhline(c_go, ls="--", color=BASE, lw=1.2, zorder=2,
                    label=f"Grid-Only (€{c_go:,.0f})")
    if "PSM'" in baselines:
        c_psm = float(baselines["PSM'"]["cost_eur"])
        ax0.axhline(c_psm, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2,
                    label=f"PSM' Target (€{c_psm:,.0f})")
    if "LP'" in baselines:
        c_lp = float(baselines["LP'"]["cost_eur"])
        ax0.axhline(c_lp, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2,
                    label=f"LP' Oracle (€{c_lp:,.0f})")

    ymax = 61500 if max_cost_upper <= 60500 else int(math.ceil((max_cost_upper + 2500) / 5000.0) * 5000)
    tick_step = 5000 if ymax <= 70000 else 10000
    ax0.set_ylim(COST_YLIM[0], ymax)
    ax0.yaxis.set_major_locator(FixedLocator(list(range(45000, ymax + 1, tick_step))))
    ax0.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax0.set_ylabel("Validation Total Cost (EUR)")
    ax0.set_xlabel("Training Steps (thousands, evaluated every 50k)")
    ax0.set_xticks(STEP_TICKS)
    ax0.set_title("A. Validation Total Cost Progression (Mean ± 1 SD, n=5 Seeds)")
    ax0.margins(x=0.02)

    ax1 = axes[1]
    for label, _prefix, colour in GROUPS:
        steps, _costs, excs = eval_curve(runs[label])
        if not steps:
            continue
        mean, sd = mean_sd(excs)
        x = [s / 1000 for s in steps]
        y = [max(v, FLOOR) for v in mean]
        lo = [max(v - s, FLOOR) for v, s in zip(mean, sd)]
        hi = [max(v + s, FLOOR) for v, s in zip(mean, sd)]
        ax1.plot(x, y, color=colour, marker="o", markersize=4.5, lw=2.0, label=label, zorder=3)
        ax1.fill_between(x, lo, hi, color=colour, alpha=0.16, linewidth=0)

    ax1.axhline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2, label="G1 Gate (0.05 kW/step)")
    ax1.set_yscale("log")
    ax1.set_ylim(FLOOR, 30)
    log_axis(ax1, "y", EXC_TICKS)
    ax1.set_ylabel("Mean Grid Exceedance\n(kW/step, log scale)")
    ax1.set_xlabel("Training Steps (thousands, evaluated every 50k)")
    ax1.set_title("B. Mean Grid Import Cap Exceedance Progression (Log Scale)")
    ax1.set_xticks(STEP_TICKS)
    ax1.margins(x=0.02)

    handles, labels = ax0.get_legend_handles_labels()
    # Add G1 handle
    handles.append(Line2D([], [], ls="--", color=GATE, lw=1.5, label="G1 Gate (0.05 kW/step)"))
    labels.append("G1 Gate (0.05 kW/step)")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.20, wspace=0.24)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True)
    fname = f"{family_name}_learning_curves.png" if family_name else "learning_curves.png"
    fig.savefig(os.path.join(dest_dir, fname), bbox_inches="tight")
    plt.close(fig)


def fig_cost_vs_exceedance(runs, baselines, family_name="clamped", out_dir=None):
    """Fig 2 - The cost / exceedance frontier, one marker per (method, seed) + mean diamond."""
    family_name, dest_dir = _resolve_family_and_out_dir(family_name, out_dir, SAFETY_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    max_y = G2_COST_EUR
    handles = []

    # Feasible region: below G2 cost and left of G1 exceedance
    ax.fill_between([FLOOR * 0.65, G1_EXC_KW], [43000, 43000], [G2_COST_EUR, G2_COST_EUR],
                    color=PALETTE["Feasible"], alpha=0.12, zorder=0, label="Feasible Region (Passes G1 & G2)")
    ax.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2)
    ax.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2)

    for label, _prefix, colour in GROUPS:
        xs, ys = [], []
        for r in runs[label]:
            m = final_metrics(r)
            xs.append(max(float(m.get("mean_exceedance_kw", 0.0)), FLOOR))
            ys.append(float(m.get("total_cost_eur", 0.0)))
        if not xs:
            continue
        max_y = max(max_y, max(ys))
        ax.scatter(xs, ys, s=55, color=colour, alpha=0.6, edgecolors="white", linewidths=0.8, zorder=4)
        m_x, m_y = statistics.fmean(xs), statistics.fmean(ys)
        ax.scatter([m_x], [m_y], marker="D", s=130, facecolors=colour, edgecolors="black", linewidths=1.3, zorder=5)
        handles.append(Line2D([], [], marker="D", ls="", color=colour, markeredgecolor="black",
                              markersize=8, label=f"{label} (Mean)"))

    # Reference baselines
    base_markers = [
        ("Grid-Only", "s", PALETTE["Grid-Only"], 130),
        ("PSM'", "^", PALETTE["PSM'"], 140),
        ("LP'", "*", PALETTE["LP'"], 210),
    ]
    for name, marker, colour, msize in base_markers:
        if name not in baselines:
            continue
        b = baselines[name]
        by = float(b["cost_eur"])
        bx = max(float(b["exceedance_kw"]), FLOOR)
        max_y = max(max_y, by)
        ax.scatter([bx], [by], marker=marker, s=msize, facecolors=colour, edgecolors="black", linewidths=1.2, zorder=6)
        handles.append(Line2D([], [], marker=marker, ls="", color=colour, markeredgecolor="black",
                              markersize=8.5, label=f"{name} ({by/1000:.1f}k, {bx:.2f} kW)"))

    ymax = 62000 if max_y <= 60500 else int(math.ceil((max_y + 2500) / 5000.0) * 5000)
    tick_step = 5000 if ymax <= 75000 else 10000

    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.65, 12)
    ax.set_ylim(44000, ymax)
    log_axis(ax, "x", EXC_TICKS)
    ax.yaxis.set_major_locator(FixedLocator(list(range(45000, ymax + 1, tick_step))))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))

    # Add gate handles
    handles.extend([
        Patch(facecolor=PALETTE["Feasible"], alpha=0.25, label="Feasible Region"),
        Line2D([], [], ls="--", color=GATE, lw=1.5, label="G1 (0.05 kW) & G2 (58.3k EUR) Gates"),
        Line2D([], [], marker="o", ls="", color="#888888", markersize=6, alpha=0.6, label="Individual Seed"),
    ])

    ax.set_xlabel("Mean Exceedance Above Cap (kW/step, log scale; 0 floored at 10⁻⁴)")
    ax.set_ylabel("Validation Total Cost (EUR)")
    ax.set_title("Validation Total Cost vs. Mean Grid Exceedance Frontier")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.22)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True)
    fname = f"{family_name}_cost_vs_exceedance.png" if family_name else "cost_vs_exceedance.png"
    fig.savefig(os.path.join(dest_dir, fname), bbox_inches="tight")
    plt.close(fig)


def fig_lambda(runs, family_name="clamped", out_dir=None):
    """Fig 3 - Lagrange multiplier lambda over training for constrained variants."""
    family_name, dest_dir = _resolve_family_and_out_dir(family_name, out_dir, TRAINING_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    handles = []
    for label in ("PPO-Lagrangian", "PPO-Residual"):
        colour = COLOURS[label]
        ends = []
        for r in runs[label]:
            traj = (r.get("lambda_summary") or {}).get("trajectory", [])
            if not traj:
                continue
            ax.plot([t["step"] / 1000 for t in traj],
                    [max(float(t["lambda"]), 1e-3) for t in traj],
                    color=colour, alpha=0.55, lw=1.3, zorder=2)
            ends.append(float(traj[-1]["lambda"]))
        if ends:
            handles.append(Line2D([], [], color=colour, lw=2.4,
                                  label=f"{label} (Final λ at 300k: {statistics.fmean(ends):.2f} ± {statistics.stdev(ends) if len(ends)>1 else 0:.2f})"))

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 100)
    ax.set_xlim(-5, 305)
    log_axis(ax, "y", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    ax.set_xticks(STEP_TICKS)
    ax.set_xlabel("Training Steps (thousands)")
    ax.set_ylabel("Dual Multiplier λ (log scale)")
    ax.set_title("Lagrange Multiplier λ Trajectory over Training")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.16)
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=True)
    fname = f"{family_name}_lambda_trajectory.png" if family_name else "lambda_trajectory.png"
    fig.savefig(os.path.join(dest_dir, fname), bbox_inches="tight")
    plt.close(fig)


def fig_per_seed(runs, baselines, family_name="clamped", out_dir=None):
    """Fig 4 - Per-seed cost and exceedance with gates."""
    family_name, dest_dir = _resolve_family_and_out_dir(family_name, out_dir, OVERVIEW_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.14})

    labels = [g[0] for g in GROUPS]
    ypos = list(range(len(labels)))[::-1]

    ax0 = axes[0]
    max_cost = G2_COST_EUR
    for y, label in zip(ypos, labels):
        vals = sorted(float(final_metrics(r).get("total_cost_eur", 0.0)) for r in runs[label])
        if not vals:
            continue
        max_cost = max(max_cost, max(vals))
        ax0.scatter(vals, [y] * len(vals), s=70, color=COLOURS[label], alpha=0.65, zorder=4,
                    edgecolors="white", linewidths=0.9)
        ax0.scatter([statistics.fmean(vals)], [y], marker="D", s=110,
                    facecolors=COLOURS[label], edgecolors="black", linewidths=1.3, zorder=5)

    ax0.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2, label=f"G2 Gate (€{G2_COST_EUR:,.0f})")
    if "Grid-Only" in baselines:
        ax0.axvline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.2, zorder=2)
    if "PSM'" in baselines:
        ax0.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    if "LP'" in baselines:
        ax0.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)

    xmax = 61500 if max_cost <= 60000 else int(math.ceil((max_cost + 2500) / 5000.0) * 5000)
    ax0.set_xlim(44000, xmax)
    ax0.xaxis.set_major_locator(FixedLocator(list(range(45000, xmax + 1, 5000))))
    ax0.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax0.set_yticks(ypos)
    ax0.set_yticklabels(labels, fontweight="normal")
    ax0.set_xlabel("Validation Total Cost (EUR)")
    ax0.set_title("A. Validation Cost per Seed")
    ax0.margins(y=0.18)

    ax1 = axes[1]
    for y, label in zip(ypos, labels):
        vals = sorted(max(float(final_metrics(r).get("mean_exceedance_kw", 0.0)), FLOOR) for r in runs[label])
        if not vals:
            continue
        ax1.scatter(vals, [y] * len(vals), s=70, color=COLOURS[label], alpha=0.65, zorder=4,
                    edgecolors="white", linewidths=0.9)
        ax1.scatter([statistics.fmean(vals)], [y], marker="D", s=110,
                    facecolors=COLOURS[label], edgecolors="black", linewidths=1.3, zorder=5)

    ax1.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2, label="G1 Gate (0.05 kW/step)")
    ax1.set_xscale("log")
    ax1.set_xlim(FLOOR * 0.65, 12)
    log_axis(ax1, "x", EXC_TICKS)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontweight="normal")
    ax1.set_xlabel("Mean Grid Exceedance (kW/step, log scale; 0 floored)")
    ax1.set_title("B. Mean Grid Exceedance per Seed")
    ax1.margins(y=0.18)

    handles = [
        Line2D([], [], marker="o", ls="", color="#888888", markersize=7, alpha=0.7, label="Single Seed (n=5)"),
        Line2D([], [], marker="D", ls="", color="#333333", markeredgecolor="black", markersize=8, label="Method Mean"),
        Line2D([], [], ls="--", color=GATE, lw=1.5, label="G1 / G2 Gates"),
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only"),
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label="PSM' Target"),
        Line2D([], [], ls="-.", color=PALETTE["LP'"], lw=1.3, label="LP' Oracle"),
    ]
    fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.18)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=True)
    fname = f"{family_name}_per_seed_results.png" if family_name else "per_seed_results.png"
    fig.savefig(os.path.join(dest_dir, fname), bbox_inches="tight")
    plt.close(fig)


def fig_cost_breakdown(runs, baselines, family_name="clamped", out_dir=None):
    """Fig 5 - Cost breakdown, exceedance, and battery throughput across methods in family."""
    family_name, dest_dir = _resolve_family_and_out_dir(family_name, out_dir, ECONOMIC_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.5), sharey=True, gridspec_kw={"width_ratios": [1.1, 0.9, 0.9], "wspace": 0.16})
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    parts = ("energy_purchase_eur", "battery_wear_eur", "unserved_penalty_eur")
    part_names = ("Grid Energy Bought", "Battery Wear", "Unserved Penalty")
    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])

    ypos = list(range(len(GROUPS)))[::-1]
    labels = [g[0] for g in GROUPS]
    bar_h = 0.52

    all_tots, all_excs, all_tps, group_data = [], [], [], []

    for y, (label, _prefix, colour) in zip(ypos, GROUPS):
        totals = []
        for i, part in enumerate(parts):
            vals = [float(final_metrics(r).get(part, 0.0)) for r in runs[label]]
            v = statistics.fmean(vals) if vals else 0.0
            totals.append(v)
        tot = sum(totals)
        all_tots.append(tot)

        excs = [float(final_metrics(r).get("mean_exceedance_kw", 0.0)) for r in runs[label]]
        mean_exc = statistics.fmean(excs) if excs else 0.0
        all_excs.append(mean_exc)

        tps = [float(final_metrics(r).get("battery_wear_eur", 0.0)) / 20.0 for r in runs[label]]
        mean_tp = statistics.fmean(tps) if tps else 0.0
        std_tp = statistics.stdev(tps) if len(tps) > 1 else 0.0
        all_tps.append(mean_tp + std_tp)

        group_data.append((y, label, colour, totals, tot, mean_exc, mean_tp, std_tp))

    max_tot = max([G2_COST_EUR] + all_tots + [float(b["cost_eur"]) for b in baselines.values() if "cost_eur" in b])
    max_exc = max([G1_EXC_KW, 3.26] + all_excs)
    max_tp = max([25.0] + all_tps)

    for y, label, colour, totals, tot, mean_exc, mean_tp, std_tp in group_data:
        # Panel A: Stacked cost bars
        left = 0.0
        for i, v in enumerate(totals):
            ax1.barh(y, v, left=left, height=bar_h, color=part_colours[i], edgecolor="white", linewidth=0.8)
            left += v
        ax1.text(tot + 600, y, f"{tot/1000:.1f}k", va="center", fontsize=9, fontweight="bold", color="#333333")

        # Panel B: Exceedance bar
        ax2.barh(y, mean_exc, height=bar_h, color=colour, edgecolor="white", linewidth=0.8)
        exc_lbl = f"{mean_exc:.2f} kW" if mean_exc > 0.005 else "0.00 kW"
        ax2.text(mean_exc + 0.08, y, exc_lbl, va="center", fontsize=9, color="#222222",
                 fontweight="bold" if mean_exc <= 0.05 else "normal")

        # Panel C: Battery Throughput bar
        ax3.barh(y, mean_tp, xerr=std_tp, height=bar_h, color=colour, edgecolor="white", linewidth=0.8, capsize=3)
        ax3.text(mean_tp + std_tp + 1.0, y, f"{mean_tp:.1f} MWh", va="center", fontsize=9, color="#222222")

    # Panel A styling
    ax1.set_xlim(0, max_tot * 1.15)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=10, fontweight="normal")
    ax1.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax1.set_xlabel("Validation Total Cost (EUR)")
    ax1.set_title("A. Cost Breakdown", fontsize=11.5)
    ax1.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.4, zorder=2)
    if "Grid-Only" in baselines:
        ax1.axvline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.2, zorder=2)
    if "PSM'" in baselines:
        ax1.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=PALETTE["PSM'"], lw=1.2, zorder=2)
    if "LP'" in baselines:
        ax1.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color=PALETTE["LP'"], lw=1.2, zorder=2)

    # Panel B styling
    ax2.set_xlim(0, max_exc * 1.20)
    ax2.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.4, zorder=2)
    ax2.set_xlabel("Mean Exceedance (kW/step)")
    ax2.set_title("B. Cap Exceedance vs. G1", fontsize=11.5)

    # Panel C styling
    psm_tp = float(baselines.get("PSM'", {}).get("battery_throughput_mwh", 0.0))
    lp_tp = float(baselines.get("LP'", {}).get("battery_throughput_mwh", 0.0))
    ax3.set_xlim(0, max_tp * 1.25)
    ax3.axvline(psm_tp, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    ax3.axvline(lp_tp, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)
    ax3.set_xlabel("Battery Energy Throughput (MWh, 26 Val Weeks)")
    ax3.set_title("C. Battery Dispatch Usage", fontsize=11.5)

    cost_handles = [Patch(facecolor=part_colours[i], label=part_names[i]) for i in range(len(parts))]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.4, label="G1 / G2 Gates"),
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only"),
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label="PSM' Target"),
        Line2D([], [], ls="-.", color=PALETTE["LP'"], lw=1.3, label="LP' Oracle"),
    ]

    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
    fig.legend(handles=cost_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=7, frameon=True)
    fname = f"{family_name}_cost_breakdown.png" if family_name else "cost_breakdown.png"
    fig.savefig(os.path.join(dest_dir, fname), bbox_inches="tight")
    plt.close(fig)


def fig_family_summary(runs, baselines, family_name="clamped", out_dir=None):
    """Unified 2x2 publication dashboard summarizing learning curves, Pareto frontier, and cost composition."""
    family_name, dest_dir = _resolve_family_and_out_dir(family_name, out_dir, OVERVIEW_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 12.0))
    ax_cost, ax_exc = axes[0, 0], axes[0, 1]
    ax_pareto, ax_bar = axes[1, 0], axes[1, 1]

    # Panel (0,0): Cost curves
    max_c_upper = G2_COST_EUR
    for label, _p, col in GROUPS:
        steps, costs, _ = eval_curve(runs[label])
        if not steps:
            continue
        mean, sd = mean_sd(costs)
        x = [s / 1000 for s in steps]
        ax_cost.plot(x, mean, color=col, marker="o", markersize=4, lw=1.8, label=label)
        ax_cost.fill_between(x, [m - s for m, s in zip(mean, sd)], [m + s for m, s in zip(mean, sd)],
                             color=col, alpha=0.15, linewidth=0)
        max_c_upper = max(max_c_upper, max([m + s for m, s in zip(mean, sd)]))

    ax_cost.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.4, label=f"G2 Gate (€{G2_COST_EUR/1000:.1f}k)")
    if "PSM'" in baselines:
        c_psm = float(baselines["PSM'"]["cost_eur"])
        ax_cost.axhline(c_psm, ls=":", color=PALETTE["PSM'"], lw=1.3,
                        label=f"PSM' (€{c_psm/1000:.1f}k)")
    if "LP'" in baselines:
        c_lp = float(baselines["LP'"]["cost_eur"])
        ax_cost.axhline(c_lp, ls="-.", color=PALETTE["LP'"], lw=1.3,
                        label=f"LP' Oracle (€{c_lp/1000:.1f}k)")
    ymax_c = 61500 if max_c_upper <= 60500 else int(math.ceil((max_c_upper + 2500) / 5000.0) * 5000)
    ax_cost.set_ylim(44000, ymax_c)
    ax_cost.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_ylabel("Validation Total Cost (EUR)")
    ax_cost.set_xlabel("Training Steps (thousands)")
    ax_cost.set_title("A. Validation Cost Progression", fontsize=11.5)
    ax_cost.set_xticks(STEP_TICKS)

    # Panel (0,1): Exceedance curves
    for label, _p, col in GROUPS:
        steps, _c, excs = eval_curve(runs[label])
        if not steps:
            continue
        mean, sd = mean_sd(excs)
        x = [s / 1000 for s in steps]
        y = [max(v, FLOOR) for v in mean]
        lo = [max(v - s, FLOOR) for v, s in zip(mean, sd)]
        hi = [max(v + s, FLOOR) for v, s in zip(mean, sd)]
        ax_exc.plot(x, y, color=col, marker="o", markersize=4, lw=1.8, label=label)
        ax_exc.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)

    ax_exc.axhline(G1_EXC_KW, ls="--", color=GATE, lw=1.4, label="G1 Gate (0.05 kW/step)")
    ax_exc.set_yscale("log")
    ax_exc.set_ylim(FLOOR, 30)
    log_axis(ax_exc, "y", EXC_TICKS)
    ax_exc.set_ylabel("Mean Exceedance (kW/step, log scale)")
    ax_exc.set_xlabel("Training Steps (thousands)")
    ax_exc.set_title("B. Mean Grid Cap Exceedance Progression", fontsize=11.5)
    ax_exc.set_xticks(STEP_TICKS)

    # Panel (1,0): Pareto scatter
    ax_pareto.fill_between([FLOOR * 0.65, G1_EXC_KW], [43000, 43000], [G2_COST_EUR, G2_COST_EUR],
                           color=PALETTE["Feasible"], alpha=0.12, zorder=0, label="Feasible Region")
    ax_pareto.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.4, zorder=2)
    ax_pareto.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.4, zorder=2)
    for label, _p, col in GROUPS:
        xs = [max(float(final_metrics(r).get("mean_exceedance_kw", 0.0)), FLOOR) for r in runs[label]]
        ys = [float(final_metrics(r).get("total_cost_eur", 0.0)) for r in runs[label]]
        if xs:
            ax_pareto.scatter(xs, ys, s=50, color=col, alpha=0.55, edgecolors="white", linewidths=0.8, zorder=4)
            ax_pareto.scatter([statistics.fmean(xs)], [statistics.fmean(ys)], marker="D", s=110,
                              facecolors=col, edgecolors="black", linewidths=1.2, zorder=5)

    for name, marker, colour in [("Grid-Only", "s", PALETTE["Grid-Only"]), ("PSM'", "^", PALETTE["PSM'"]), ("LP'", "*", PALETTE["LP'"])]:
        if name in baselines:
            ax_pareto.scatter([max(float(baselines[name]["exceedance_kw"]), FLOOR)], [float(baselines[name]["cost_eur"])],
                              marker=marker, s=140 if marker != "*" else 190, facecolors=colour, edgecolors="black", linewidths=1.2, zorder=6)

    ax_pareto.set_xscale("log")
    ax_pareto.set_xlim(FLOOR * 0.65, 12)
    ax_pareto.set_ylim(44000, ymax_c)
    log_axis(ax_pareto, "x", EXC_TICKS)
    ax_pareto.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_pareto.set_xlabel("Mean Exceedance (kW/step, log scale)")
    ax_pareto.set_ylabel("Validation Total Cost (EUR)")
    ax_pareto.set_title("C. Validation Cost vs. Exceedance Frontier", fontsize=11.5)

    # Panel (1,1): Cost breakdown bar
    parts = ("energy_purchase_eur", "battery_wear_eur", "unserved_penalty_eur")
    part_names = ("Grid Energy Bought", "Battery Wear", "Unserved Penalty")
    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])
    ypos = list(range(len(GROUPS)))[::-1]
    bar_h = 0.50

    for y, (label, _p, _c) in zip(ypos, GROUPS):
        left = 0.0
        tot = 0.0
        for i, part in enumerate(parts):
            vals = [float(final_metrics(r).get(part, 0.0)) for r in runs[label]]
            v = statistics.fmean(vals) if vals else 0.0
            ax_bar.barh(y, v, left=left, height=bar_h, color=part_colours[i], edgecolor="white", linewidth=0.8)
            left += v
            tot += v
        ax_bar.text(tot + 600, y, f"{tot/1000:.1f}k", va="center", fontsize=8.5, fontweight="bold")

    ax_bar.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.4, zorder=2)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels([g[0] for g in GROUPS], fontsize=9.5, fontweight="normal")
    ax_bar.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_bar.set_xlabel("Validation Cost Breakdown (EUR)")
    ax_bar.set_title("D. Economic Breakdown", fontsize=11.5)
    ax_bar.set_xlim(0, max_c_upper * 1.12)

    handles = [Line2D([], [], color=g[2], lw=2.0, marker="o", markersize=6, label=g[0]) for g in GROUPS]
    handles += [
        Line2D([], [], ls="--", color=GATE, lw=1.4, label="G1 / G2 Gates"),
        Patch(facecolor=PALETTE["Feasible"], alpha=0.25, label="Feasible Region"),
        Patch(facecolor=part_colours[0], label="Energy Purchase"),
        Patch(facecolor=part_colours[1], label="Battery Wear"),
        Patch(facecolor=part_colours[2], label="Unserved Penalty"),
    ]

    fig.suptitle(f"Unified Performance Dashboard — Family: {family_name.upper()}", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10, hspace=0.30, wspace=0.25)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=True)
    fname = f"{family_name}_family_summary.png" if family_name else "family_summary.png"
    fig.savefig(os.path.join(dest_dir, fname), bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 2. CROSS-FAMILY COMPARISON FIGURES (Consolidated, Grouped, Publication-Grade)
# ===========================================================================

def fig_unified_pareto(all_families, baselines, out_dir=None, safety_runs=None):
    """Unified Pareto frontier: Cost vs. Exceedance across environments, baselines, and safety shields."""
    dest_dir = out_dir or OVERVIEW_DIR
    os.makedirs(dest_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.0, 8.2))

    fam_styles = [
        ("unclamped", "Unclamped", "x", 7.0),
        ("clamped", "Clamped (Full-Batch)", "o", 6.5),
        ("clamped_mb", "Clamped Minibatch", "^", 7.5),
    ]

    # Shaded feasible quadrant (x <= G1, y <= G2)
    ax.fill_between([FLOOR * 0.5, G1_EXC_KW], [43000, 43000], [G2_COST_EUR, G2_COST_EUR],
                    color=PALETTE["Feasible"], alpha=0.12, zorder=1)
    ax.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.6, zorder=2)
    ax.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.6, zorder=2)

    # Label Feasible Region clearly
    ax.text(0.0015, 50500, f"FEASIBLE REGION\n(G1 & G2 Gates Passed)\n• Cost ≤ €{G2_COST_EUR:,.0f}\n• Exceedance ≤ {G1_EXC_KW:.2f} kW",
            fontsize=9.5, fontweight="bold", color="#145A32",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#2ECC71", alpha=0.9), zorder=3)

    # Deterministic separation slots for zero-exceedance methods
    zero_slots = {
        "LP'": 0.70e-4,
        "PSM'": 0.82e-4,
        "Shield + PSM'": 0.94e-4,
        "Shield + Idle": 1.06e-4,
        "Shielded PPO_clamped": 1.18e-4,
        "Shielded PPO_clamped_mb": 1.30e-4,
        "Shielded PPO_unclamped": 1.18e-4,
        "PPO-Aux-Shield": 1.54e-4,
        "PPO-Diff-Shield": 1.66e-4,
    }

    # Plot RL Families
    for label, _prefix, colour in GROUPS:
        for fam_key, _fam_name, marker, msize in fam_styles:
            runs = all_families.get(fam_key, {}).get(label, [])
            if not runs:
                continue
            costs = [float(final_metrics(r).get("total_cost_eur", 0.0)) for r in runs]
            excs = [float(final_metrics(r).get("mean_exceedance_kw", 0.0)) for r in runs]

            mean_cost = statistics.fmean(costs)
            mean_exc = statistics.fmean(excs)

            is_zero = mean_exc <= 1e-4
            slot_key = f"{label}_{fam_key}"
            base_x = zero_slots.get(slot_key, 1.25e-4) if is_zero else max(mean_exc, FLOOR)

            # Scatter individual seeds with subtle jitter
            for s_idx, (c, e) in enumerate(zip(costs, excs)):
                if is_zero or e <= 1e-4:
                    jitter = (s_idx - 2) * 0.025e-4
                    x_pt = base_x + jitter
                else:
                    x_pt = max(e, FLOOR)
                ax.scatter(x_pt, c, marker=marker, color=colour, alpha=0.45,
                           s=msize ** 2 * 0.45, zorder=3)

            # Scatter method mean marker
            if marker == "x":
                ax.scatter([base_x], [mean_cost], marker=marker, color=colour, linewidths=2.2,
                           s=msize ** 2 * 1.8, zorder=5)
            else:
                ax.scatter([base_x], [mean_cost], marker=marker, facecolors=colour, edgecolors="black",
                           linewidths=1.3, s=msize ** 2 * 1.8, zorder=5)

    # Plot In-The-Loop Safety Shields
    if safety_runs:
        safety_specs = [
            ("PPO-Aux-Shield", "D", 7.5, PALETTE["PPO-Aux-Shield"]),
            ("PPO-Diff-Shield", "p", 8.0, PALETTE["PPO-Diff-Shield"]),
        ]
        for s_label, s_marker, s_msize, s_col in safety_specs:
            s_runs = safety_runs.get(s_label, [])
            if not s_runs:
                continue
            costs = [float(final_metrics(r).get("total_cost_eur", 0.0)) for r in s_runs]
            excs = [float(final_metrics(r).get("mean_exceedance_kw", 0.0)) for r in s_runs]
            mean_cost = statistics.fmean(costs)
            base_x = zero_slots.get(s_label, 1.55e-4)

            for s_idx, (c, _e) in enumerate(zip(costs, excs)):
                jitter = (s_idx - 2) * 0.025e-4
                ax.scatter(base_x + jitter, c, marker=s_marker, color=s_col, alpha=0.55,
                           s=s_msize ** 2 * 0.55, zorder=4)

            ax.scatter([base_x], [mean_cost], marker=s_marker, facecolors=s_col, edgecolors="black",
                       linewidths=1.4, s=s_msize ** 2 * 1.8, zorder=6)

    # Plot Baselines
    base_plot_info = [
        ("Grid-Only", "s", PALETTE["Grid-Only"], 140, None),
        ("PSM'", "^", PALETTE["PSM'"], 140, zero_slots["PSM'"]),
        ("LP'", "*", PALETTE["LP'"], 220, zero_slots["LP'"]),
        ("Shield + Idle", "v", PALETTE["Shield + Idle"], 120, zero_slots["Shield + Idle"]),
        ("Shield + PSM'", "s", PALETTE["Shield + PSM'"], 120, zero_slots["Shield + PSM'"]),
    ]
    for name, marker, colour, msize, z_x in base_plot_info:
        if name not in baselines:
            continue
        by = float(baselines[name]["cost_eur"])
        bx = z_x if z_x is not None else max(float(baselines[name]["exceedance_kw"]), FLOOR)
        ax.scatter([bx], [by], marker=marker, s=msize, facecolors=colour, edgecolors="black",
                   linewidths=1.3, zorder=6)

    # Reference lines for baselines
    if "Grid-Only" in baselines:
        ax.axhline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.1, zorder=2)
    if "PSM'" in baselines:
        ax.axhline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=PALETTE["PSM'"], lw=1.2, zorder=2)
    if "LP'" in baselines:
        ax.axhline(float(baselines["LP'"]["cost_eur"]), ls="-.", color=PALETTE["LP'"], lw=1.2, zorder=2)

    # Visual guide for exact zero band
    ax.axvline(1.85e-4, ls=":", color="#AAAAAA", lw=1.0, zorder=2)
    ax.text(1.75e-4, 44400, "← Strict Zero Exceedance (0 kW)", fontsize=8, color="#666666", ha="right")

    # Dynamic reference points from loaded tables
    lp_cost = float(baselines.get("LP'", {}).get("cost_eur", 0.0))
    psm_cost = float(baselines.get("PSM'", {}).get("cost_eur", 0.0))
    grid_cost = float(baselines.get("Grid-Only", {}).get("cost_eur", 0.0))
    grid_exc = float(baselines.get("Grid-Only", {}).get("exceedance_kw", 0.0))

    sh_mb_runs = (all_families or {}).get("clamped_mb", {}).get("Shielded PPO", [])
    sh_cost = statistics.fmean([float(final_metrics(r).get("total_cost_eur", 0.0)) for r in sh_mb_runs]) if sh_mb_runs else 59526.0

    res_mb_runs = (all_families or {}).get("clamped_mb", {}).get("PPO-Residual", [])
    res_cost = statistics.fmean([float(final_metrics(r).get("total_cost_eur", 0.0)) for r in res_mb_runs]) if res_mb_runs else 59700.0
    res_exc = statistics.fmean([float(final_metrics(r).get("mean_exceedance_kw", 0.0)) for r in res_mb_runs]) if res_mb_runs else 0.28

    # Non-overlapping callout annotations
    ax.annotate(f"LP' Oracle (€{lp_cost/1000:.1f}k, 0 kW)\nTheoretical Upper Bound",
                xy=(zero_slots["LP'"], lp_cost),
                xytext=(3.0e-4, 46500),
                arrowprops=dict(arrowstyle="->", color="#196F3D", lw=1.2),
                fontsize=8.5, fontweight="bold", color="#196F3D",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#196F3D", alpha=0.9))

    ax.annotate(f"PSM' / Shield+PSM' (€{psm_cost/1000:.1f}k, 0 kW)\nTuned Heuristic Baseline",
                xy=(zero_slots["PSM'"], psm_cost),
                xytext=(3.0e-4, 54500),
                arrowprops=dict(arrowstyle="->", color="#1A5276", lw=1.2),
                fontsize=8.5, fontweight="bold", color="#1A5276",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1A5276", alpha=0.9))

    ax.annotate(f"Shielded PPO & Shields (~€{sh_cost/1000:.1f}k, 0 kW)\nStrict Zero Violation Guarantee",
                xy=(zero_slots["Shielded PPO_clamped_mb"], sh_cost),
                xytext=(3.0e-4, 62000),
                arrowprops=dict(arrowstyle="->", color="#884969", lw=1.2),
                fontsize=8.5, fontweight="bold", color="#884969",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CC79A7", alpha=0.9))

    ax.annotate(f"Grid-Only (€{grid_cost/1000:.1f}k, {grid_exc:.2f} kW)\nUncontrolled Peak Import",
                xy=(grid_exc, grid_cost),
                xytext=(0.15, 45000),
                arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.2),
                fontsize=8.5, fontweight="bold", color="#4A5568",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#4A5568", alpha=0.9))

    ax.annotate(f"PPO-Residual Cluster (~€{res_cost/1000:.1f}k)\nPasses G1 Soft, Fails Strict",
                xy=(res_exc, res_cost),
                xytext=(0.015, 65500),
                arrowprops=dict(arrowstyle="->", color="#009E73", lw=1.2),
                fontsize=8.5, fontweight="bold", color="#009E73",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#009E73", alpha=0.9))

    # Axis limits and formatters
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.45, 14)
    log_axis(ax, "x", EXC_TICKS)
    ax.set_ylim(43500, 74000)
    ax.yaxis.set_major_locator(FixedLocator(list(range(45000, 74001, 5000))))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.set_xlabel("Mean Grid Import Cap Exceedance (kW/step, log scale; exact zero floored at 10⁻⁴)")
    ax.set_ylabel("Validation Total Cost (EUR)")
    ax.set_title("Unified Pareto Frontier: Total Cost vs. Grid Exceedance across All Environments", fontsize=12.5)

    # Clean structured legend placed on the right
    method_handles = [
        Line2D([], [], color=PALETTE["PPO"], marker="s", ls="", markersize=7.5, label="PPO (Unconstrained)"),
        Line2D([], [], color=PALETTE["PPO-Lagrangian"], marker="s", ls="", markersize=7.5, label="PPO-Lagrangian"),
        Line2D([], [], color=PALETTE["PPO-Residual"], marker="s", ls="", markersize=7.5, label="PPO-Residual"),
        Line2D([], [], color=PALETTE["Shielded PPO"], marker="s", ls="", markersize=7.5, label="Shielded PPO (Env Shield)"),
        Line2D([], [], color=PALETTE["PPO-Aux-Shield"], marker="D", ls="", markersize=7.0, label="PPO-Aux-Shield (Aux Loss)"),
        Line2D([], [], color=PALETTE["PPO-Diff-Shield"], marker="p", ls="", markersize=7.5, label="PPO-Diff-Shield (Diff Layer)"),
    ]
    env_handles = [
        Line2D([], [], color="#333333", marker=mk, ls="", markersize=ms, label=fn)
        for _, fn, mk, ms in fam_styles
    ]
    c_go = float(baselines.get("Grid-Only", {}).get("cost_eur", 0.0))
    c_psm = float(baselines.get("PSM'", {}).get("cost_eur", 0.0))
    c_lp = float(baselines.get("LP'", {}).get("cost_eur", 0.0))
    c_idle = float(baselines.get("Shield + Idle", {}).get("cost_eur", 0.0))
    c_s_psm = float(baselines.get("Shield + PSM'", {}).get("cost_eur", 0.0))

    ref_handles = [
        Line2D([], [], color=PALETTE["Grid-Only"], marker="s", ls="", markersize=7.5, label=f"Grid-Only (€{c_go/1000:.1f}k)"),
        Line2D([], [], color=PALETTE["PSM'"], marker="^", ls="", markersize=7.5, label=f"PSM' Target (€{c_psm/1000:.1f}k)"),
        Line2D([], [], color=PALETTE["LP'"], marker="*", ls="", markersize=9.5, label=f"LP' Oracle (€{c_lp/1000:.1f}k)"),
        Line2D([], [], color=PALETTE["Shield + Idle"], marker="v", ls="", markersize=7.0, label=f"Shield + Idle (€{c_idle/1000:.1f}k)"),
        Line2D([], [], color=PALETTE["Shield + PSM'"], marker="s", ls="", markersize=7.0, label=f"Shield + PSM' (€{c_s_psm/1000:.1f}k)"),
        Line2D([], [], ls="--", color=GATE, lw=1.5, label=f"G1 ({G1_EXC_KW:.2f} kW) & G2 (€{G2_COST_EUR/1000:.1f}k)"),
        Patch(facecolor=PALETTE["Feasible"], alpha=0.25, label="Feasible Region (Passes G1 & G2)"),
    ]

    all_legend_handles = (
        [Line2D([], [], color="none", label="── RL FAMILIES ──")] + method_handles +
        [Line2D([], [], color="none", label="── ENVIRONMENTS ──")] + env_handles +
        [Line2D([], [], color="none", label="── BASELINES & GATES ──")] + ref_handles
    )

    fig.subplots_adjust(left=0.09, right=0.76, top=0.93, bottom=0.10)
    ax.legend(handles=all_legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.50),
              frameon=True, fontsize=8.5, labelspacing=0.45)
    fig.savefig(os.path.join(dest_dir, "unified_pareto.png"), bbox_inches="tight")
    plt.close(fig)


def _extract_metric_row(key, cat, all_families, baselines, safety_runs):
    """Safely extract economic, safety, and battery metrics for any controller from persistent results."""
    return extract_metric_row(
        key=key,
        cat=cat,
        all_families=all_families,
        baselines=baselines,
        safety_runs=safety_runs,
        per_seed_table=PER_SEED_TABLE,
    )


def fig_ppo_ablation_cost_breakdown(all_families, baselines, out_dir=None):
    """Study Case 1: Clean, spacious cost composition bar chart for PPO ablations."""
    dest_dir = out_dir or ECONOMIC_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 5.0))

    items = [
        ("unclamped_ppo", "PPO [Unclamped]", "Ablation", "#7FB3D5", "Legacy Simulation"),
        ("clamped_ppo", "PPO [Clamped Full-Batch]", "Ablation", "#5499C7", "Full-Batch Inaction"),
        ("clamped_mb_ppo", "PPO [Clamped Minibatch]", "Ablation", PALETTE["PPO"], "★ Best PPO Baseline"),
    ]

    parts = ("energy_purchase_eur", "battery_wear_eur", "unserved_penalty_eur")
    part_names = ("Grid Energy Purchases", "Battery Wear Degradation", "Unserved Load Penalty")
    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])

    bar_h = 0.55
    y_pos = 0.0
    y_ticks, y_labels = [], []
    row_data = []

    for key, display_name, cat, colour, tag in reversed(items):
        y = y_pos
        y_pos += 1.05
        y_ticks.append(y)
        y_labels.append(display_name)

        purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, all_families, baselines, None
        )
        row_data.append((y, display_name, colour, tag, [purch, wear, unserv], tot, s_tot))

    for y, _name, col, tag, totals, tot, s_tot in row_data:
        left = 0.0
        for i, val in enumerate(totals):
            ax.barh(y, val, left=left, height=bar_h, color=part_colours[i], edgecolor="white", linewidth=0.8)
            left += val

        is_best = "Best" in tag
        cost_lbl = f"€{tot/1000:.1f}k" + (f"  ({tag})" if is_best else "")
        ax.text(tot + 600, y, cost_lbl, va="center", fontsize=9.5,
                fontweight="bold" if is_best else "normal", color="#222222")

    # Reference gate line
    ax.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2)
    ax.text(G2_COST_EUR + 400, y_pos - 0.45, f"G2 Cost Gate (€{G2_COST_EUR/1000:.1f}k)",
            fontsize=8.5, fontweight="bold", color=GATE, va="center")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10.5, fontweight="normal")
    ax.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax.set_xlim(0, 76000)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.set_xlabel("Validation Total Cost (EUR, 26 Val Weeks / 4,368h)", fontsize=10.5, fontweight="bold")
    ax.set_title("Study Case 1: PPO Environment & Optimizer Ablation — Cost Composition",
                 fontsize=12, fontweight="bold", pad=12)

    legend_handles = [Patch(facecolor=part_colours[i], label=part_names[i]) for i in range(len(parts))]
    legend_handles.append(Line2D([], [], ls="--", color=GATE, lw=1.5, label=f"G2 Cost Gate (€{G2_COST_EUR/1000:.1f}k)"))
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True, fontsize=9.5)

    fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.18)
    out_path = os.path.join(dest_dir, "ppo_ablation_cost_breakdown.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_safe_benchmark_cost_breakdown(all_families, baselines, out_dir=None, safety_runs=None):
    """Study Case 2: Cross-Architecture Safe RL Benchmark stacked cost and safety breakdown."""
    dest_dir = out_dir or ECONOMIC_DIR
    os.makedirs(dest_dir, exist_ok=True)
    if safety_runs is None:
        safety_runs = load_safety_runs()

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 8.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 0.85, 0.85], "wspace": 0.22})
    ax_cost, ax_exc, ax_batt = axes[0], axes[1], axes[2]

    parts = ("energy_purchase_eur", "battery_wear_eur", "unserved_penalty_eur")
    part_names = ("Grid Energy Purchases", "Battery Wear Degradation", "Unserved Load Penalty")
    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])

    items = [
        # Reference Baselines
        ("Grid-Only", "Grid-Only (Unmanaged)", "Baseline", PALETTE["Grid-Only"]),
        ("PSM'", "PSM' (Tuned Heuristic)", "Baseline", PALETTE["PSM'"]),
        ("LP'", "LP' Oracle (Theoretical Bound)", "Baseline", PALETTE["LP'"]),
        ("Shield + Idle", "Shield + Idle (0-Battery)", "Baseline", PALETTE["Shield + Idle"]),
        ("Shield + PSM'", "Shield + PSM' (Heuristic + Shield)", "Baseline", PALETTE["Shield + PSM'"]),
        (None, "DIVIDER_1", "Divider", None),
        # Unconstrained Representative
        ("clamped_mb_ppo", "PPO [Clamped Minibatch]", "Unconstrained", PALETTE["PPO"]),
        (None, "DIVIDER_2", "Divider", None),
        # Safe RL Architectures
        ("clamped_mb_ppo_lagrangian", "PPO-Lagrangian [Clamped MB]", "Safe RL", PALETTE["PPO-Lagrangian"]),
        ("clamped_ppo_residual", "PPO-Residual [Clamped Full-Batch]", "Safe RL", PALETTE["PPO-Residual"]),
        ("clamped_ppo_shielded", "Shielded PPO [Clamped Full-Batch]", "Safe RL", PALETTE["Shielded PPO"]),
        ("clamped_mb_ppo_shielded", "Shielded PPO [Clamped Minibatch]", "Safe RL", "#884969"),
        ("PPO-Aux-Shield", "PPO-Aux-Shield (Aux. Shield Loss)", "In-The-Loop", PALETTE["PPO-Aux-Shield"]),
        ("PPO-Diff-Shield", "PPO-Diff-Shield (Diff. Safety Layer)", "In-The-Loop", PALETTE["PPO-Diff-Shield"]),
    ]

    y_pos = 0.0
    y_ticks, y_labels = [], []
    row_data = []
    bar_h = 0.58
    dividers = []

    for key, display_name, cat, colour in reversed(items):
        if key is None:
            dividers.append(y_pos)
            y_pos += 0.85
            continue
        y = y_pos
        y_pos += 1.05
        y_ticks.append(y)
        y_labels.append(display_name)

        purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, _, _, _, _ = _extract_metric_row(
            key, cat, all_families, baselines, safety_runs
        )
        row_data.append((y, display_name, colour, [purch, wear, unserv], tot, exc, tp, s_tp))

    max_tot = max([G2_COST_EUR] + [r[4] for r in row_data])
    max_exc = max([G1_EXC_KW, 3.26] + [r[5] for r in row_data])
    max_tp = max([25.0] + [r[6] for r in row_data])

    for div_y in dividers:
        for ax in (ax_cost, ax_exc, ax_batt):
            ax.axhline(div_y, color="#CCCCCC", lw=1.0, ls="--", alpha=0.8, zorder=2)

    for y, name, col, totals, tot, exc, tp, std_tp in row_data:
        is_highlight = "Diff-Shield" in name or "Clamped Minibatch]" in name
        # Panel A: Stacked Cost Bars
        left = 0.0
        for i, val in enumerate(totals):
            ax_cost.barh(y, val, left=left, height=bar_h, color=part_colours[i], edgecolor="white", linewidth=0.8)
            left += val
        ax_cost.text(tot + 800, y, f"€{tot/1000:.1f}k", va="center", fontsize=8.8,
                     fontweight="bold" if is_highlight else "normal", color="#333333")

        # Panel B: Exceedance Bar
        ax_exc.barh(y, exc, height=bar_h, color=col, edgecolor="white", linewidth=0.8)
        exc_str = "0.00 kW" if exc <= 1e-4 else f"{exc:.2f} kW"
        ax_exc.text(exc + 0.08, y, exc_str, va="center", fontsize=8.8,
                    color="#145A32" if exc <= 0.05 else "#922B21",
                    fontweight="bold" if exc <= 0.05 else "normal")

        # Panel C: Battery Throughput Bar
        ax_batt.barh(y, tp, xerr=std_tp if std_tp > 0 else None, height=bar_h, color=col,
                     edgecolor="white", linewidth=0.8, capsize=2.5)
        ax_batt.text(tp + (std_tp or 0) + 1.2, y, f"{tp:.1f} MWh", va="center", fontsize=8.8,
                     fontweight="bold" if is_highlight else "normal", color="#222222")

    # Panel A styling
    ax_cost.set_yticks(y_ticks)
    ax_cost.set_yticklabels(y_labels, fontsize=9.5, fontweight="normal")
    ax_cost.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax_cost.set_xlim(0, max_tot * 1.25)
    ax_cost.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_xlabel("Validation Total Cost (EUR)", fontsize=10.5, fontweight="bold")
    ax_cost.set_title("A. Economic Breakdown", fontsize=11.5, pad=10)
    ax_cost.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2)
    if "PSM'" in baselines:
        ax_cost.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    if "LP'" in baselines:
        ax_cost.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)

    # Panel B styling
    ax_exc.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax_exc.set_xlim(0, max_exc * 1.28)
    ax_exc.set_xlabel("Mean Exceedance (kW/step)", fontsize=10.5, fontweight="bold")
    ax_exc.set_title("B. Cap Exceedance vs. G1", fontsize=11.5, pad=10)
    ax_exc.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2)

    # Panel C styling
    psm_tp = float(baselines.get("PSM'", {}).get("battery_throughput_mwh", 0.0))
    lp_tp = float(baselines.get("LP'", {}).get("battery_throughput_mwh", 0.0))
    psm_cost = float(baselines.get("PSM'", {}).get("cost_eur", 0.0))
    lp_cost = float(baselines.get("LP'", {}).get("cost_eur", 0.0))

    ax_batt.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax_batt.set_xlim(0, max_tp * 1.28)
    ax_batt.set_xlabel("Battery Throughput (MWh, 4,368h)", fontsize=10.5, fontweight="bold")
    ax_batt.set_title("C. Battery Dispatch Usage", fontsize=11.5, pad=10)
    ax_batt.axvline(psm_tp, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    ax_batt.axvline(lp_tp, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)

    cost_handles = [Patch(facecolor=part_colours[i], label=part_names[i]) for i in range(len(parts))]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.5, label="G1 / G2 Gates"),
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label=f"PSM' Target (€{psm_cost/1000:.1f}k / {psm_tp:.1f} MWh)"),
        Line2D([], [], ls="-.", color=PALETTE["LP'"], lw=1.3, label=f"LP' Oracle (€{lp_cost/1000:.1f}k / {lp_tp:.1f} MWh)"),
    ]

    fig.subplots_adjust(left=0.25, right=0.97, top=0.90, bottom=0.15)
    fig.legend(handles=cost_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=True, fontsize=9.0)
    out_path = os.path.join(dest_dir, "safe_benchmark_cost_breakdown.png")
    fig.savefig(out_path, bbox_inches="tight")
    # Also save alias to grouped_cost_breakdown.png for backward compatibility
    alias_path = os.path.join(dest_dir, "grouped_cost_breakdown.png")
    shutil.copyfile(out_path, alias_path)
    plt.close(fig)
    fig.legend(handles=cost_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=True, fontsize=9.0)
    out_path = os.path.join(dest_dir, "safe_benchmark_cost_breakdown.png")
    fig.savefig(out_path, bbox_inches="tight")
    # Also save alias to grouped_cost_breakdown.png for backward compatibility
    alias_path = os.path.join(dest_dir, "grouped_cost_breakdown.png")
    shutil.copyfile(out_path, alias_path)
    plt.close(fig)


def fig_grouped_cost_breakdown(all_families, baselines, out_dir=None, safety_runs=None):
    """Backward-compatible wrapper generating both Study Case 1 and Study Case 2 cost figures."""
    fig_ppo_ablation_cost_breakdown(all_families, baselines, out_dir)
    fig_safe_benchmark_cost_breakdown(all_families, baselines, out_dir, safety_runs)


def fig_safety_architectures_comparison(safety_runs, baselines, out_dir=None):
    """Multi-panel comparison of the Safety & Shielding Spectrum across 4 core dimensions."""
    dest_dir = out_dir or SAFETY_DIR
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(23.5, 8.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 0.90, 0.90, 0.90], "wspace": 0.22})
    ax_cost, ax_exc, ax_batt, ax_soc = axes[0], axes[1], axes[2], axes[3]

    items = [
        ("Grid-Only", "Grid-Only (Uncontrolled)", "Baseline", PALETTE["Grid-Only"]),
        ("PSM'", "PSM' (Heuristic Shaving)", "Baseline", PALETTE["PSM'"]),
        ("LP'", "LP' Oracle (Hindsight Dispatch)", "Baseline", PALETTE["LP'"]),
        (None, "", "Divider", None),
        ("Shield + Idle", "Shield + Idle (0-Battery Control)", "Shield Control", PALETTE["Shield + Idle"]),
        ("Shield + PSM'", "Shield + PSM' (Heuristic + Shield)", "Shield Control", PALETTE["Shield + PSM'"]),
        (None, "", "Divider", None),
        ("Shielded PPO (Clamped)", "Shielded PPO (Clamped Full-Batch)", "Post-Hoc Shield", PALETTE["Shielded PPO"]),
        ("Shielded PPO (Clamped MB)", "Shielded PPO (Clamped Minibatch)", "Post-Hoc Shield", "#884969"),
        (None, "", "Divider", None),
        ("PPO-Aux-Shield", "PPO-Aux-Shield (Aux. Shield Loss)", "In-The-Loop Safe RL", PALETTE["PPO-Aux-Shield"]),
        ("PPO-Diff-Shield", "PPO-Diff-Shield (Diff. Safety Layer)", "In-The-Loop Safe RL", PALETTE["PPO-Diff-Shield"]),
    ]

    parts = ("energy_purchase_eur", "battery_wear_eur", "unserved_penalty_eur")
    part_names = ("Grid Energy", "Battery Wear", "Unserved Penalty")
    part_colours = (PALETTE["Energy"], PALETTE["Wear"], PALETTE["Unserved"])

    y_pos = 0.0
    y_ticks, y_labels = [], []
    bar_h = 0.65
    row_data = []

    for key, display_name, cat, colour in reversed(items):
        if key is None:
            y_pos += 0.70
            continue
        y = y_pos
        y_pos += 1.10
        y_ticks.append(y)
        y_labels.append(display_name)

        purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, _, _, soc, s_soc = _extract_metric_row(
            key, cat, None, baselines, safety_runs
        )
        row_data.append((y, display_name, colour, [purch, wear, unserv], tot, s_tot, exc, s_exc, tp, s_tp, soc, s_soc))

    max_tot = max([G2_COST_EUR] + [r[4] for r in row_data])
    max_exc = max([G1_EXC_KW, 3.26] + [r[6] for r in row_data])
    max_tp = max([30.0] + [r[8] for r in row_data])

    for y, _name, col, totals, tot, s_tot, exc, _s_exc, tp, s_tp, soc, s_soc in row_data:
        # Panel A: Stacked Cost Bars
        left = 0.0
        for i, val in enumerate(totals):
            ax_cost.barh(y, val, left=left, height=bar_h, color=part_colours[i], edgecolor="white", linewidth=0.8)
            left += val
        if s_tot > 0:
            ax_cost.errorbar(tot, y, xerr=s_tot, color="black", capsize=3, fmt="none", lw=1.2)
        ax_cost.text(tot + (s_tot or 0) + 700, y, f"{tot/1000:.1f}k", va="center", fontsize=8.5, fontweight="bold", color="#333333")

        # Panel B: Exceedance
        ax_exc.barh(y, exc, height=bar_h, color=col, edgecolor="white", linewidth=0.8)
        exc_str = "0.00 kW" if exc <= 1e-4 else f"{exc:.2f} kW"
        ax_exc.text(exc + 0.08, y, exc_str, va="center", fontsize=8.5,
                    color="#145A32" if exc <= 0.05 else "#922B21",
                    fontweight="bold" if exc <= 0.05 else "normal")

        # Panel C: Battery Throughput
        ax_batt.barh(y, tp, xerr=s_tp if s_tp > 0 else None, height=bar_h, color=col, edgecolor="white", linewidth=0.8, capsize=2.5)
        ax_batt.text(tp + (s_tp or 0) + 1.2, y, f"{tp:.1f} MWh", va="center", fontsize=8.5, color="#222222")

        # Panel D: Active SoC %
        ax_soc.barh(y, soc, xerr=s_soc if s_soc > 0 else None, height=bar_h, color=col, edgecolor="white", linewidth=0.8, capsize=2.5)
        ax_soc.text(soc + (s_soc or 0) + 2.0, y, f"{soc:.1f}%", va="center", fontsize=8.5, color="#222222")

    # Panel A styling
    ax_cost.set_yticks(y_ticks)
    ax_cost.set_yticklabels(y_labels, fontsize=9.5, fontweight="normal")
    ax_cost.set_ylim(min(y_ticks) - 0.8, max(y_ticks) + 0.8)
    ax_cost.set_xlim(0, max_tot * 1.18)
    ax_cost.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_xlabel("Validation Total Cost (EUR)")
    ax_cost.set_title("A. Economic Breakdown", fontsize=11.5)
    ax_cost.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2)
    if "PSM'" in baselines:
        ax_cost.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    if "LP'" in baselines:
        ax_cost.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)

    # Panel B styling
    ax_exc.set_ylim(min(y_ticks) - 0.8, max(y_ticks) + 0.8)
    ax_exc.set_xlim(0, max_exc * 1.25)
    ax_exc.set_xlabel("Mean Exceedance (kW/step)")
    ax_exc.set_title("B. Safety Enforcement (G1)", fontsize=11.5)
    ax_exc.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2)

    # Panel C styling
    psm_tp = float(baselines.get("PSM'", {}).get("battery_throughput_mwh", 0.0))
    lp_tp = float(baselines.get("LP'", {}).get("battery_throughput_mwh", 0.0))
    psm_soc = float(baselines.get("PSM'", {}).get("active_soc_pct", 0.0))
    lp_soc = float(baselines.get("LP'", {}).get("active_soc_pct", 0.0))

    ax_batt.set_ylim(min(y_ticks) - 0.8, max(y_ticks) + 0.8)
    ax_batt.set_xlim(0, max_tp * 1.28)
    ax_batt.set_xlabel("Battery Throughput (MWh)")
    ax_batt.set_title("C. Battery Dispatch Usage", fontsize=11.5)
    ax_batt.axvline(psm_tp, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    ax_batt.axvline(lp_tp, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)

    # Panel D styling
    ax_soc.set_ylim(min(y_ticks) - 0.8, max(y_ticks) + 0.8)
    ax_soc.set_xlim(0, 125)
    ax_soc.set_xlabel("Active Unpinned SoC Time (%)")
    ax_soc.set_title("D. Battery Operating Dynamics", fontsize=11.5)
    ax_soc.axvline(psm_soc, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    ax_soc.axvline(lp_soc, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)

    cost_handles = [Patch(facecolor=part_colours[i], label=part_names[i]) for i in range(len(parts))]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.5, label="G1 / G2 Gates"),
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label="PSM' Target"),
        Line2D([], [], ls="-.", color=PALETTE["LP'"], lw=1.3, label="LP' Oracle"),
    ]

    fig.subplots_adjust(left=0.22, right=0.98, top=0.91, bottom=0.15)
    fig.legend(handles=cost_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=True)
    fig.savefig(os.path.join(dest_dir, "safety_architectures_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fig_safety_learning_curves(safety_runs, baselines, out_dir=None):
    """Clean multi-panel learning dynamics comparing RL Shielding Architectures over 300k steps."""
    dest_dir = out_dir or TRAINING_DIR
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), gridspec_kw={"wspace": 0.22})
    ax_cost, ax_unserv, ax_wear = axes[0], axes[1], axes[2]

    plot_agents = [
        ("Shielded PPO (Clamped)", PALETTE["Shielded PPO"], "o", "-", 2.0),
        ("Shielded PPO (Clamped MB)", "#884969", "^", "--", 2.0),
        ("PPO-Aux-Shield", PALETTE["PPO-Aux-Shield"], "D", "-.", 2.2),
        ("PPO-Diff-Shield", PALETTE["PPO-Diff-Shield"], "p", ":", 2.4),
    ]

    max_cost = G2_COST_EUR
    for label, col, mk, ls, lw in plot_agents:
        runs = (safety_runs or {}).get(label, [])
        if not runs:
            continue
        steps, costs, _ = eval_curve(runs)
        if not steps:
            continue
        mean_c, sd_c = mean_sd(costs)
        x = [s / 1000 for s in steps]
        ax_cost.plot(x, mean_c, color=col, ls=ls, lw=lw, marker=mk, markersize=5.0, label=label, zorder=4)
        ax_cost.fill_between(x, [m - s for m, s in zip(mean_c, sd_c)],
                             [m + s for m, s in zip(mean_c, sd_c)], color=col, alpha=0.14, linewidth=0)
        max_cost = max(max_cost, max([m + s for m, s in zip(mean_c, sd_c)]))

        # Unserved Penalty curve
        u_series = []
        w_series = []
        for r in runs:
            ec = r.get("eval_checkpoints", {})
            s_sorted = sorted(int(k.split("_")[0]) for k in ec)
            u_vals = [float(ec["%d_steps" % st].get("unserved_penalty_eur", 0.0)) for st in s_sorted]
            w_vals = [float(ec["%d_steps" % st].get("battery_wear_eur", 0.0)) for st in s_sorted]
            u_series.append(u_vals)
            w_series.append(w_vals)

        if u_series and len(u_series[0]) == len(steps):
            m_u, s_u = mean_sd(u_series)
            ax_unserv.plot(x, m_u, color=col, ls=ls, lw=lw, marker=mk, markersize=5.0, label=label, zorder=4)
            ax_unserv.fill_between(x, [m - s for m, s in zip(m_u, s_u)],
                                   [m + s for m, s in zip(m_u, s_u)], color=col, alpha=0.14, linewidth=0)

        if w_series and len(w_series[0]) == len(steps):
            m_w, s_w = mean_sd(w_series)
            ax_wear.plot(x, m_w, color=col, ls=ls, lw=lw, marker=mk, markersize=5.0, label=label, zorder=4)
            ax_wear.fill_between(x, [m - s for m, s in zip(m_w, s_w)],
                                 [m + s for m, s in zip(m_w, s_w)], color=col, alpha=0.14, linewidth=0)

    # Dynamic baseline reference metrics
    psm_cost = float(baselines.get("PSM'", {}).get("cost_eur", 0.0))
    idle_cost = float(baselines.get("Shield + Idle", {}).get("cost_eur", 0.0))
    psm_unserv = float(baselines.get("PSM'", {}).get("unserved_penalty_eur", 0.0))
    idle_unserv = float(baselines.get("Shield + Idle", {}).get("unserved_penalty_eur", 0.0))
    psm_wear = float(baselines.get("PSM'", {}).get("battery_wear_eur", 0.0))
    idle_wear = float(baselines.get("Shield + Idle", {}).get("battery_wear_eur", 0.0))

    # Panel A: Cost
    ax_cost.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2, label=f"G2 Cost Gate (€{G2_COST_EUR/1000:.1f}k)")
    if "PSM'" in baselines:
        ax_cost.axhline(psm_cost, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2, label=f"PSM' Target (€{psm_cost/1000:.1f}k)")
    if "Shield + Idle" in baselines:
        ax_cost.axhline(idle_cost, ls="--", color=PALETTE["Shield + Idle"], lw=1.3, zorder=2, label=f"Shield + Idle (€{idle_cost/1000:.1f}k)")
    ax_cost.set_ylim(44000, 62000)
    ax_cost.yaxis.set_major_locator(FixedLocator(list(range(45000, 62001, 5000))))
    ax_cost.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_ylabel("Validation Total Cost (EUR)")
    ax_cost.set_xlabel("Training Steps (thousands)")
    ax_cost.set_xticks(STEP_TICKS)
    ax_cost.set_title("A. Validation Total Cost Progression (Mean ± 1 SD)")

    # Panel B: Unserved Penalty
    ax_unserv.axhline(idle_unserv, ls="--", color=PALETTE["Shield + Idle"], lw=1.3, zorder=2, label=f"Shield + Idle (€{idle_unserv/1000:.1f}k)")
    ax_unserv.axhline(psm_unserv, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2, label=f"PSM' Target (€{psm_unserv/1000:.1f}k)")
    ax_unserv.set_ylim(8500, 16000)
    ax_unserv.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_unserv.set_ylabel("Unserved Load Penalty (EUR)")
    ax_unserv.set_xlabel("Training Steps (thousands)")
    ax_unserv.set_xticks(STEP_TICKS)
    ax_unserv.set_title("B. Unserved Load Penalty Progression")

    # Panel C: Battery Wear
    ax_wear.axhline(psm_wear, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2, label=f"PSM' Wear (€{psm_wear:.0f})")
    ax_wear.axhline(idle_wear, ls="--", color=PALETTE["Shield + Idle"], lw=1.3, zorder=2, label=f"Shield + Idle (€{idle_wear:.0f})")
    ax_wear.set_ylim(-50, 2400)
    ax_wear.set_ylabel("Battery Wear Cost (EUR)")
    ax_wear.set_xlabel("Training Steps (thousands)")
    ax_wear.set_xticks(STEP_TICKS)
    ax_wear.set_title("C. Battery Degradation Wear Progression")

    handles = [Line2D([], [], color=col, ls=ls, lw=lw, marker=mk, markersize=6, label=label)
               for label, col, mk, ls, lw in plot_agents]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.5, label="G2 Gate (58.3k EUR)"),
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label="PSM' Target"),
        Line2D([], [], ls="--", color=PALETTE["Shield + Idle"], lw=1.3, label="Shield + Idle Baseline"),
    ]

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.20, wspace=0.22)
    fig.legend(handles=handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True)
    fig.savefig(os.path.join(dest_dir, "safety_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)


def fig_learning_and_constraint_dynamics(all_families, baselines, out_dir=None, safety_runs=None):
    """Multi-panel learning dynamics (Val Cost, Exceedance, and dual multiplier lambda evolution)."""
    dest_dir = out_dir or TRAINING_DIR
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(21.0, 6.0), gridspec_kw={"wspace": 0.24})
    ax_cost, ax_exc, ax_lam = axes[0], axes[1], axes[2]

    rep_methods = [
        ("PPO", "clamped_mb", PALETTE["PPO"], "-", "o"),
        ("PPO-Lagrangian", "clamped_mb", PALETTE["PPO-Lagrangian"], "-", "^"),
        ("PPO-Residual", "clamped_mb", PALETTE["PPO-Residual"], "-", "s"),
        ("Shielded PPO", "clamped_mb", PALETTE["Shielded PPO"], "-", "D"),
    ]

    for m_label, f_key, colour, ls, mk in rep_methods:
        runs = all_families.get(f_key, {}).get(m_label, [])
        if not runs:
            continue
        steps, costs, excs = eval_curve(runs)
        if not steps:
            continue
        mean_c, sd_c = mean_sd(costs)
        x = [s / 1000 for s in steps]
        ax_cost.plot(x, mean_c, color=colour, ls=ls, lw=1.8, marker=mk, markersize=4.5,
                     label=f"{m_label} [Clamped MB]", zorder=4)
        ax_cost.fill_between(x, [m - s for m, s in zip(mean_c, sd_c)],
                             [m + s for m, s in zip(mean_c, sd_c)], color=colour, alpha=0.13, linewidth=0)

        mean_e, sd_e = mean_sd(excs)
        y_exc = [max(v, FLOOR) for v in mean_e]
        ax_exc.plot(x, y_exc, color=colour, ls=ls, lw=1.8, marker=mk, markersize=4.5, label=f"{m_label} [Clamped MB]", zorder=4)
        ax_exc.fill_between(x, [max(m - s, FLOOR) for m, s in zip(mean_e, sd_e)],
                            [max(m + s, FLOOR) for m, s in zip(mean_e, sd_e)], color=colour, alpha=0.13, linewidth=0)

    # In-the-loop shields on cost and exceedance
    if safety_runs:
        for s_label, col, mk in [("PPO-Aux-Shield", PALETTE["PPO-Aux-Shield"], "D"), ("PPO-Diff-Shield", PALETTE["PPO-Diff-Shield"], "p")]:
            s_runs = safety_runs.get(s_label, [])
            if not s_runs:
                continue
            steps, costs, excs = eval_curve(s_runs)
            if not steps:
                continue
            mean_c, sd_c = mean_sd(costs)
            x = [s / 1000 for s in steps]
            ax_cost.plot(x, mean_c, color=col, ls="-.", lw=2.0, marker=mk, markersize=4.5, label=s_label, zorder=4)
            ax_cost.fill_between(x, [m - s for m, s in zip(mean_c, sd_c)], [m + s for m, s in zip(mean_c, sd_c)],
                                 color=col, alpha=0.13, linewidth=0)

            mean_e, _ = mean_sd(excs)
            y_exc = [max(v, FLOOR) for v in mean_e]
            ax_exc.plot(x, y_exc, color=col, ls="-.", lw=2.0, marker=mk, markersize=4.5, label=s_label, zorder=4)

    # Panel A: Cost
    ax_cost.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2, label=f"G2 Gate (€{G2_COST_EUR/1000:.1f}k)")
    if "PSM'" in baselines:
        c_psm = float(baselines["PSM'"]["cost_eur"])
        ax_cost.axhline(c_psm, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2,
                        label=f"PSM' (€{c_psm/1000:.1f}k)")
    if "LP'" in baselines:
        c_lp = float(baselines["LP'"]["cost_eur"])
        ax_cost.axhline(c_lp, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2,
                        label=f"LP' Oracle (€{c_lp/1000:.1f}k)")
    ax_cost.set_ylim(44000, 75000)
    ax_cost.yaxis.set_major_locator(FixedLocator(list(range(45000, 75001, 5000))))
    ax_cost.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_ylabel("Val Cost (EUR)")
    ax_cost.set_xlabel("Training Steps (thousands)")
    ax_cost.set_xticks(STEP_TICKS)
    ax_cost.set_title("A. Validation Total Cost Progression (EUR)")

    # Panel B: Exceedance
    ax_exc.axhline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2, label="G1 Gate (0.05 kW/step)")
    ax_exc.set_yscale("log")
    ax_exc.set_ylim(FLOOR, 30)
    log_axis(ax_exc, "y", EXC_TICKS)
    ax_exc.set_ylabel("Mean Exceedance\n(kW/step, log scale)")
    ax_exc.set_xlabel("Training Steps (thousands)")
    ax_exc.set_xticks(STEP_TICKS)
    ax_exc.set_title("B. Mean Grid Exceedance Progression (Log Scale)")

    # Panel C: Lambda Dynamics
    lam_methods = [
        ("PPO-Lagrangian", "clamped_mb", PALETTE["PPO-Lagrangian"], "-"),
        ("PPO-Residual", "clamped_mb", PALETTE["PPO-Residual"], "-"),
    ]
    for m_label, f_key, col, ls in lam_methods:
        runs = all_families.get(f_key, {}).get(m_label, [])
        for r in runs:
            traj = (r.get("lambda_summary") or {}).get("trajectory", [])
            if not traj:
                continue
            ax_lam.plot([t["step"] / 1000 for t in traj],
                        [max(float(t["lambda"]), 1e-3) for t in traj],
                        color=col, alpha=0.55, lw=1.3, ls=ls, zorder=2)

    ax_lam.set_yscale("log")
    ax_lam.set_ylim(1e-3, 100)
    log_axis(ax_lam, "y", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    ax_lam.set_ylabel("Dual Multiplier λ\n(log scale)")
    ax_lam.set_xlabel("Training Steps (thousands)")
    ax_lam.set_title("C. Dual Multiplier λ Trajectory (Lagrangian & Residual Dynamics)")
    ax_lam.set_xticks(STEP_TICKS)

    handles = [
        Line2D([], [], color=PALETTE["PPO"], lw=1.8, marker="o", markersize=5, label="PPO"),
        Line2D([], [], color=PALETTE["PPO-Lagrangian"], lw=1.8, marker="^", markersize=5, label="PPO-Lagrangian"),
        Line2D([], [], color=PALETTE["PPO-Residual"], lw=1.8, marker="s", markersize=5, label="PPO-Residual"),
        Line2D([], [], color=PALETTE["Shielded PPO"], lw=1.8, marker="D", markersize=5, label="Shielded PPO"),
        Line2D([], [], color=PALETTE["PPO-Aux-Shield"], lw=1.8, marker="D", markersize=5, label="PPO-Aux-Shield"),
        Line2D([], [], color=PALETTE["PPO-Diff-Shield"], lw=1.8, marker="p", markersize=5, label="PPO-Diff-Shield"),
        Line2D([], [], ls="--", color=GATE, lw=1.5, label="G1 / G2 Gates"),
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label="PSM' Target"),
    ]

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.20, wspace=0.24)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True)
    fig.savefig(os.path.join(dest_dir, "learning_and_constraint_dynamics.png"), bbox_inches="tight")
    plt.close(fig)


def fig_stress_weeks(out_dir: str = None):
    """Visual comparison across 5 stress test weeks with clean formatting."""
    dest_dir = out_dir or SAFETY_DIR
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(TABLES, "test_stress_weeks.csv")
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    crits = ["lowest_pv", "highest_price", "hottest", "stormiest", "highest_exceedance"]
    crit_labels = [
        "Lowest PV\n(W48)",
        "Highest Price\n(W45)",
        "Hottest\n(W30)",
        "Stormiest\n(W51)",
        "★ Highest Exc.\n(W49 Cap-Binding)",
    ]
    methods = ["Grid-Only", "PSM'", "Shield + Idle", "PPO-Residual", "Shielded PPO"]
    colors = [PALETTE["Grid-Only"], PALETTE["PSM'"], PALETTE["Shield + Idle"], PALETTE["PPO-Residual"], PALETTE["Shielded PPO"]]

    def get_group(m):
        if m in ("Grid-Only", "PSM'", "Shield + Idle", "Shield + PSM'"):
            return m
        if "shielded" in m:
            return "Shielded PPO"
        if "residual" in m:
            return "PPO-Residual"
        if "lagrangian" in m:
            return "PPO-Lagrangian"
        return "PPO"

    # Panel A data
    data_cost = {m: {c: [] for c in crits} for m in methods}
    for r in rows:
        fam = r["family"]
        g = get_group(r["model"])
        c = r["stress_criterion"]
        if fam == "baseline" and g in methods:
            if c in data_cost[g]:
                data_cost[g][c].append(float(r["cost_eur"]))
        elif fam == "clamped_mb" and g in methods:
            if c in data_cost[g]:
                data_cost[g][c].append(float(r["cost_eur"]))

    fig, axes = plt.subplots(1, 2, figsize=(18.0, 7.0), gridspec_kw={"width_ratios": [1.1, 0.9], "wspace": 0.25})
    ax1, ax2 = axes[0], axes[1]

    # Panel A: Grouped vertical bars across 5 stress weeks
    x1 = np.arange(len(crits))
    n_m = len(methods)
    bar_w1 = 0.14

    for i, m in enumerate(methods):
        c_means = [float(np.mean(data_cost[m][c])) for c in crits]
        c_errs = [float(np.std(data_cost[m][c], ddof=1)) if len(data_cost[m][c]) > 1 else 0.0 for c in crits]
        pos = x1 - (n_m - 1) * bar_w1 / 2 + i * bar_w1
        ax1.bar(pos, c_means, bar_w1, yerr=c_errs, color=colors[i], edgecolor="white", linewidth=0.8, capsize=2.5)

    # Shaded highlight for W49 in Panel A
    ax1.axvspan(3.5, 4.5, color="#FADBD8", alpha=0.35, zorder=0)
    ax1.text(4.0, 4800, "★ Constraint Binds\n(Net Load > 368 kW)", ha="center", fontsize=8.5,
             fontweight="bold", color="#922B21",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C0392B", alpha=0.9))

    ax1.text(1.5, 4800, "Constraint Never Binds on Weather Weeks\n(0 kW Exceedance for All Methods)",
             ha="center", fontsize=8.5, color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9))

    ax1.set_xticks(x1)
    ax1.set_xticklabels(crit_labels, fontsize=9.5, fontweight="normal")
    ax1.set_ylabel("Total Stress Week Cost (EUR)")
    ax1.set_title("A. Total Cost Across All 5 Stress Test Weeks", fontsize=11.5)
    ax1.set_ylim(0, 5200)

    # Panel B: Cap-Binding Stress Week (W49) Safety Enforcement dynamically computed from test stress results
    w49_by_prefix = TEST_STRESS_DATA.get("w49_by_prefix", {})
    w49_configs = [
        ("Grid-Only", "Grid-Only", "Grid-Only (Uncontrolled)", PALETTE["Grid-Only"]),
        ("clamped_ppo", "PPO (FB)", "PPO [Clamped FB]", "#85929E"),
        ("clamped_mb_ppo", "PPO (MB)", "PPO [Clamped MB]", PALETTE["PPO"]),
        ("clamped_mb_ppo_lagrangian", "PPO-Lag (MB)", "PPO-Lagrangian [Clamped MB]", PALETTE["PPO-Lagrangian"]),
        ("clamped_ppo_residual", "PPO-Res (FB)", "PPO-Residual [Clamped FB]", "#52BE80"),
        ("clamped_mb_ppo_residual", "PPO-Res (MB)", "PPO-Residual [Clamped MB]", PALETTE["PPO-Residual"]),
        ("clamped_ppo_shielded", "Shielded (FB)", "Shielded PPO [Clamped FB]", PALETTE["Shielded PPO"]),
        ("clamped_mb_ppo_shielded", "Shielded (MB)", "Shielded PPO [Clamped MB]", "#884969"),
        ("Shield + Idle", "Shield+Idle", "Shield + Idle (0 Battery)", PALETTE["Shield + Idle"]),
        ("PSM'", "PSM'", "PSM' (Tuned Heuristic)", PALETTE["PSM'"]),
    ]

    w49_models = []
    for prefix_key, m_id, disp, col in w49_configs:
        entries = w49_by_prefix.get(prefix_key, [])
        if entries:
            v_h = statistics.fmean([e["violation_hours"] for e in entries])
            exc = statistics.fmean([e["mean_exceedance_kw"] for e in entries])
        else:
            v_h, exc = 0.0, 0.0
        w49_models.append((m_id, disp, v_h, exc, col))

    y2 = np.arange(len(w49_models))
    bar_h2 = 0.62

    for idx, (m_id, disp, viol_h, exc_kw, col) in enumerate(reversed(w49_models)):
        y_pos = idx
        is_safe = viol_h == 0.0
        bar_col = col
        ax2.barh(y_pos, viol_h, height=bar_h2, color=bar_col, edgecolor="white", linewidth=0.8)

        if is_safe:
            ax2.text(2.0, y_pos, "0h (0.0 kW exc, 100% SAFE)", va="center", fontsize=8.5,
                     fontweight="bold", color="#145A32")
        else:
            ax2.text(viol_h + 1.5, y_pos, f"{viol_h:.0f}h ({exc_kw:.1f} kW exc)", va="center", fontsize=8.5,
                     color="#922B21", fontweight="bold")

    grid_viol_h = next((m[2] for m in w49_models if m[0] == "Grid-Only"), 66.0)
    ax2.axvline(grid_viol_h, ls="--", color=PALETTE["Grid-Only"], lw=1.4, label=f"Grid-Only ({grid_viol_h:.0f}h)")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([m[1] for m in reversed(w49_models)], fontsize=9.0)
    ax2.set_xlim(0, 95)
    ax2.set_xlabel("Violation Hours on Cap-Binding Week W49 (out of 168h)")
    ax2.set_title("B. Cap-Binding Week (2019-W49): Violation Hours", fontsize=11.5)

    legend_handles = [
        Patch(facecolor=PALETTE["Grid-Only"], label="Grid-Only"),
        Patch(facecolor=PALETTE["PSM'"], label="PSM'"),
        Patch(facecolor=PALETTE["Shield + Idle"], label="Shield + Idle"),
        Patch(facecolor=PALETTE["PPO-Residual"], label="PPO-Residual [Clamped MB]"),
        Patch(facecolor=PALETTE["Shielded PPO"], label="Shielded PPO [Clamped MB]"),
    ]

    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.16, wspace=0.25)
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=True)
    fig.savefig(os.path.join(dest_dir, "stress_weeks_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fig_w49_dispatch_trajectory(out_dir: str = None):
    """Clear 168-hour dispatch trajectory showing Net Load, Battery Dispatch, Shedding, and Cap shaving."""
    dest_dir = out_dir or BATTERY_DIR
    os.makedirs(dest_dir, exist_ok=True)
    try:
        from tools.report.plot_stress_trajectory import (
            simulate_policy, GridOnlyPolicy, PeakShaveMinPolicy, SmartGridEnv,
            CapShield, load_config_with_base, load_agent_for_method
        )
    except Exception as exc:
        print(f"Warning: could not import simulation modules for W49 trajectory: {exc}")
        return

    print("Simulating 2019-W49 stress week dispatch trajectories for publication figure...")
    env_base = SmartGridEnv({"split": "test", "charge_clamp": True})
    hist_grid = simulate_policy(env_base, policy=GridOnlyPolicy())
    hist_psm = simulate_policy(env_base, policy=PeakShaveMinPolicy())

    cfg_r = load_config_with_base("clamped_mb_ppo_residual")
    cfg_r["seed"] = 101
    agent_r = load_agent_for_method(cfg_r["method"], cfg_r)
    agent_r.load_models(os.path.join(ROOT, "checkpoints", "ppo_residual", "clamped_mb_ppo_residual_seed101_final.pt"))
    hist_res = simulate_policy(env_base, agent_wrapper=agent_r)

    cfg_s = load_config_with_base("clamped_mb_ppo_shielded")
    cfg_s["seed"] = 101
    agent_s = load_agent_for_method(cfg_s["method"], cfg_s)
    agent_s.load_models(os.path.join(ROOT, "checkpoints", "ppo_shielded", "clamped_mb_ppo_shielded_seed101_final.pt"))
    env_shielded = CapShield(SmartGridEnv({"split": "test", "charge_clamp": True}))
    hist_shi = simulate_policy(env_shielded, agent_wrapper=agent_s)

    hours = np.arange(168)
    cap_kw = float(getattr(env_base, "_contract_cap_kw", CONTRACT_CAP_KW))

    fig, axes = plt.subplots(4, 1, figsize=(16.5, 11.5), sharex=True, gridspec_kw={"height_ratios": [1.15, 0.95, 0.95, 0.95]})
    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

    # Panel 1: Grid Import & Cap Shaving
    ax1.plot(hours, hist_grid["grid_import"], label="Uncontrolled Net Load (Grid-Only)", color=PALETTE["Grid-Only"], ls="--", lw=1.6, alpha=0.85)
    ax1.plot(hours, hist_psm["grid_import"], label="PSM' (Tuned Heuristic)", color=PALETTE["PSM'"], lw=1.8)
    ax1.plot(hours, hist_res["grid_import"], label="PPO-Residual [Clamped MB]", color=PALETTE["PPO-Residual"], lw=1.8)
    ax1.plot(hours, hist_shi["grid_import"], label="Shielded PPO [Clamped MB]", color=PALETTE["Shielded PPO"], lw=2.2)
    ax1.axhline(cap_kw, color=GATE, ls="--", lw=2.0, label=f"Contract Cap ({cap_kw:.2f} kW)")

    # Highlight residual exceedances
    res_exc_mask = hist_res["grid_import"] > cap_kw
    ax1.fill_between(hours, cap_kw, hist_res["grid_import"], where=res_exc_mask, color="#E74C3C", alpha=0.45, label="Residual Cap Exceedance")

    ax1.set_ylabel("Grid Import (kW)")
    ax1.set_title("A. Power Balance & Grid Import vs. Contract Cap (W49 Peak Exceedance Week)", fontsize=11.5)
    ax1.set_ylim(0, 460)
    ax1.legend(loc="upper right", ncol=3, fontsize=8.5, frameon=True)

    # Panel 2: Battery Power Command
    ax2.plot(hours, hist_psm["battery_power"], label="PSM'", color=PALETTE["PSM'"], lw=1.6)
    ax2.plot(hours, hist_res["battery_power"], label="PPO-Residual", color=PALETTE["PPO-Residual"], lw=1.6)
    ax2.plot(hours, hist_shi["battery_power"], label="Shielded PPO", color=PALETTE["Shielded PPO"], lw=1.8)
    ax2.axhline(0.0, color="#888888", ls=":", lw=1.0)
    ax2.set_ylabel("Battery Power (kW)\n[- Disch / + Charge]")
    ax2.set_title("B. Battery Active Power Dispatch (Active Peak Shaving)", fontsize=11.5)
    ax2.set_ylim(-260, 160)
    ax2.legend(loc="upper right", ncol=3, fontsize=8.5, frameon=True)

    # Panel 3: Battery State of Charge (SoC %)
    ax3.plot(hours, hist_psm["soc"] * 100.0, label="PSM'", color=PALETTE["PSM'"], lw=1.6)
    ax3.plot(hours, hist_res["soc"] * 100.0, label="PPO-Residual", color=PALETTE["PPO-Residual"], lw=1.6)
    ax3.plot(hours, hist_shi["soc"] * 100.0, label="Shielded PPO", color=PALETTE["Shielded PPO"], lw=1.8)
    ax3.axhline(90.0, color="#888888", ls=":", lw=1.0)
    ax3.axhline(10.0, color="#888888", ls=":", lw=1.0)
    ax3.fill_between(hours, 10.0, 90.0, color=PALETTE["Feasible"], alpha=0.08, label="Safe Operating Band [10%, 90%]")
    ax3.set_ylabel("Battery SoC (%)")
    ax3.set_title("C. Battery State of Charge (SoC) Dynamics Across Daily Cycles", fontsize=11.5)
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper right", ncol=4, fontsize=8.5, frameon=True)

    # Panel 4: Load Shedding Profiles
    ax4.plot(hours, hist_shi["unserved"], label="Shielded PPO (Defensive Shedding)", color=PALETTE["Shielded PPO"], lw=2.0)
    ax4.plot(hours, hist_psm["unserved"], label="PSM' (Rule Shedding)", color=PALETTE["PSM'"], lw=1.6, ls="--")
    ax4.plot(hours, hist_res["unserved"], label="PPO-Residual (Under-Shedding)", color=PALETTE["PPO-Residual"], lw=1.6, alpha=0.9)
    ax4.set_ylabel("Load Shed (kW)")
    ax4.set_xlabel("Hour of Week (0 = Mon 00:00, 167 = Sun 23:00, Dec 2–8, 2019)")
    ax4.set_title("D. Load Shedding Dispatch (Safety vs. Unserved Penalty Trade-off)", fontsize=11.5)
    ax4.set_ylim(-2, 110)
    ax4.set_xlim(0, 167)
    ax4.legend(loc="upper right", ncol=3, fontsize=8.5, frameon=True)

    # Day of week separators and tick marks
    day_ticks = [i * 24 for i in range(8)]
    day_labels = ["0h\n(Mon)", "24h\n(Tue)", "48h\n(Wed)", "72h\n(Thu)", "96h\n(Fri)", "120h\n(Sat)", "144h\n(Sun)", "168h\n(End)"]
    for ax in (ax1, ax2, ax3, ax4):
        for d in day_ticks[1:-1]:
            ax.axvline(d, color="#E0E0E0", ls="-", lw=0.8, zorder=0)

    ax4.set_xticks(day_ticks)
    ax4.set_xticklabels(day_labels, fontsize=9.0)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.06, hspace=0.26)
    out_file = os.path.join(dest_dir, "w49_dispatch_trajectory.png")
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def fig_ppo_ablation_battery_dynamics(all_families, baselines, out_dir=None):
    """Study Case 1: Spacious, uncluttered battery dynamics for PPO ablations."""
    dest_dir = out_dir or BATTERY_DIR
    os.makedirs(dest_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.22})
    ax_tp, ax_soc = axes[0], axes[1]

    items = [
        ("unclamped_ppo", "PPO [Unclamped]", "Ablation", "#7FB3D5", "Legacy Simulation"),
        ("clamped_ppo", "PPO [Clamped Full-Batch]", "Ablation", "#5499C7", "Full-Batch Inaction"),
        ("clamped_mb_ppo", "PPO [Clamped Minibatch]", "Ablation", PALETTE["PPO"], "★ Best PPO Baseline"),
    ]

    bar_h = 0.55
    y_pos = 0.0
    y_ticks, y_labels = [], []
    row_data = []

    for key, display_name, cat, colour, tag in reversed(items):
        y = y_pos
        y_pos += 1.05
        y_ticks.append(y)
        y_labels.append(display_name)

        purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, m_wear, s_wear, soc, s_soc = _extract_metric_row(
            key, cat, all_families, baselines, None
        )
        row_data.append((y, display_name, colour, tag, tp, s_tp, wear, soc, s_soc))

    max_tp = max([r[4] + (r[5] or 0) for r in row_data])

    for y, _name, col, tag, tp, s_tp, wear, soc, s_soc in row_data:
        is_best = "Best" in tag

        # Panel 1: Throughput Bar
        ax_tp.barh(y, tp, xerr=s_tp if s_tp > 0 else None, height=bar_h, color=col,
                   edgecolor="white", linewidth=0.8, capsize=3.0)
        wear_lbl = f"{tp:.1f} MWh (€{wear:.0f})"
        ax_tp.text(tp + (s_tp or 0) + 1.5, y, wear_lbl, va="center", fontsize=9.5,
                   fontweight="bold" if is_best else "normal", color="#222222")

        # Panel 2: Active SoC Bar
        ax_soc.barh(y, soc, xerr=s_soc if s_soc > 0 else None, height=bar_h, color=col,
                    edgecolor="white", linewidth=0.8, capsize=3.0)
        soc_lbl = f"{soc:.1f}%  ({tag})"
        ax_soc.text(soc + (s_soc or 0) + 1.2, y, soc_lbl, va="center", fontsize=9.5,
                    color="#222222", fontweight="bold" if is_best else "normal")

    # Panel 1 styling
    ax_tp.set_yticks(y_ticks)
    ax_tp.set_yticklabels(y_labels, fontsize=10.5, fontweight="normal")
    ax_tp.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax_tp.set_xlim(0, max_tp * 1.30)
    ax_tp.set_xlabel("Battery Energy Throughput (MWh, 4,368 Val Hours)", fontsize=10.5, fontweight="bold")
    ax_tp.set_title("A. Battery Energy Throughput & Cell Wear", fontsize=11.5, fontweight="bold", pad=28)

    ax_wear = ax_tp.secondary_xaxis('top', functions=(lambda x: x * 20.0, lambda w: w / 20.0))
    ax_wear.set_xlabel("Equivalent Battery Cell Degradation Wear (€, at cw = €20/MWh)", color="#333333", fontsize=9.5, fontweight="bold", labelpad=7)
    ax_wear.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"€{int(round(v)):,}"))

    # Panel 2 styling
    ax_soc.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax_soc.set_xlim(0, 125)
    ax_soc.set_xlabel("Active Unpinned State-of-Charge Time (% in [10%, 90%])", fontsize=10.5, fontweight="bold")
    ax_soc.set_title("B. Active Dynamic Operating Time (% of 4,368h)", fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle("Study Case 1: PPO Environment & Optimizer Ablation — Battery Dynamics & Asset Utilization",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.82, bottom=0.16)
    out_path = os.path.join(dest_dir, "ppo_ablation_battery_dynamics.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_safe_benchmark_battery_dynamics(all_families, baselines, out_dir=None, safety_runs=None):
    """Study Case 2: Cross-Architecture Safe RL Benchmark battery dynamics & SoC operating time."""
    dest_dir = out_dir or BATTERY_DIR
    os.makedirs(dest_dir, exist_ok=True)
    if safety_runs is None:
        safety_runs = load_safety_runs()

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 8.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.24})
    ax_tp, ax_soc = axes[0], axes[1]

    items = [
        # Reference Baselines
        ("Grid-Only", "Grid-Only (Unmanaged)", "Baseline", PALETTE["Grid-Only"], "(Zero Dispatch)"),
        ("PSM'", "PSM' (Tuned Heuristic)", "Baseline", PALETTE["PSM'"], "(Heuristic Target)"),
        ("LP'", "LP' Theoretical Oracle", "Baseline", PALETTE["LP'"], "(Hindsight Bound)"),
        ("Shield + Idle", "Shield + Idle (0-Battery)", "Baseline", PALETTE["Shield + Idle"], "(Zero Dispatch)"),
        ("Shield + PSM'", "Shield + PSM' (Heuristic + Shield)", "Baseline", PALETTE["Shield + PSM'"], "(Heuristic + Shield)"),
        (None, "DIVIDER_1", "Divider", None, ""),
        # Unconstrained Representative
        ("clamped_mb_ppo", "PPO [Clamped Minibatch]", "Unconstrained", PALETTE["PPO"], "(★ Best PPO Baseline)"),
        (None, "DIVIDER_2", "Divider", None, ""),
        # Safe RL Architectures
        ("clamped_mb_ppo_lagrangian", "PPO-Lagrangian [Clamped MB]", "Safe RL", PALETTE["PPO-Lagrangian"], "(Multiplier Instability)"),
        ("clamped_ppo_residual", "PPO-Residual [Clamped Full-Batch]", "Safe RL", PALETTE["PPO-Residual"], "(Base Perturbation)"),
        ("clamped_ppo_shielded", "Shielded PPO [Clamped Full-Batch]", "Safe RL", PALETTE["Shielded PPO"], "(\"Battery Parking\")"),
        ("clamped_mb_ppo_shielded", "Shielded PPO [Clamped Minibatch]", "Safe RL", "#884969", "(\"High Churn\")"),
        ("PPO-Aux-Shield", "PPO-Aux-Shield (Aux. Shield Loss)", "In-The-Loop", PALETTE["PPO-Aux-Shield"], "(Auxiliary Loss)"),
        ("PPO-Diff-Shield", "PPO-Diff-Shield (Diff. Safety Layer)", "In-The-Loop", PALETTE["PPO-Diff-Shield"], "(★ Best Safe RL)"),
    ]

    bar_h = 0.58
    y_pos = 0.0
    y_ticks, y_labels = [], []
    row_data = []
    dividers = []

    for key, display_name, cat, colour, tag in reversed(items):
        if key is None:
            dividers.append(y_pos)
            y_pos += 0.85
            continue
        y = y_pos
        y_pos += 1.05
        y_ticks.append(y)
        y_labels.append(display_name)

        purch, wear, unserv, tot, s_tot, exc, s_exc, tp, s_tp, m_wear, s_wear, soc, s_soc = _extract_metric_row(
            key, cat, all_families, baselines, safety_runs
        )
        row_data.append((y, display_name, cat, colour, tag, tp, s_tp, wear, soc, s_soc))

    max_tp = max([25.0] + [r[5] + (r[6] or 0) for r in row_data])

    for div_y in dividers:
        ax_tp.axhline(div_y, color="#CCCCCC", lw=1.0, ls="--", alpha=0.8, zorder=2)
        ax_soc.axhline(div_y, color="#CCCCCC", lw=1.0, ls="--", alpha=0.8, zorder=2)

    for y, _name, cat, col, tag, tp, s_tp, wear, soc, s_soc in row_data:
        is_highlight = "★" in tag
        # Panel 1: Throughput Bar
        ax_tp.barh(y, tp, xerr=s_tp if s_tp > 0 else None, height=bar_h, color=col,
                   edgecolor="white", linewidth=0.8, capsize=2.5)
        wear_lbl = f"{tp:.1f} MWh (€{wear:.0f})" if tp > 0.05 else "0.0 MWh (€0)"
        ax_tp.text(tp + (s_tp or 0) + 1.5, y, wear_lbl, va="center", fontsize=8.8,
                   fontweight="bold" if is_highlight else "normal", color="#222222")

        # Panel 2: Active SoC Bar
        ax_soc.barh(y, soc, xerr=s_soc if s_soc > 0 else None, height=bar_h, color=col,
                    edgecolor="white", linewidth=0.8, capsize=2.5)
        soc_lbl = f"{soc:.1f}%" + (f"  {tag}" if tag else "")
        ax_soc.text(soc + (s_soc or 0) + 1.5, y, soc_lbl, va="center", fontsize=8.8,
                    color="#222222", fontweight="bold" if is_highlight else "normal")

    # Dynamic baseline reference points
    psm_tp = float(baselines.get("PSM'", {}).get("battery_throughput_mwh", 12.348))
    lp_tp = float(baselines.get("LP'", {}).get("battery_throughput_mwh", 24.990))
    psm_soc = float(baselines.get("PSM'", {}).get("active_soc_pct", 0.0))
    lp_soc = float(baselines.get("LP'", {}).get("active_soc_pct", 0.0))

    # Panel 1 styling
    ax_tp.set_yticks(y_ticks)
    ax_tp.set_yticklabels(y_labels, fontsize=9.5, fontweight="normal")
    ax_tp.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax_tp.set_xlim(0, max_tp * 1.30)
    ax_tp.axvline(psm_tp, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    ax_tp.axvline(lp_tp, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)
    ax_tp.set_xlabel("Battery Energy Throughput (MWh, 26 Val Weeks / 4,368h)", fontsize=10.5, fontweight="bold")
    ax_tp.set_title("A. Battery Energy Throughput & Cell Wear", fontsize=11.5, pad=28)

    ax_wear = ax_tp.secondary_xaxis('top', functions=(lambda x: x * WEAR_COST_PER_MWH, lambda w: w / WEAR_COST_PER_MWH))
    ax_wear.set_xlabel(f"Equivalent Battery Cell Degradation Wear (€, at cw = €{WEAR_COST_PER_MWH/1000:.2f}/kWh)", color="#333333", fontsize=9.5, fontweight="bold", labelpad=8)
    ax_wear.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"€{int(round(v)):,}"))

    # Panel 2 styling
    ax_soc.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)
    ax_soc.set_xlim(0, 135)
    ax_soc.axvline(psm_soc, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2)
    ax_soc.axvline(lp_soc, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2)
    ax_soc.set_xlabel("Active Unpinned State-of-Charge Time (% of 4,368 Val Hours in [10%, 90%])", fontsize=10.5, fontweight="bold")
    ax_soc.set_title("B. Active Dynamic Operating Time (% Active Cycles)", fontsize=11.5, pad=10)

    method_handles = [
        Patch(facecolor=PALETTE["Grid-Only"], label="Grid-Only"),
        Patch(facecolor=PALETTE["PSM'"], label="PSM' Target"),
        Patch(facecolor=PALETTE["LP'"], label="LP' Oracle"),
        Patch(facecolor=PALETTE["PPO"], label="Best PPO"),
        Patch(facecolor=PALETTE["PPO-Lagrangian"], label="PPO-Lagrangian"),
        Patch(facecolor=PALETTE["PPO-Residual"], label="PPO-Residual"),
        Patch(facecolor=PALETTE["Shielded PPO"], label="Shielded PPO [FB]"),
        Patch(facecolor="#884969", label="Shielded PPO [MB]"),
        Patch(facecolor=PALETTE["PPO-Aux-Shield"], label="PPO-Aux-Shield"),
        Patch(facecolor=PALETTE["PPO-Diff-Shield"], label="PPO-Diff-Shield (Best Safe RL)"),
    ]
    ref_handles = [
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label=f"PSM' Target ({psm_tp:.2f} MWh / {psm_soc:.1f}%)"),
        Line2D([], [], ls="-.", color=PALETTE["LP'"], lw=1.3, label=f"LP' Oracle ({lp_tp:.2f} MWh / {lp_soc:.1f}%)"),
    ]

    fig.subplots_adjust(left=0.25, right=0.97, top=0.88, bottom=0.15)
    fig.legend(handles=method_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=True, fontsize=8.5)
    out_path = os.path.join(dest_dir, "safe_benchmark_battery_dynamics.png")
    fig.savefig(out_path, bbox_inches="tight")
    # Also save alias to battery_usage_comparison.png for backward compatibility
    alias_path = os.path.join(dest_dir, "battery_usage_comparison.png")
    shutil.copyfile(out_path, alias_path)
    plt.close(fig)


def fig_battery_usage(all_families, baselines, out_dir=None, safety_runs=None):
    """Backward-compatible wrapper generating both Study Case 1 and Study Case 2 battery figures."""
    fig_ppo_ablation_battery_dynamics(all_families, baselines, out_dir)
    fig_safe_benchmark_battery_dynamics(all_families, baselines, out_dir, safety_runs)


def fig_targeted_learning_curves(all_families, baselines, out_dir=None):
    """Learning curves for competitive methods (Residual and Shielded) across environments."""
    dest_dir = out_dir or TRAINING_DIR
    os.makedirs(dest_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax_cost, ax_exc = axes[0], axes[1]

    comp_methods = [
        ("PPO-Residual", PALETTE["PPO-Residual"]),
        ("Shielded PPO", PALETTE["Shielded PPO"]),
    ]
    fam_styles = [
        ("unclamped", "Unclamped", ":", "x", 1.6),
        ("clamped", "Clamped", "--", "o", 1.8),
        ("clamped_mb", "Clamped Minibatch", "-.", "^", 2.0),
    ]

    max_cost_upper = G2_COST_EUR
    for m_label, colour in comp_methods:
        for f_key, _f_name, ls, marker, lw in fam_styles:
            runs = all_families.get(f_key, {}).get(m_label, [])
            if not runs:
                continue
            steps, costs, excs = eval_curve(runs)
            if not steps:
                continue
            mean_c, _ = mean_sd(costs)
            mean_e, _ = mean_sd(excs)
            x = [s / 1000 for s in steps]

            ax_cost.plot(x, mean_c, color=colour, ls=ls, lw=lw, marker=marker,
                         markersize=5.5 if marker != "*" else 8.0, zorder=4)
            max_cost_upper = max(max_cost_upper, max(mean_c))

            y_exc = [max(v, FLOOR) for v in mean_e]
            ax_exc.plot(x, y_exc, color=colour, ls=ls, lw=lw, marker=marker,
                        markersize=5.5 if marker != "*" else 8.0, zorder=4)

    ax_cost.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2, label=f"G2 Cost Gate (€{G2_COST_EUR/1000:.1f}k)")
    if "PSM'" in baselines:
        c_psm = float(baselines["PSM'"]["cost_eur"])
        ax_cost.axhline(c_psm, ls=":", color=PALETTE["PSM'"], lw=1.3, zorder=2,
                        label=f"PSM' Target (€{c_psm/1000:.1f}k)")
    if "LP'" in baselines:
        c_lp = float(baselines["LP'"]["cost_eur"])
        ax_cost.axhline(c_lp, ls="-.", color=PALETTE["LP'"], lw=1.3, zorder=2,
                        label=f"LP' Oracle (€{c_lp/1000:.1f}k)")

    ymax = int(math.ceil((max_cost_upper + 2500) / 2500.0) * 2500)
    ax_cost.set_ylim(44000, ymax)
    ax_cost.yaxis.set_major_locator(FixedLocator(list(range(45000, ymax + 1, 5000))))
    ax_cost.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_ylabel("Validation Total Cost (EUR)")
    ax_cost.set_title("A. Validation Cost Progression (Targeted Competitive Methods)")

    ax_exc.axhline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2, label="G1 Gate (0.05 kW/step)")
    ax_exc.set_yscale("log")
    ax_exc.set_ylim(FLOOR, 10)
    log_axis(ax_exc, "y", EXC_TICKS)
    ax_exc.set_ylabel("Mean Exceedance\n(kW/step, log scale)")
    ax_exc.set_xlabel("Training Steps (thousands)")
    ax_exc.set_title("B. Mean Grid Cap Exceedance Progression (Targeted Competitive Methods)")
    ax_exc.set_xticks(STEP_TICKS)

    method_handles = [Line2D([], [], color=c, lw=2.5, label=m) for m, c in comp_methods]
    env_handles = [
        Line2D([], [], color="#444444", ls=ls, lw=lw, marker=marker,
               markersize=6, label=fn) for _, fn, ls, marker, lw in fam_styles
    ]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.5, label="G1 / G2 Gates"),
        Line2D([], [], ls=":", color=PALETTE["PSM'"], lw=1.3, label="PSM' Target"),
        Line2D([], [], ls="-.", color=PALETTE["LP'"], lw=1.3, label="LP' Oracle"),
    ]

    fig.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.18, hspace=0.20)
    fig.legend(handles=method_handles + env_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=True)
    fig.savefig(os.path.join(dest_dir, "targeted_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 3. PIPELINE ORCHESTRATION & MAIN CLI
# ===========================================================================

def plot_family(family: str, baselines: dict, out_dir: str = None):
    runs = load_runs(family=family)
    total = sum(len(v) for v in runs.values())
    if total == 0:
        print(f"no run JSONs found in {RUNS} for family '{family}' - skipping")
        return

    print(f"Generating thematic per-family figures ({total} runs in {len(runs)} groups) for family '{family}'")
    for fn, args in (
        (fig_learning_curves, (runs, baselines, family, out_dir)),
        (fig_cost_vs_exceedance, (runs, baselines, family, out_dir)),
        (fig_lambda, (runs, family, out_dir)),
        (fig_per_seed, (runs, baselines, family, out_dir)),
        (fig_cost_breakdown, (runs, baselines, family, out_dir)),
        (fig_family_summary, (runs, baselines, family, out_dir)),
    ):
        try:
            fn(*args)
            print(f"  ok  {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")


def plot_comparison(baselines: dict, out_dir: str = None):
    for d in THEMATIC_DIRS:
        os.makedirs(d, exist_ok=True)

    all_families = {
        "unclamped": load_runs("unclamped"),
        "clamped": load_runs("clamped"),
        "clamped_mb": load_runs("clamped_mb"),
    }
    safety_runs = load_safety_runs()

    print("Generating thematic cross-family comparison figures...")
    for fn, args in (
        (fig_unified_pareto, (all_families, baselines, out_dir, safety_runs)),
        (fig_ppo_ablation_cost_breakdown, (all_families, baselines, out_dir)),
        (fig_safe_benchmark_cost_breakdown, (all_families, baselines, out_dir, safety_runs)),
        (fig_safety_architectures_comparison, (safety_runs, baselines, out_dir)),
        (fig_safety_learning_curves, (safety_runs, baselines, out_dir)),
        (fig_learning_and_constraint_dynamics, (all_families, baselines, out_dir, safety_runs)),
        (fig_ppo_ablation_battery_dynamics, (all_families, baselines, out_dir)),
        (fig_safe_benchmark_battery_dynamics, (all_families, baselines, out_dir, safety_runs)),
        (fig_targeted_learning_curves, (all_families, baselines, out_dir)),
        (fig_stress_weeks, (out_dir,)),
        (fig_w49_dispatch_trajectory, (out_dir,)),
    ):
        try:
            fn(*args)
            print(f"  ok  {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")


def main():
    family = "all"
    if "--family" in sys.argv:
        idx = sys.argv.index("--family")
        if idx + 1 < len(sys.argv):
            family = sys.argv[idx + 1]

    baselines = load_baselines()

    for d in THEMATIC_DIRS:
        os.makedirs(d, exist_ok=True)

    if family in ("comparison", "compare"):
        plot_comparison(baselines)
    elif family in ("safety", "shield", "shields"):
        safety_runs = load_safety_runs()
        fig_safety_architectures_comparison(safety_runs, baselines)
        fig_safety_learning_curves(safety_runs, baselines)
        print("safety architecture figures written to safety_and_exceedance/ and training_dynamics/")
    elif family == "all":
        for fam in ("unclamped", "clamped", "clamped_mb"):
            plot_family(fam, baselines)
        plot_comparison(baselines)
    else:
        plot_family(family, baselines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
