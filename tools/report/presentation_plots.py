"""16:9 widescreen presentation figure suite for Safe Microgrid RL Benchmark.

Generates twelve clean, clutter-free slide figures (PNG at 250 DPI and optional SVG)
in ``results/figures/presentation/`` by loading every metric and physical parameter
dynamically from ``tools/report/load_results.py``, ``env/smartgrid_env.py``,
``configs/config.yaml``, ``data/processed/``, and ``tools/report/eval_regimes_val.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib import patheffects as pe
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
from agents.mpc import MPCPolicy
from agents.safety_layer import DifferentiableSafetyLayer
from env.cap_shield import CapShield, SHED_ORDER
from env.smartgrid_env import (
    ACTION_DIM,
    BASE_OBS_DIM,
    FORECAST_FEATURE_DIM,
    SmartGridEnv,
    ZONE_NAMES,
    default_config,
)
from experiments.train_core import (
    agent_name,
    build_env,
    is_shielded_run,
    load_agent_for_method,
    load_config_with_base,
)
from tools.report.eval_regimes_val import _resolve_val_stress_week_idx
from tools.report.load_results import (
    load_baselines,
    load_per_seed_table,
    load_project_constants,
    load_regime_benchmarks,
    load_test_evaluation_summary,
    load_test_stress_weeks_data,
    load_training_curves,
)
from tools.report.style import (
    METHOD_LABELS,
    PALETTE,
    SLIDE_FIGSIZE,
    add_slide_header,
    apply_presentation_rc,
    check_axes_text_overlaps,
    repel_y_positions,
    save_slide_fig,
    style_clean_axes,
)

PRES_DIR = os.path.join(REPO_ROOT, "results", "figures", "presentation")

METHOD_COLOR_MAP: Dict[str, str] = {
    "Grid-Only": PALETTE["Grid-Only"],
    "PSM'": PALETTE["PSM'"],
    "LP'": PALETTE["LP'"],
    "MILP'": PALETTE["MILP'"],
    "Shield + Idle": PALETTE["Shield + Idle"],
    "Shield + PSM'": PALETTE["Shield + PSM'"],
    "MPCPolicy(perfect, H=24)": PALETTE["MPC-Perfect"],
    "MPCPolicy(causal_noisy, H=24)": PALETTE["MPC"],
    "MPCPolicy(persistence, H=24)": PALETTE["MPC-Persistence"],
    "clamped_ppo": PALETTE["PPO_FB"],
    "clamped_mb_ppo": PALETTE["PPO"],
    "clamped_mb_ppo_lagrangian": PALETTE["PPO-Lagrangian"],
    "clamped_ppo_residual": PALETTE["PPO-Residual"],
    "clamped_ppo_shielded": PALETTE["Shielded PPO"],
    "clamped_mb_ppo_shielded": PALETTE["Shielded PPO_MB"],
    "clamped_ppo_shield_aux": PALETTE["PPO-Aux-Shield"],
    "clamped_ppo_diff_shield": PALETTE["PPO-Diff-Shield"],
    "causal23_ppo_diff_shield": PALETTE["Causal-Diff-Shield"],
    "ci_ppo_shielded": PALETTE["Shielded PPO"],
    "ci_ppo_diff_shield": PALETTE["PPO-Diff-Shield"],
    "ci_ppo_diff_6d": PALETTE["Diff-Shield-6D"],
}

OURS_KEYS = {"clamped_ppo_diff_shield", "causal23_ppo_diff_shield", "ci_ppo_diff_shield"}


def _label(key: str) -> str:
    """Resolve concise display label from ``METHOD_LABELS``."""
    return METHOD_LABELS.get(key, key)


def _color(key: str) -> str:
    """Resolve consistent Okabe-Ito color for a method or run prefix."""
    return METHOD_COLOR_MAP.get(key, PALETTE.get(key, "#334155"))


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Compute sample mean and std (ddof=1, 0.0 for n < 2)."""
    if not values:
        return 0.0, 0.0
    m = statistics.fmean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return float(m), float(s)


def _dest_dir(out_dir: Optional[str]) -> str:
    """Resolve target directory defaulting to ``PRES_DIR``."""
    return out_dir or PRES_DIR


def _new_slide() -> Figure:
    """Initialize a fresh 16:9 slide figure with presentation rcParams."""
    apply_presentation_rc()
    return plt.figure(figsize=SLIDE_FIGSIZE)


def _val_total_hours(regime_rows: Sequence[Dict[str, Any]], baselines: Dict[str, Any]) -> int:
    """Derive Val split total hours dynamically from benchmark rows or baseline Grid-Only rate."""
    del baselines
    for row in regime_rows:
        if row.get("suite") == "regime1_val" and row.get("n_weeks"):
            n_weeks = int(row["n_weeks"])
            if (
                row.get("method") == "Grid-Only"
                and row.get("violation_hours")
                and row.get("violation_rate_pct")
            ):
                ep_len = round(
                    (float(row["violation_hours"]) / (float(row["violation_rate_pct"]) / 100.0))
                    / n_weeks
                )
                return n_weeks * int(ep_len)
    for row in regime_rows:
        if row.get("suite") == "regime1_val" and row.get("n_weeks"):
            return int(row["n_weeks"]) * _episode_hours_from_split_report()
    return 26 * _episode_hours_from_split_report()


def _episode_hours_from_split_report() -> int:
    """Compute hours per week dynamically from data/processed/split_report.json."""
    sr_path = os.path.join(REPO_ROOT, "data", "processed", "split_report.json")
    if os.path.exists(sr_path):
        with open(sr_path, encoding="utf-8") as f:
            val_meta = json.load(f).get("splits", {}).get("val", {})
        if val_meta.get("n_hours") and val_meta.get("n_weeks"):
            return int(round(float(val_meta["n_hours"]) / float(val_meta["n_weeks"])))
    if hasattr(SmartGridEnv, "DEFAULTS"):
        return int(SmartGridEnv.DEFAULTS["episode_length_hours"])
    return 168


def _find_regime_row(
    rows: Sequence[Dict[str, Any]],
    suite: str,
    method: str,
    row_type: str = "single",
    checkpoint_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Find a single matching row in parsed ``regime_benchmarks_val.csv`` rows."""
    for row in rows:
        if (
            row.get("suite") != suite
            or row.get("method") != method
            or row.get("row_type") != row_type
        ):
            continue
        if checkpoint_type is not None and row.get("checkpoint_type") != checkpoint_type:
            continue
        return row
    raise KeyError(
        f"Missing regime row: suite={suite!r}, method={method!r}, "
        f"row_type={row_type!r}, ckpt={checkpoint_type!r}"
    )


def _has_regime_row(
    rows: Sequence[Dict[str, Any]],
    suite: str,
    method: str,
    row_type: str = "single",
    checkpoint_type: Optional[str] = None,
) -> bool:
    """Return True if a matching row exists in parsed ``regime_benchmarks_val.csv`` rows."""
    try:
        _find_regime_row(rows, suite, method, row_type, checkpoint_type)
        return True
    except KeyError:
        return False


def _find_regime_seed_rows(
    rows: Sequence[Dict[str, Any]],
    suite: str,
    method: str,
    checkpoint_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all per_seed rows matching suite, method, and optional checkpoint_type."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if (
            row.get("suite") != suite
            or row.get("method") != method
            or row.get("row_type") != "per_seed"
        ):
            continue
        if checkpoint_type is not None and row.get("checkpoint_type") != checkpoint_type:
            continue
        out.append(row)
    return out


def _per_seed_summary(
    per_seed: Dict[str, List[Dict[str, float]]],
    prefix: str,
    total_hours: int,
) -> Dict[str, float]:
    """Summarize mean and std metrics for an RL prefix from ``per_seed_metrics.csv``."""
    rows = per_seed[prefix]
    c_m, c_s = _mean_std([r["cost_eur"] for r in rows])
    e_m, _ = _mean_std([r["energy_purchase_eur"] for r in rows])
    w_m, _ = _mean_std([r["battery_wear_eur"] for r in rows])
    u_m, _ = _mean_std([r["unserved_penalty_eur"] for r in rows])
    exc_m, exc_s = _mean_std([r["exceedance_kw"] for r in rows])
    vh_m, vh_s = _mean_std([r["violation_hours"] for r in rows])
    vr_vals = [(r["violation_hours"] / float(total_hours)) * 100.0 for r in rows]
    vr_m, vr_s = _mean_std(vr_vals)
    soc_m, soc_s = _mean_std([r["active_soc_pct"] for r in rows])
    return {
        "cost_mean": c_m,
        "cost_std": c_s,
        "energy_mean": e_m,
        "wear_mean": w_m,
        "unserved_mean": u_m,
        "exc_mean": exc_m,
        "exc_std": exc_s,
        "vh_mean": vh_m,
        "vh_std": vh_s,
        "viol_rate_mean": vr_m,
        "viol_rate_std": vr_s,
        "active_soc_mean": soc_m,
        "active_soc_std": soc_s,
    }


def _highlight_row_band_and_tick(
    ax: Axes,
    row_idx: int,
    label_text: str,
    color: str,
) -> None:
    """Highlight hero row subtly and style its y-tick label in bold hero color."""
    ax.axhspan(row_idx - 0.42, row_idx + 0.42, color=color, alpha=0.08, zorder=0)
    ticks = ax.get_yticklabels()
    if 0 <= row_idx < len(ticks):
        ticks[row_idx].set_fontweight("bold")
        ticks[row_idx].set_color(color)
        ticks[row_idx].set_text(label_text)


def _draw_horizontal_stacked_bars(
    ax: Axes,
    y_positions: Sequence[float],
    components_by_row: Sequence[Sequence[Tuple[str, float, str]]],
    totals: Sequence[float],
    total_stds: Sequence[float],
    height: float = 0.56,
) -> None:
    """Draw horizontal stacked cost bars with optional total-cost ±1 SD error bars."""
    for y, parts, tot, std in zip(y_positions, components_by_row, totals, total_stds):
        left = 0.0
        for _comp_name, val, col in parts:
            if val > 0.0:
                ax.barh(
                    y,
                    val,
                    left=left,
                    height=height,
                    color=col,
                    edgecolor="#FFFFFF",
                    linewidth=0.7,
                    zorder=3,
                )
                left += val
        if std > 0.001:
            ax.errorbar(
                tot,
                y,
                xerr=std,
                fmt="none",
                ecolor="#0F172A",
                elinewidth=1.4,
                capsize=3.5,
                capthick=1.3,
                zorder=5,
            )


def _halo() -> List[Any]:
    """Subtle white text stroke so labels remain crisp over reference lines without pill boxes."""
    return [pe.withStroke(linewidth=3.2, foreground="#FFFFFF")]


def _fmt_signed_eur(val: float) -> str:
    """Format signed EUR delta with leading currency sign (e.g. ``-€1,397``)."""
    rounded = round(float(val))
    if rounded < 0:
        return f"-€{abs(rounded):,}"
    if rounded > 0:
        return f"+€{rounded:,}"
    return "€0"


def _assert_no_overlaps(fig: Figure, texts: Sequence[Text], tag: str) -> None:
    """Verify zero bounding-box overlaps across annotated ``Text`` artists."""
    overlaps = check_axes_text_overlaps(fig, texts)
    if overlaps:
        pairs = ", ".join(f"({a.get_text()!r}, {b.get_text()!r})" for a, b in overlaps)
        raise RuntimeError(f"[{tag}] Detected overlapping annotation Text objects: {pairs}")


def fig_01_ablation_pathologies(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot Regime 1 ablation pathologies: leakage, churn, and battery parking."""
    fig = _new_slide()
    baselines = load_baselines()
    per_seed = load_per_seed_table()
    constants = load_project_constants()
    regime_rows = load_regime_benchmarks()
    total_hours = _val_total_hours(regime_rows, baselines)

    keys = [
        "Grid-Only",
        "clamped_ppo",
        "clamped_mb_ppo",
        "clamped_mb_ppo_lagrangian",
        "clamped_ppo_shielded",
        "clamped_ppo_diff_shield",
    ]
    records: List[Dict[str, Any]] = []
    for k in keys:
        if k == "Grid-Only":
            b = baselines[k]
            vr = (float(b["violation_hours"]) / float(total_hours)) * 100.0
            records.append({
                "key": k,
                "label": _label(k),
                "color": _color(k),
                "energy": float(b["energy_purchase_eur"]),
                "wear": float(b["battery_wear_eur"]),
                "unserved": float(b["unserved_penalty_eur"]),
                "cost_mean": float(b["cost_eur"]),
                "cost_std": 0.0,
                "viol_rate_mean": vr,
                "viol_rate_std": 0.0,
                "active_soc_mean": float(b["active_soc_pct"]),
                "active_soc_std": 0.0,
            })
        else:
            s = _per_seed_summary(per_seed, k, total_hours)
            records.append({
                "key": k,
                "label": _label(k),
                "color": _color(k),
                "energy": s["energy_mean"],
                "wear": s["wear_mean"],
                "unserved": s["unserved_mean"],
                "cost_mean": s["cost_mean"],
                "cost_std": s["cost_std"],
                "viol_rate_mean": s["viol_rate_mean"],
                "viol_rate_std": s["viol_rate_std"],
                "active_soc_mean": s["active_soc_mean"],
                "active_soc_std": s["active_soc_std"],
            })

    psm_cost = float(baselines["PSM'"]["cost_eur"])
    g2_cost = float(constants["g2_cost_eur"])

    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1.15], wspace=0.14)
    ax_cost = fig.add_subplot(outer[0])
    inner_r = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.14)
    ax_viol = fig.add_subplot(inner_r[0], sharey=ax_cost)
    ax_soc = fig.add_subplot(inner_r[1], sharey=ax_cost)

    ys = np.arange(len(records), dtype=float)
    y_labels = [r["label"] for r in records]
    ours_idx = keys.index("clamped_ppo_diff_shield")

    # Panel A: Cost decomposition
    style_clean_axes(ax_cost, grid_axis="x")
    stacks = [
        [
            ("Energy", r["energy"], PALETTE["Energy"]),
            ("Wear", r["wear"], PALETTE["Wear"]),
            ("Unserved", r["unserved"], PALETTE["Unserved"]),
        ]
        for r in records
    ]
    _draw_horizontal_stacked_bars(
        ax_cost,
        ys,
        stacks,
        [r["cost_mean"] for r in records],
        [r["cost_std"] for r in records],
    )
    ax_cost.axvline(psm_cost, color=PALETTE["PSM'"], ls=":", lw=1.7, zorder=2)
    ax_cost.axvline(g2_cost, color=PALETTE["Gate"], ls="--", lw=1.7, zorder=2)

    max_cost_right = max(r["cost_mean"] + r["cost_std"] for r in records)
    ax_cost.set_xlim(0.0, max_cost_right * 1.24)
    ax_cost.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.0f}k"))
    ax_cost.set_xlabel("26-Week Validation Cost (€k, mean ± SD)")
    ax_cost.set_yticks(ys)
    ax_cost.set_yticklabels(y_labels, fontsize=10.5)
    ax_cost.invert_yaxis()

    cost_texts: List[Text] = []
    for y, r in zip(ys, records):
        x_txt = r["cost_mean"] + r["cost_std"] + max_cost_right * 0.015
        txt = (
            f"€{r['cost_mean'] / 1000.0:.1f}k"
            if r["cost_std"] < 1.0
            else f"€{r['cost_mean'] / 1000.0:.1f}k ± {r['cost_std'] / 1000.0:.1f}k"
        )
        is_ours = r["key"] in OURS_KEYS
        t = ax_cost.text(
            x_txt,
            y,
            txt,
            ha="left",
            va="center",
            fontsize=9.2,
            fontweight="bold" if is_ours else "normal",
            color=PALETTE["PPO-Diff-Shield"] if is_ours else "#1E293B",
            path_effects=_halo(),
            zorder=6,
        )
        cost_texts.append(t)

    ax_cost.set_title("A. Total Cost Decomposition & G2 Gate", loc="left", pad=28, fontsize=12.0)
    cost_legend = [
        Patch(facecolor=PALETTE["Energy"], label="Energy"),
        Patch(facecolor=PALETTE["Wear"], label="Wear"),
        Patch(facecolor=PALETTE["Unserved"], label="Unserved"),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.7, label=f"PSM' (€{psm_cost / 1000.0:.1f}k)"),
        Line2D([0], [0], color=PALETTE["Gate"], ls="--", lw=1.7, label=f"G2 (€{g2_cost / 1000.0:.1f}k)"),
    ]
    ax_cost.legend(
        handles=cost_legend,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.008),
        ncol=5,
        fontsize=8.3,
        columnspacing=0.9,
        handlelength=1.4,
        frameon=False,
    )

    # Panel B1: Cap Violations
    style_clean_axes(ax_viol, grid_axis="x")
    ax_viol.tick_params(labelleft=False)
    max_viol = max(r["viol_rate_mean"] + r["viol_rate_std"] for r in records)
    ax_viol.set_xlim(-0.8, max(14.5, max_viol * 1.42))
    ax_viol.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_viol.set_xlabel("Violation Rate (%)")
    ax_viol.set_title("B1. Cap Violations", loc="left", pad=12, fontsize=11.5)

    viol_texts: List[Text] = []
    for y, r in zip(ys, records):
        v = r["viol_rate_mean"]
        vs = r["viol_rate_std"]
        is_ours = r["key"] in OURS_KEYS
        col = (
            PALETTE["PPO-Diff-Shield"]
            if is_ours
            else (PALETTE["Gate"] if v > 0.05 else "#475569")
        )
        ax_viol.hlines(y, 0.0, v, color=col, lw=2.3, alpha=0.85, zorder=3)
        if vs > 0.05:
            ax_viol.errorbar(
                v, y, xerr=vs, fmt="none", ecolor=col, elinewidth=1.2, capsize=2.8, zorder=4
            )
        ax_viol.scatter(
            [v], [y], color=col, s=68 if is_ours else 52, zorder=5, edgecolors="#FFFFFF", lw=0.9
        )
        note = f"{v:.2f}%"
        if r["key"] == "clamped_mb_ppo_lagrangian":
            note = f"{v:.2f}% (churn)"
        elif is_ours:
            note = f"{v:.2f}% (safe)"
        t = ax_viol.text(
            v + vs + 0.45,
            y,
            note,
            ha="left",
            va="center",
            fontsize=8.9,
            fontweight="bold" if (is_ours or v > 0.05) else "normal",
            color=col,
        )
        viol_texts.append(t)

    # Panel B2: Battery Activity
    style_clean_axes(ax_soc, grid_axis="x")
    ax_soc.tick_params(labelleft=False)
    max_soc = max(r["active_soc_mean"] + r["active_soc_std"] for r in records)
    ax_soc.set_xlim(-2.0, max(72.0, max_soc * 1.35))
    ax_soc.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_soc.set_xlabel("Active Unpinned SoC (%)")
    ax_soc.set_title("B2. Battery Activity", loc="left", pad=12, fontsize=11.5)

    soc_texts: List[Text] = []
    for y, r in zip(ys, records):
        sm = r["active_soc_mean"]
        ss = r["active_soc_std"]
        is_ours = r["key"] in OURS_KEYS
        is_parked = r["key"] == "clamped_ppo_shielded"
        col = (
            PALETTE["PPO-Diff-Shield"]
            if is_ours
            else (PALETTE["Shielded PPO"] if is_parked else r["color"])
        )
        ax_soc.hlines(y, 0.0, sm, color=col, lw=2.3, alpha=0.85, zorder=3)
        if ss > 0.1:
            ax_soc.errorbar(
                sm, y, xerr=ss, fmt="none", ecolor=col, elinewidth=1.2, capsize=2.8, zorder=4
            )
        ax_soc.scatter(
            [sm], [y], color=col, s=68 if is_ours else 52, zorder=5, edgecolors="#FFFFFF", lw=0.9
        )
        if is_parked:
            lbl = f"{sm:.1f}% (parked)"
        elif r["key"] == "Grid-Only":
            lbl = f"{sm:.1f}% (idle)"
        else:
            lbl = f"{sm:.1f}%"
        t = ax_soc.text(
            sm + ss + 1.5,
            y,
            lbl,
            ha="left",
            va="center",
            fontsize=8.9,
            fontweight="bold" if (is_ours or is_parked) else "normal",
            color=col,
        )
        soc_texts.append(t)

    for ax in (ax_cost, ax_viol, ax_soc):
        _highlight_row_band_and_tick(ax, ours_idx, records[ours_idx]["label"], PALETTE["PPO-Diff-Shield"])

    _ = next(r for r in records if r["key"] == "clamped_ppo_shielded")
    _ = records[ours_idx]
    _ = next(r for r in records if r["key"] == "clamped_mb_ppo_lagrangian")
    _ = len(per_seed["clamped_ppo_diff_shield"])
    _ = round(total_hours / max(1, _episode_hours_from_split_report()))

    add_slide_header(
        fig,
        "Differentiable Safety Layer Eliminates Leakage, Churn, and Battery Parking",
        "",
    )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.785, bottom=0.13)
    _assert_no_overlaps(fig, cost_texts + viol_texts + soc_texts, "fig_01")
    return save_slide_fig(fig, _dest_dir(out_dir), "01_ablation_pathologies", save_svg=save_svg)


def fig_02_pareto_cost_safety(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot cost vs. exceedance Pareto frontier alongside inference latency."""
    fig = _new_slide()
    baselines = load_baselines()
    per_seed = load_per_seed_table()
    constants = load_project_constants()
    regime_rows = load_regime_benchmarks()
    total_hours = _val_total_hours(regime_rows, baselines)

    g1_kw = float(constants["g1_exceedance_kw"])
    g2_eur = float(constants["g2_cost_eur"])

    mpc_row = _find_regime_row(regime_rows, "regime1_val", "MPCPolicy(causal_noisy, H=24)", "single")
    ds_best_mean = _find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "mean", "best")
    ds_best_std = _find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "std", "best")
    ds_fin_mean = _find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "mean", "final")
    ds_fin_std = _find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "std", "final")
    sh_mb = _per_seed_summary(per_seed, "clamped_mb_ppo_shielded", total_hours)
    aux_mb = _per_seed_summary(per_seed, "clamped_ppo_shield_aux", total_hours)

    col1_pts: List[Dict[str, Any]] = [
        {"key": "LP'", "label": "LP' Oracle", "cost": float(baselines["LP'"]["cost_eur"]), "std": 0.0, "color": _color("LP'"), "marker": "*", "size": 190},
        {"key": "MPCPolicy(causal_noisy, H=24)", "label": _label("MPCPolicy(causal_noisy, H=24)"), "cost": float(mpc_row["cost_eur"]), "std": 0.0, "color": _color("MPCPolicy(causal_noisy, H=24)"), "marker": "s", "size": 75},
        {"key": "PSM'", "label": _label("PSM'"), "cost": float(baselines["PSM'"]["cost_eur"]), "std": 0.0, "color": _color("PSM'"), "marker": "^", "size": 85},
        {"key": "clamped_ppo_diff_shield", "label": "PPO-Diff-Shield [Best]", "cost": float(ds_best_mean["cost_eur"]), "std": float(ds_best_std["cost_eur"]), "color": PALETTE["PPO-Diff-Shield"], "marker": "D", "size": 95, "ours": True},
        {"key": "clamped_ppo_diff_shield", "label": "PPO-Diff-Shield [Final]", "cost": float(ds_fin_mean["cost_eur"]), "std": float(ds_fin_std["cost_eur"]), "color": PALETTE["PPO-Diff-Shield"], "marker": "P", "size": 120, "ours": True},
        {"key": "clamped_mb_ppo_shielded", "label": _label("clamped_mb_ppo_shielded"), "cost": sh_mb["cost_mean"], "std": sh_mb["cost_std"], "color": _color("clamped_mb_ppo_shielded"), "marker": "p", "size": 80},
        {"key": "clamped_ppo_shield_aux", "label": _label("clamped_ppo_shield_aux"), "cost": aux_mb["cost_mean"], "std": aux_mb["cost_std"], "color": _color("clamped_ppo_shield_aux"), "marker": "h", "size": 80},
    ]

    col2_specs = [
        ("clamped_ppo_residual", "PPO-Residual", "D", 80),
        ("clamped_mb_ppo_lagrangian", "PPO-Lagrangian", "s", 80),
        ("clamped_mb_ppo", "PPO (Minibatch)", "o", 80),
        ("Grid-Only", "Grid-Only", "X", 95),
    ]
    col2_pts: List[Dict[str, Any]] = []
    for k, lbl, mk, sz in col2_specs:
        if k == "Grid-Only":
            b = baselines[k]
            col2_pts.append({
                "key": k, "label": lbl, "cost": float(b["cost_eur"]), "std": 0.0,
                "exc": float(b["exceedance_kw"]), "exc_std": 0.0,
                "color": _color(k), "marker": mk, "size": sz,
            })
        else:
            s = _per_seed_summary(per_seed, k, total_hours)
            col2_pts.append({
                "key": k, "label": lbl, "cost": s["cost_mean"], "std": s["cost_std"],
                "exc": s["exc_mean"], "exc_std": s["exc_std"],
                "color": _color(k), "marker": mk, "size": sz,
            })

    grid_r1 = _find_regime_row(regime_rows, "regime1_val", "Grid-Only", "single")
    psm_r1 = _find_regime_row(regime_rows, "regime1_val", "PSM'", "single")
    col3_pts: List[Dict[str, Any]] = [
        {"label": "Grid-Only", "cost": float(grid_r1["cost_eur"]), "std": 0.0, "lat": float(grid_r1["latency_p50_ms"]), "lat_std": 0.0, "color": _color("Grid-Only"), "marker": "X", "size": 85},
        {"label": "MPC (Causal)", "cost": float(mpc_row["cost_eur"]), "std": 0.0, "lat": float(mpc_row["latency_p50_ms"]), "lat_std": 0.0, "color": _color("MPCPolicy(causal_noisy, H=24)"), "marker": "s", "size": 75},
        {"label": "PSM'", "cost": float(psm_r1["cost_eur"]), "std": 0.0, "lat": float(psm_r1["latency_p50_ms"]), "lat_std": 0.0, "color": _color("PSM'"), "marker": "^", "size": 80},
        {"label": "Diff-Shield [Best]", "cost": float(ds_best_mean["cost_eur"]), "std": float(ds_best_std["cost_eur"]), "lat": float(ds_best_mean["latency_p50_ms"]), "lat_std": float(ds_best_std["latency_p50_ms"]), "color": PALETTE["PPO-Diff-Shield"], "marker": "D", "size": 85, "ours": True},
        {"label": "Diff-Shield [Final]", "cost": float(ds_fin_mean["cost_eur"]), "std": float(ds_fin_std["cost_eur"]), "lat": float(ds_fin_mean["latency_p50_ms"]), "lat_std": float(ds_fin_std["latency_p50_ms"]), "color": PALETTE["PPO-Diff-Shield"], "marker": "P", "size": 110, "ours": True},
    ]

    all_costs = [p["cost"] - p["std"] for p in col1_pts + col2_pts] + [
        p["cost"] + p["std"] for p in col1_pts + col2_pts
    ]
    y_min = min(all_costs) - 2600.0
    y_max = max(all_costs) + 2800.0

    outer = GridSpec(1, 2, figure=fig, width_ratios=[3.15, 0.95], wspace=0.14)
    inner_left = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.15, 2.0], wspace=0.05)
    ax_zero = fig.add_subplot(inner_left[0])
    ax_exc = fig.add_subplot(inner_left[1], sharey=ax_zero)
    ax_lat = fig.add_subplot(outer[1], sharey=ax_zero)

    annotation_texts: List[Text] = []

    # Panel 1: Exact Zero Safe Lane
    style_clean_axes(ax_zero, grid_axis="y")
    ax_zero.spines["right"].set_visible(False)
    ax_zero.set_xlim(0.0, 1.0)
    ax_zero.set_ylim(y_min, y_max)
    ax_zero.axhspan(y_min, g2_eur, facecolor=PALETTE["Feasible"], alpha=0.10, zorder=0)
    ax_zero.axhline(g2_eur, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=2)
    ax_zero.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y / 1000.0:.0f}k"))
    ax_zero.set_ylabel("26-Week Validation Cost (€k)")
    ax_zero.set_xticks([0.16])
    ax_zero.set_xticklabels(["0.00 kW\n(Exact Zero)"])
    ax_zero.set_title("Exact Zero Safe Lane", loc="left", pad=12, fontsize=11.5)

    x_offsets = np.linspace(0.10, 0.22, len(col1_pts))
    col1_sorted = sorted(enumerate(col1_pts), key=lambda item: item[1]["cost"])
    repelled_y1 = repel_y_positions(
        [item[1]["cost"] for item in col1_sorted],
        min_sep=max(2600.0, (y_max - y_min) / 10.8),
        y_min=y_min + 1400.0,
        y_max=y_max - 1400.0,
    )
    label_y_by_idx1 = {orig_i: ry for (orig_i, _), ry in zip(col1_sorted, repelled_y1)}

    for idx, p in enumerate(col1_pts):
        mx = float(x_offsets[idx])
        my = p["cost"]
        if p["std"] > 1.0:
            ax_zero.errorbar(
                mx, my, yerr=p["std"], fmt="none", ecolor=p["color"],
                elinewidth=1.4, capsize=3.2, alpha=0.85, zorder=4
            )
        ax_zero.scatter(
            [mx], [my], marker=p["marker"], color=p["color"], s=p["size"],
            edgecolors="#FFFFFF", lw=1.0, zorder=5
        )
        ly = label_y_by_idx1[idx]
        ax_zero.plot([mx + 0.015, 0.295], [my, ly], color=p["color"], lw=0.95, alpha=0.75, zorder=3)
        is_ours = bool(p.get("ours"))
        t = ax_zero.text(
            0.31,
            ly,
            f"{p['label']} (€{my / 1000.0:.1f}k)",
            ha="left",
            va="center",
            fontsize=8.6,
            fontweight="bold" if is_ours else "normal",
            color=p["color"],
            path_effects=_halo(),
            zorder=6,
        )
        annotation_texts.append(t)

    t_sz = ax_zero.text(
        0.05,
        y_min + 1200.0,
        f"SAFE ZONE\n≤G1 ({g1_kw:.2f} kW) & ≤G2 (€{g2_eur / 1000.0:.1f}k)",
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color="#15803D",
    )
    annotation_texts.append(t_sz)

    # Panel 2: Unsafe / Leaky Region
    style_clean_axes(ax_exc, grid_axis="both")
    ax_exc.spines["left"].set_visible(False)
    ax_exc.tick_params(labelleft=False, left=False)
    ax_exc.set_xscale("log")
    exc_min, exc_max = 0.025, 6.8
    ax_exc.set_xlim(exc_min, exc_max)
    ax_exc.fill_between([exc_min, g1_kw], y_min, g2_eur, facecolor=PALETTE["Feasible"], alpha=0.10, zorder=0)
    ax_exc.axvline(g1_kw, color=PALETTE["Gate"], ls="-.", lw=1.6, zorder=2)
    ax_exc.axhline(g2_eur, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=2)
    ax_exc.set_xlabel("Mean Grid Exceedance (kW/step, log scale)")
    ax_exc.set_title("Unsafe / Leaky Region", loc="left", pad=12, fontsize=11.5)

    d = 0.015
    ax_zero.plot((1 - d, 1 + d), (-d, +d), transform=ax_zero.transAxes, color="#64748B", lw=1.2, clip_on=False)
    ax_exc.plot((-d, +d), (-d, +d), transform=ax_exc.transAxes, color="#64748B", lw=1.2, clip_on=False)

    t_g1 = ax_exc.text(
        g1_kw * 1.12,
        y_max - 1400.0,
        f"G1 Gate ≤ {g1_kw:.2f} kW",
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["Gate"],
    )
    t_g2 = ax_exc.text(
        exc_max * 0.92,
        g2_eur + 550.0,
        f"G2 Cost Gate ≤ €{g2_eur / 1000.0:.1f}k",
        ha="right",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["Gate"],
    )
    annotation_texts.extend([t_g1, t_g2])

    col2_sorted = sorted(enumerate(col2_pts), key=lambda item: item[1]["cost"])
    repelled_y2 = repel_y_positions(
        [item[1]["cost"] for item in col2_sorted],
        min_sep=max(2900.0, (y_max - y_min) / 9.0),
        y_min=y_min + 1400.0,
        y_max=y_max - 3200.0,
    )
    label_y_by_idx2 = {orig_i: ry for (orig_i, _), ry in zip(col2_sorted, repelled_y2)}

    for idx, p in enumerate(col2_pts):
        ex = p["exc"]
        cy = p["cost"]
        if p["std"] > 1.0 or p["exc_std"] > 0.01:
            ax_exc.errorbar(
                ex,
                cy,
                xerr=p["exc_std"] if p["exc_std"] > 0.01 else None,
                yerr=p["std"] if p["std"] > 1.0 else None,
                fmt="none",
                ecolor=p["color"],
                elinewidth=1.4,
                capsize=3.2,
                alpha=0.85,
                zorder=4,
            )
        ax_exc.scatter(
            [ex], [cy], marker=p["marker"], color=p["color"], s=p["size"],
            edgecolors="#FFFFFF", lw=1.0, zorder=5
        )
        ly = label_y_by_idx2[idx]
        if ex > 2.0:
            tx = ex * 0.56
            ha = "right"
            lx_end = tx * 1.05
        else:
            tx = ex * 0.72 if p["key"] == "clamped_ppo_residual" else ex * 1.30
            ha = "right" if p["key"] == "clamped_ppo_residual" else "left"
            lx_end = tx * 1.04 if ha == "right" else tx * 0.96
            if p["key"] == "clamped_ppo_residual":
                ly = cy + p["std"] + 1500.0
            elif p["key"] == "clamped_mb_ppo_lagrangian":
                ly = cy + p["std"] + 1100.0
        ax_exc.plot([ex, lx_end], [cy, ly], color=p["color"], lw=0.9, alpha=0.7, zorder=3)
        t = ax_exc.text(
            tx,
            ly,
            f"{p['label']}\n€{cy / 1000.0:.1f}k | {ex:.2f} kW",
            ha=ha,
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=p["color"],
        )
        annotation_texts.append(t)

    # Panel 3: Inference Speed
    style_clean_axes(ax_lat, grid_axis="both")
    ax_lat.tick_params(labelleft=False)
    ax_lat.set_xscale("log")
    ax_lat.set_xlim(0.0008, 65.0)
    ax_lat.axhline(g2_eur, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=2)
    ax_lat.axhspan(y_min, g2_eur, facecolor=PALETTE["Feasible"], alpha=0.07, zorder=0)
    ax_lat.set_xlabel("P50 Step Latency (ms, log)")
    ax_lat.set_title("Inference Speed", loc="left", pad=12, fontsize=11.5)

    col3_sorted = sorted(enumerate(col3_pts), key=lambda item: item[1]["cost"])
    repelled_y3 = repel_y_positions(
        [item[1]["cost"] for item in col3_sorted],
        min_sep=max(2900.0, (y_max - y_min) / 9.2),
        y_min=y_min + 1600.0,
        y_max=y_max - 1600.0,
    )
    label_y_by_idx3 = {orig_i: ry for (orig_i, _), ry in zip(col3_sorted, repelled_y3)}

    for idx, p in enumerate(col3_pts):
        lx = p["lat"]
        cy = p["cost"]
        if p["std"] > 1.0:
            ax_lat.errorbar(
                lx, cy, yerr=p["std"], fmt="none", ecolor=p["color"],
                elinewidth=1.2, capsize=2.8, alpha=0.8, zorder=4
            )
        ax_lat.scatter(
            [lx], [cy], marker=p["marker"], color=p["color"], s=p["size"],
            edgecolors="#FFFFFF", lw=1.0, zorder=5
        )
        ly = label_y_by_idx3[idx]
        if lx >= 1.5:
            tx = lx * 0.72
            ha = "right"
        else:
            tx = lx * 1.45
            ha = "left"
        ax_lat.plot([lx, tx], [cy, ly], color=p["color"], lw=0.85, alpha=0.7, zorder=3)
        t = ax_lat.text(
            tx,
            ly,
            f"{p['label']}: {lx:.2f} ms" if lx >= 0.01 else f"{p['label']}: {lx:.4f} ms",
            ha=ha,
            va="center",
            fontsize=8.2,
            fontweight="bold" if p.get("ours") else "normal",
            color=p["color"],
        )
        annotation_texts.append(t)

    diff_lat = float(ds_fin_mean["latency_p50_ms"])
    mpc_lat = float(mpc_row["latency_p50_ms"])
    speedup = mpc_lat / max(diff_lat, 1e-9)
    t_spd = ax_lat.text(
        0.5,
        0.96,
        f"Diff-Shield {speedup:.1f}× faster than MPC",
        transform=ax_lat.transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["PPO-Diff-Shield"],
    )
    annotation_texts.append(t_spd)

    add_slide_header(
        fig,
        "PPO-Diff-Shield Combines Exact-Zero Violations, G2-Passing Cost, and Sub-ms Speed",
        "26-week Validation (2019 H1) · error bars: ±1 SD across 5 seeds",
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.81, bottom=0.13)
    _assert_no_overlaps(fig, annotation_texts, "fig_02")
    return save_slide_fig(fig, _dest_dir(out_dir), "02_pareto_cost_safety", save_svg=save_svg)


def fig_03_learning_dynamics(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot 50k–300k validation learning dynamics across 5 seeds."""
    fig = _new_slide()
    baselines = load_baselines()
    constants = load_project_constants()
    prefixes = [
        "clamped_mb_ppo",
        "clamped_mb_ppo_lagrangian",
        "clamped_ppo_shielded",
        "clamped_mb_ppo_shielded",
        "clamped_ppo_diff_shield",
    ]
    curves = load_training_curves(prefixes)

    psm_cost = float(baselines["PSM'"]["cost_eur"])
    g2_cost = float(constants["g2_cost_eur"])

    axes = fig.subplots(1, 3, sharex=True)
    ax_cost, ax_viol, ax_soc = axes[0], axes[1], axes[2]

    panel_specs = [
        (ax_cost, "total_cost_eur", 0.001, "A. Validation Cost (€k)", "Val Cost (€k)", lambda y, _: f"€{y:.0f}k"),
        (ax_viol, "violation_rate_pct", 1.0, "B. Cap Violation Rate (%)", "Violation Rate (%)", lambda y, _: f"{y:.0f}%"),
        (ax_soc, "active_soc_pct", 1.0, "C. Active Unpinned SoC (%)", "Active SoC (%)", lambda y, _: f"{y:.0f}%"),
    ]
    for ax, metric, scale, ptitle, ylabel, fmt in panel_specs:
        style_clean_axes(ax, grid_axis="both")
        ax.set_title(ptitle, loc="left", pad=10, fontsize=11.8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training Steps")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000.0:.0f}k"))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))

        for idx, pfx in enumerate(prefixes):
            pts = curves.get(pfx, [])
            if not pts:
                continue
            steps = np.array([pt["step"] for pt in pts], dtype=float)
            mean_v = np.array([pt[f"{metric}_mean"] for pt in pts], dtype=float) * scale
            std_v = np.array([pt[f"{metric}_std"] for pt in pts], dtype=float) * scale
            is_ours = pfx in OURS_KEYS
            col = _color(pfx)
            lw = 2.9 if is_ours else 1.8
            ls = "-" if is_ours else ("--" if "lagrangian" in pfx else ("-." if "shielded" in pfx else ":"))
            mk = "o" if is_ours else ("s" if "lagrangian" in pfx else "^")
            z = 6 if is_ours else 3 + idx * 0.2

            ax.fill_between(
                steps, mean_v - std_v, mean_v + std_v, color=col,
                alpha=0.17 if is_ours else 0.10, zorder=2
            )
            ax.plot(
                steps, mean_v, color=col, lw=lw, ls=ls, marker=mk,
                markersize=5.2 if is_ours else 4.2, label=_label(pfx), zorder=z
            )

    ax_cost.axhline(psm_cost / 1000.0, color=PALETTE["PSM'"], ls=":", lw=1.6, zorder=2)
    ax_cost.axhline(g2_cost / 1000.0, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=2)
    step_min = min(pt["step"] for pts in curves.values() for pt in pts)
    step_max = max(pt["step"] for pts in curves.values() for pt in pts)
    t_g2 = ax_cost.text(
        step_min + 4000,
        g2_cost / 1000.0 + 0.5,
        f"G2 (€{g2_cost / 1000.0:.1f}k)",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["Gate"],
        va="bottom",
        path_effects=_halo(),
        zorder=7,
    )
    t_psm = ax_cost.text(
        step_max - 4000,
        psm_cost / 1000.0 - 0.6,
        f"PSM' (€{psm_cost / 1000.0:.1f}k)",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["PSM'"],
        va="top",
        ha="right",
        path_effects=_halo(),
        zorder=7,
    )

    ax_viol.set_ylim(bottom=-0.5)
    ax_soc.set_ylim(bottom=-2.0)

    handles, labels = ax_cost.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.865),
        ncol=5,
        fontsize=9.3,
        frameon=False,
        columnspacing=1.2,
    )

    ds_end = curves["clamped_ppo_diff_shield"][-1]
    _ = curves["clamped_ppo_shielded"][-1]
    _ = curves["clamped_mb_ppo_lagrangian"][-1]
    add_slide_header(
        fig,
        "Diff-Shield Converges Steadily Below G2 While Naive Shielded PPO Parks and Lagrangian Diverges",
        f"Shaded bands: mean ± 1 SD across {ds_end['n_seeds']} deterministic seeds ({step_min // 1000}k–{step_max // 1000}k steps)",
    )
    fig.subplots_adjust(left=0.065, right=0.98, top=0.74, bottom=0.14, wspace=0.22)
    _assert_no_overlaps(fig, [t_g2, t_psm], "fig_03")
    return save_slide_fig(fig, _dest_dir(out_dir), "03_learning_dynamics", save_svg=save_svg)


def fig_04_regime1_causal_mpc(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot Regime 1 validation comparison vs. offline LP'/MILP' oracles and MPC."""
    fig = _new_slide()
    regime_rows = load_regime_benchmarks()
    baselines = load_baselines()
    constants = load_project_constants()

    specs = [
        ("LP'", None, "LP' Oracle"),
        ("MILP'", None, "MILP' Oracle"),
        ("MPCPolicy(perfect, H=24)", None, _label("MPCPolicy(perfect, H=24)")),
        ("MPCPolicy(causal_noisy, H=24)", None, _label("MPCPolicy(causal_noisy, H=24)")),
        ("MPCPolicy(persistence, H=24)", None, _label("MPCPolicy(persistence, H=24)")),
        ("PSM'", None, _label("PSM'")),
        ("clamped_ppo_diff_shield", "best", f"{_label('clamped_ppo_diff_shield')} [Best]"),
        ("clamped_ppo_diff_shield", "final", f"{_label('clamped_ppo_diff_shield')} [Final]"),
        ("causal23_ppo_diff_shield", "best", f"{_label('causal23_ppo_diff_shield')} [Best]"),
        ("causal23_ppo_diff_shield", "final", f"{_label('causal23_ppo_diff_shield')} [Final]"),
    ]
    entries: List[Dict[str, Any]] = []
    for method, ckpt, disp in specs:
        if ckpt is None:
            r = _find_regime_row(regime_rows, "regime1_val", method, "single")
            entries.append({
                "method": method,
                "label": disp,
                "color": _color(method),
                "cost_mean": float(r["cost_eur"]),
                "cost_std": 0.0,
                "seed_costs": [],
                "cvar_mean": r["cvar95_cost_eur"],
                "cvar_std": 0.0,
                "lat_mean": r["latency_p50_ms"],
                "lat_std": 0.0,
                "ours": method in OURS_KEYS,
            })
        else:
            m_row = _find_regime_row(regime_rows, "regime1_val", method, "mean", ckpt)
            s_row = _find_regime_row(regime_rows, "regime1_val", method, "std", ckpt)
            seeds = _find_regime_seed_rows(regime_rows, "regime1_val", method, ckpt)
            entries.append({
                "method": method,
                "label": disp,
                "color": _color(method),
                "cost_mean": float(m_row["cost_eur"]),
                "cost_std": float(s_row["cost_eur"]),
                "seed_costs": [float(sr["cost_eur"]) for sr in seeds],
                "cvar_mean": float(m_row["cvar95_cost_eur"]),
                "cvar_std": float(s_row["cvar95_cost_eur"]),
                "lat_mean": float(m_row["latency_p50_ms"]),
                "lat_std": float(s_row["latency_p50_ms"]),
                "ours": method in OURS_KEYS,
            })

    psm_cost = float(baselines["PSM'"]["cost_eur"])
    g2_cost = float(constants["g2_cost_eur"])

    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.15], wspace=0.14)
    ax_cost = fig.add_subplot(outer[0])
    inner_r = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.14)
    ax_cvar = fig.add_subplot(inner_r[0], sharey=ax_cost)
    ax_lat = fig.add_subplot(inner_r[1], sharey=ax_cost)

    ys = np.arange(len(entries), dtype=float)
    jitter = np.linspace(-0.18, 0.18, 5)

    # Panel A: Total Cost
    style_clean_axes(ax_cost, grid_axis="both")
    ax_cost.axvline(psm_cost, color=PALETTE["PSM'"], ls=":", lw=1.6, zorder=2)
    ax_cost.axvline(g2_cost, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=2)

    all_c_vals = (
        [e["cost_mean"] - e["cost_std"] for e in entries]
        + [e["cost_mean"] + e["cost_std"] for e in entries]
        + [c for e in entries for c in e["seed_costs"]]
    )
    c_lo = min(all_c_vals) - 1000.0
    c_hi = max(all_c_vals) + 3800.0
    ax_cost.set_xlim(c_lo, c_hi)
    ax_cost.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.0f}k"))
    ax_cost.set_xlabel("26-Week Validation Cost (€k, mean ± SD & seeds)")
    ax_cost.set_yticks(ys)
    ax_cost.set_yticklabels([e["label"] for e in entries], fontsize=10.0)
    ax_cost.invert_yaxis()
    ax_cost.set_ylim(len(entries) - 0.35, -0.88)
    ax_cost.set_title("A. Total Cost vs. Oracles & MPC", loc="left", pad=16, fontsize=11.8)

    texts_all: List[Text] = []
    for idx, (y, e) in enumerate(zip(ys, entries)):
        col = e["color"]
        if e["ours"]:
            for ax in (ax_cost, ax_cvar, ax_lat):
                _highlight_row_band_and_tick(ax, idx, e["label"], col)
        if e["seed_costs"]:
            for j_off, sc in zip(jitter, sorted(e["seed_costs"])):
                ax_cost.scatter([sc], [y + j_off], color=col, alpha=0.45, s=26, edgecolors="none", zorder=3)
            ax_cost.errorbar(
                e["cost_mean"], y, xerr=e["cost_std"], fmt="none", ecolor=col,
                elinewidth=2.0, capsize=4.0, zorder=4
            )
        ax_cost.scatter(
            [e["cost_mean"]], [y], color=col, s=82 if e["ours"] else 68,
            edgecolors="#FFFFFF", lw=1.0, zorder=5
        )
        lbl = (
            f"€{e['cost_mean'] / 1000.0:.2f}k ± {e['cost_std'] / 1000.0:.2f}k"
            if e["seed_costs"]
            else f"€{e['cost_mean'] / 1000.0:.2f}k"
        )
        if e["method"] == "MPCPolicy(persistence, H=24)":
            x_txt = e["cost_mean"] - 190.0
            ha = "right"
        else:
            x_end = max([e["cost_mean"] + e["cost_std"]] + e["seed_costs"])
            x_txt = x_end + 170.0
            ha = "left"
        t = ax_cost.text(
            x_txt,
            y,
            lbl,
            ha=ha,
            va="center",
            fontsize=8.6,
            fontweight="bold" if e["ours"] else "normal",
            color=col if e["ours"] else "#1E293B",
            path_effects=_halo(),
            zorder=6,
        )
        texts_all.append(t)

    t_psm = ax_cost.text(psm_cost + 80.0, -0.74, "PSM'", fontsize=8.4, color=PALETTE["PSM'"], fontweight="bold", va="top", ha="left")
    t_g2 = ax_cost.text(g2_cost + 80.0, -0.74, "G2", fontsize=8.4, color=PALETTE["Gate"], fontweight="bold", va="top", ha="left")
    texts_all.extend([t_psm, t_g2])

    # Panel B1: Tail Risk CVaR95
    style_clean_axes(ax_cvar, grid_axis="both")
    ax_cvar.tick_params(labelleft=False)
    cvar_vals = [float(e["cvar_mean"]) for e in entries if e["cvar_mean"] is not None]
    ax_cvar.set_xlim(min(cvar_vals) - 4.0, max(cvar_vals) + 6.5)
    ax_cvar.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x:.0f}"))
    ax_cvar.set_xlabel("CVaR₉₅ Hourly Cost (€)")
    ax_cvar.set_title("B1. Tail Risk (CVaR₉₅)", loc="left", pad=16, fontsize=11.5)

    for y, e in zip(ys, entries):
        if e["cvar_mean"] is None:
            t = ax_cvar.text(
                min(cvar_vals) - 2.0, y, "N/A (offline)", ha="left", va="center",
                fontsize=8.3, color="#64748B", style="italic"
            )
            texts_all.append(t)
            continue
        cm = float(e["cvar_mean"])
        cs = float(e["cvar_std"])
        col = e["color"]
        if cs > 0.05:
            ax_cvar.errorbar(cm, y, xerr=cs, fmt="none", ecolor=col, elinewidth=1.6, capsize=3.0, zorder=4)
        ax_cvar.scatter([cm], [y], color=col, s=72 if e["ours"] else 58, edgecolors="#FFFFFF", lw=0.9, zorder=5)
        t = ax_cvar.text(
            cm + cs + 0.6,
            y,
            f"€{cm:.1f}",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold" if e["ours"] else "normal",
            color=col if e["ours"] else "#1E293B",
        )
        texts_all.append(t)

    # Panel B2: Step Latency P50
    style_clean_axes(ax_lat, grid_axis="both")
    ax_lat.tick_params(labelleft=False)
    lat_vals = [float(e["lat_mean"]) for e in entries if e["lat_mean"] is not None]
    ax_lat.set_xlim(-0.25, max(lat_vals) + 1.35)
    ax_lat.set_xlabel("Latency P50 (ms)")
    ax_lat.set_title("B2. Step Latency P50", loc="left", pad=16, fontsize=11.5)

    for y, e in zip(ys, entries):
        if e["lat_mean"] is None:
            t = ax_lat.text(
                0.05, y, "Offline", ha="left", va="center",
                fontsize=8.3, color="#64748B", style="italic"
            )
            texts_all.append(t)
            continue
        lm = float(e["lat_mean"])
        ls_val = float(e["lat_std"])
        col = e["color"]
        ax_lat.hlines(y, 0.0, lm, color=col, lw=2.0, alpha=0.8, zorder=3)
        if ls_val > 0.005:
            ax_lat.errorbar(lm, y, xerr=ls_val, fmt="none", ecolor=col, elinewidth=1.4, capsize=2.8, zorder=4)
        ax_lat.scatter([lm], [y], color=col, s=72 if e["ours"] else 58, edgecolors="#FFFFFF", lw=0.9, zorder=5)
        t = ax_lat.text(
            lm + ls_val + 0.14,
            y,
            f"{lm:.2f} ms",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold" if e["ours"] else "normal",
            color=col if e["ours"] else "#1E293B",
        )
        texts_all.append(t)

    ds_fin = next(e for e in entries if e["method"] == "clamped_ppo_diff_shield" and "Final" in e["label"])
    mpc_c = next(e for e in entries if e["method"] == "MPCPolicy(causal_noisy, H=24)")
    mpc_p = next(e for e in entries if e["method"] == "MPCPolicy(persistence, H=24)")
    lat_ratio = mpc_c["lat_mean"] / ds_fin["lat_mean"]
    r1_weeks = int(_find_regime_row(regime_rows, "regime1_val", "Grid-Only", "single")["n_weeks"])
    r1_seeds = len(_find_regime_seed_rows(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "final"))

    add_slide_header(
        fig,
        f"Diff-Shield Matches Causal Control Efficiency at {lat_ratio:.1f}× Lower Latency Than LP-MPC",
        (
            f"Regime 1 Val (all safe rows 0.00% viol): PPO-Diff-Shield (€{ds_fin['cost_mean'] / 1000.0:.2f}k, "
            f"{ds_fin['lat_mean']:.2f} ms) beats persistence MPC (€{mpc_p['cost_mean'] / 1000.0:.2f}k) and runs "
            f"{lat_ratio:.1f}× faster than causal MPC (€{mpc_c['cost_mean'] / 1000.0:.2f}k, {mpc_c['lat_mean']:.2f} ms)."
        ),
        f"Source: results/tables/regime_benchmarks_val.csv (suite=regime1_val, {r1_weeks} weeks, {r1_seeds} seeds for RL).",
    )
    fig.subplots_adjust(left=0.22, right=0.975, top=0.82, bottom=0.13)
    _assert_no_overlaps(fig, texts_all, "fig_04")
    return save_slide_fig(fig, _dest_dir(out_dir), "04_regime1_causal_mpc", save_svg=save_svg)


def fig_05_regime2_ci_ablation(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot Regime 2 C&I cost breakdown, demand charge savings, and 6-D motor leakage."""
    fig = _new_slide()
    regime_rows = load_regime_benchmarks()

    # Required specs + optional Shield baselines added in T3/T4
    specs: List[Tuple[str, str, str, bool]] = [
        ("LP'", "single", "LP' Oracle", False),
        ("MPCPolicy(causal_noisy, H=24)", "single", _label("MPCPolicy(causal_noisy, H=24)"), False),
        ("MPCPolicy(perfect, H=24)", "single", _label("MPCPolicy(perfect, H=24)"), False),
        ("ci_ppo_diff_shield", "mean", _label("ci_ppo_diff_shield"), False),
        ("PSM'", "single", _label("PSM'"), False),
        ("Shield + PSM'", "single", _label("Shield + PSM'"), True),
        ("ci_ppo_shielded", "mean", _label("ci_ppo_shielded"), False),
        ("ci_ppo_diff_6d", "mean", _label("ci_ppo_diff_6d"), False),
        ("Shield + Idle", "single", _label("Shield + Idle"), True),
        ("Grid-Only", "single", _label("Grid-Only"), False),
    ]
    entries: List[Dict[str, Any]] = []
    for method, rtype, disp, is_optional in specs:
        ckpt = "final" if rtype == "mean" else None
        if is_optional and not _has_regime_row(regime_rows, "regime2_val", method, rtype, ckpt):
            continue
        row = _find_regime_row(regime_rows, "regime2_val", method, rtype, ckpt)
        std_val = 0.0
        if rtype == "mean":
            std_row = _find_regime_row(regime_rows, "regime2_val", method, "std", "final")
            std_val = float(std_row["cost_eur"])
        entries.append({
            "method": method,
            "label": disp,
            "color": _color(method),
            "cost": float(row["cost_eur"]),
            "cost_std": std_val,
            "energy": float(row["energy_purchase_eur"]),
            "wear": float(row["battery_wear_eur"]),
            "unserved": float(row["unserved_penalty_eur"]),
            "dc": float(row["demand_charge_eur"]),
            "trip": float(row["industrial_trip_eur"]),
            "viol_rate": float(row["violation_rate_pct"]),
            "viol_hrs": float(row["violation_hours"]),
            "ours": method in OURS_KEYS,
        })

    ax_full, ax_non = fig.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [1.15, 1.15], "wspace": 0.10}
    )
    ys = np.arange(len(entries), dtype=float)

    # Panel A: Full C&I Cost Breakdown
    style_clean_axes(ax_full, grid_axis="x")
    full_stacks = [
        [
            ("Energy", e["energy"], PALETTE["Energy"]),
            ("Demand Charge", e["dc"], PALETTE["DemandCharge"]),
            ("Unserved VoLL", e["unserved"], PALETTE["Unserved"]),
            ("Wear", e["wear"], PALETTE["Wear"]),
            ("Trip", e["trip"], PALETTE["Trip"]),
        ]
        for e in entries
    ]
    _draw_horizontal_stacked_bars(
        ax_full,
        ys,
        full_stacks,
        [e["cost"] for e in entries],
        [e["cost_std"] for e in entries],
    )
    max_tot = max(e["cost"] + e["cost_std"] for e in entries)
    ax_full.set_xlim(0.0, max_tot * 1.46)
    ax_full.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.0f}k"))
    ax_full.set_xlabel("Total C&I Cost (€k, mean ± SD)")
    ax_full.set_yticks(ys)
    ax_full.set_yticklabels([e["label"] for e in entries], fontsize=10.2)
    ax_full.invert_yaxis()
    ax_full.set_title("A. Total C&I Cost Breakdown", loc="left", pad=22, fontsize=11.8)

    comp_legend = [
        Patch(facecolor=PALETTE["Energy"], label="Energy"),
        Patch(facecolor=PALETTE["DemandCharge"], label="Demand Charge"),
        Patch(facecolor=PALETTE["Unserved"], label="Unserved VoLL"),
        Patch(facecolor=PALETTE["Wear"], label="Wear"),
        Patch(facecolor=PALETTE["Trip"], label="Trip"),
    ]
    ax_full.legend(
        handles=comp_legend,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=5,
        fontsize=8.1,
        columnspacing=0.8,
        handlelength=1.2,
        frameon=False,
    )

    texts_all: List[Text] = []
    for idx, (y, e) in enumerate(zip(ys, entries)):
        if e["ours"]:
            _highlight_row_band_and_tick(ax_full, idx, e["label"], PALETTE["PPO-Diff-Shield"])
        unsafe = e["viol_rate"] > 0.0
        tag = " [UNSAFE]" if unsafe else ""
        x_txt = e["cost"] + e["cost_std"] + max_tot * 0.015
        txt = (
            f"€{e['cost'] / 1000.0:.1f}k{tag}"
            if e["cost_std"] < 1.0
            else f"€{e['cost'] / 1000.0:.1f}k ± {e['cost_std'] / 1000.0:.1f}k{tag}"
        )
        col = PALETTE["Gate"] if unsafe else (PALETTE["PPO-Diff-Shield"] if e["ours"] else "#1E293B")
        t = ax_full.text(
            x_txt,
            y,
            txt,
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="bold" if (e["ours"] or unsafe) else "normal",
            color=col,
        )
        texts_all.append(t)

    # Panel B: Non-Energy Cost (DC + Unserved + Wear + Trip)
    style_clean_axes(ax_non, grid_axis="x")
    ax_non.tick_params(labelleft=False)
    non_stacks = [
        [
            ("Demand Charge", e["dc"], PALETTE["DemandCharge"]),
            ("Unserved + Wear", e["unserved"] + e["wear"] + e["trip"], PALETTE["Unserved"]),
        ]
        for e in entries
    ]
    non_totals = [e["dc"] + e["unserved"] + e["wear"] + e["trip"] for e in entries]
    _draw_horizontal_stacked_bars(ax_non, ys, non_stacks, non_totals, [0.0] * len(entries))

    psm_e = next(e for e in entries if e["method"] == "PSM'")
    ds_e = next(e for e in entries if e["method"] == "ci_ppo_diff_shield")
    d6_e = next(e for e in entries if e["method"] == "ci_ppo_diff_6d")
    dc_diff = ds_e["dc"] - psm_e["dc"]

    ax_non.axvline(psm_e["dc"], color=PALETTE["PSM'"], ls=":", lw=1.5, zorder=2)
    max_non = max(non_totals)
    ax_non.set_xlim(0.0, max_non * 1.72)
    ax_non.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.0f}k"))
    ax_non.set_xlabel("Non-Energy Cost (€k: Demand Charge + Unserved/Wear)")
    ax_non.set_title("B. Demand Charge vs. Shedding & Safety Leakage", loc="left", pad=22, fontsize=11.8)

    non_legend = [
        Patch(facecolor=PALETTE["DemandCharge"], label="Peak Demand Charge"),
        Patch(facecolor=PALETTE["Unserved"], label="Unserved VoLL + Wear"),
        Line2D([0], [0], color=PALETTE["PSM'"], ls=":", lw=1.5, label=f"PSM' DC (€{psm_e['dc'] / 1000.0:.1f}k)"),
    ]
    ax_non.legend(
        handles=non_legend,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=3,
        fontsize=8.3,
        columnspacing=0.9,
        handlelength=1.4,
        frameon=False,
    )

    for idx, (y, e, nt) in enumerate(zip(ys, entries, non_totals)):
        if e["ours"]:
            _highlight_row_band_and_tick(ax_non, idx, e["label"], PALETTE["PPO-Diff-Shield"])
        base_x = max(nt, psm_e["dc"]) if e["method"] == "Grid-Only" else nt
        x_txt = base_x + max_non * 0.02
        if e["method"] == "ci_ppo_diff_shield":
            lbl = f"DC €{e['dc'] / 1000.0:.1f}k ({_fmt_signed_eur(dc_diff)} vs PSM') | 0% viol"
            col = PALETTE["PPO-Diff-Shield"]
            fw = "bold"
        elif e["method"] == "ci_ppo_diff_6d":
            lbl = f"{e['viol_rate']:.2f}% ({e['viol_hrs']:.1f}h) motor leakage"
            col = PALETTE["Gate"]
            fw = "bold"
        elif e["viol_rate"] > 0.0:
            lbl = f"{e['viol_rate']:.2f}% ({e['viol_hrs']:.0f}h) cap viol"
            col = PALETTE["Gate"]
            fw = "bold"
        else:
            lbl = f"DC €{e['dc'] / 1000.0:.1f}k | 0% viol"
            col = "#334155"
            fw = "normal"
        t = ax_non.text(
            x_txt,
            y,
            lbl,
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight=fw,
            color=col,
            path_effects=_halo(),
            zorder=6,
        )
        texts_all.append(t)

    sh_e = next(e for e in entries if e["method"] == "ci_ppo_shielded")
    ci_cfg = load_config_with_base("ci_ppo_diff_shield")
    dc_rate = float(ci_cfg["demand_charge_eur_per_kw_week"])
    trip_eur = float(ci_cfg["industrial_trip_penalty_eur"])
    r2_weeks = int(_find_regime_row(regime_rows, "regime2_val", "Grid-Only", "single")["n_weeks"])

    add_slide_header(
        fig,
        f"Regime 2 C&I: Diff-Shield Cuts Peak Demand Charges While Unshielded 6-D Leaks {d6_e['viol_rate']:.2f}% Violations",
        (
            f"PPO-Diff-Shield achieves €{ds_e['cost'] / 1000.0:.2f}k ({_fmt_signed_eur(dc_diff)} demand-charge savings vs PSM', "
            f"beating Shielded PPO €{sh_e['cost'] / 1000.0:.2f}k), whereas unshielded 6-D suffers "
            f"{d6_e['viol_rate']:.2f}% ({d6_e['viol_hrs']:.1f}h) leakage."
        ),
        (
            f"Source: results/tables/regime_benchmarks_val.csv (suite=regime2_val, {r2_weeks} weeks, "
            f"{dc_rate:.4f} €/kW-wk demand charge, zone VoLL, {trip_eur:.0f} € trip)."
        ),
    )
    fig.subplots_adjust(left=0.17, right=0.975, top=0.80, bottom=0.13)
    _assert_no_overlaps(fig, texts_all, "fig_05")
    return save_slide_fig(fig, _dest_dir(out_dir), "05_regime2_ci_ablation", save_svg=save_svg)


def fig_06_ood_stress_2019w04(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot OOD stress suite results on the representative Val stress week (2019-W04)."""
    fig = _new_slide()
    regime_rows = load_regime_benchmarks()
    ood_rows = [r for r in regime_rows if str(r.get("suite", "")).startswith("ood_val_")]
    suite_name = ood_rows[0]["suite"] if ood_rows else "ood_val_2019-W04"
    week_label = suite_name.replace("ood_val_", "")

    contingencies = [
        ("nominal", "Nominal (No Fault)"),
        ("pv_inverter_trip", "PV Inverter Trip\n(11:00–15:00 = 0)"),
        ("bess_derate", "BESS Derate\n(50% Max Power)"),
        ("grid_cap_reduction", "Grid Cap Cut\n(80% Contract Cap)"),
        ("forecast_noise", "Forecast Noise\n(20% Lookahead Noise)"),
    ]

    # Index by (contingency, method, row_type) so mean/single never collide with std/per_seed
    lookup_by_type: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in ood_rows:
        rt = str(r.get("row_type") or "single")
        lookup_by_type[(r["contingency"], r["method"], rt)] = r

    def _pick_ood_row(c_key: str, m_key: str) -> Optional[Dict[str, Any]]:
        for rt in ("mean", "single"):
            found = lookup_by_type.get((c_key, m_key, rt))
            if found is not None:
                return found
        return None

    candidate_methods = [
        ("PSM'", _label("PSM'"), _color("PSM'"), False),
        ("Shield + PSM'", _label("Shield + PSM'"), _color("Shield + PSM'"), True),
        (
            "MPCPolicy(causal_noisy, H=24)",
            _label("MPCPolicy(causal_noisy, H=24)"),
            _color("MPCPolicy(causal_noisy, H=24)"),
            False,
        ),
        (
            "Shield + MPCPolicy(causal_noisy, H=24)",
            "Shield + MPC (Causal, 24h)",
            PALETTE["MPC-Persistence"],
            True,
        ),
        (
            "clamped_ppo_diff_shield",
            _label("clamped_ppo_diff_shield"),
            PALETTE["PPO-Diff-Shield"],
            False,
        ),
    ]
    methods: List[Tuple[str, str, str]] = []
    for m_key, m_lbl, m_col, is_opt in candidate_methods:
        has_all = all(_pick_ood_row(c_key, m_key) is not None for c_key, _ in contingencies)
        if is_opt and not has_all:
            continue
        if has_all or not is_opt:
            methods.append((m_key, m_lbl, m_col))

    plotted_rows = [
        row
        for c_key, _ in contingencies
        for m_key, _, _ in methods
        for row in (_pick_ood_row(c_key, m_key),)
        if row is not None
    ]

    ax_viol, ax_cost = fig.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.10}
    )
    ys = np.arange(len(contingencies), dtype=float)
    n_m = len(methods)
    if n_m == 3:
        offsets = [-0.25, 0.0, 0.25]
        bar_h = 0.22
        txt_fs = 8.2
    else:
        offsets = list(np.linspace(-0.33, 0.33, n_m))
        bar_h = min(0.22, 0.72 / max(1, n_m))
        txt_fs = 7.2 if n_m >= 5 else 7.8

    noise_idx = [c[0] for c in contingencies].index("forecast_noise")
    for ax in (ax_viol, ax_cost):
        ax.axhspan(noise_idx - 0.44, noise_idx + 0.44, color=PALETTE["PPO-Diff-Shield"], alpha=0.07, zorder=0)

    # Panel A: Violation Rate & Peak Exceedance
    style_clean_axes(ax_viol, grid_axis="x")
    ep_len = _episode_hours_from_split_report()
    max_vr = max(float(r["violation_rate_pct"]) for r in plotted_rows) if plotted_rows else 100.0
    ax_viol.set_xlim(0.0, max(125.0, max_vr * 1.30))
    ax_viol.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_viol.set_xlabel(f"Constraint Violation Rate (% of {ep_len}h)")
    ax_viol.set_yticks(ys)
    ax_viol.set_yticklabels([c[1] for c in contingencies], fontsize=10.0)
    ax_viol.invert_yaxis()
    ax_viol.set_title("A. Safety Violation Rate & Peak Exceedance", loc="left", pad=12, fontsize=11.8)

    # Panel B: Weekly Cost & CVaR95
    style_clean_axes(ax_cost, grid_axis="x")
    ax_cost.tick_params(labelleft=False)
    max_wc = max(float(r["cost_eur"]) for r in plotted_rows) if plotted_rows else 1000.0
    ax_cost.set_xlim(0.0, max_wc * 1.45)
    ax_cost.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.0f}k"))
    ax_cost.set_xlabel("Weekly Operational Cost (€k)")
    ax_cost.set_title("B. Weekly Cost & Tail Risk (CVaR₉₅)", loc="left", pad=12, fontsize=11.8)

    texts_all: List[Text] = []
    for c_idx, (c_key, _c_lbl) in enumerate(contingencies):
        for m_idx, (m_key, m_lbl, m_col) in enumerate(methods):
            row = _pick_ood_row(c_key, m_key)
            if row is None:
                continue
            y = ys[c_idx] + offsets[m_idx]
            vr = float(row["violation_rate_pct"])
            max_exc = float(row["max_hourly_exceedance_kw"])
            cost = float(row["cost_eur"])
            cvar95 = float(row["cvar95_cost_eur"])
            is_ours = m_key in OURS_KEYS

            v_col = (
                PALETTE["PPO-Diff-Shield"]
                if is_ours
                else (PALETTE["Gate"] if vr > 0.0 else m_col)
            )
            if vr > 0.0:
                ax_viol.barh(y, vr, height=bar_h, color=v_col, edgecolor="#FFFFFF", lw=0.6, zorder=3)
                v_txt = f"{vr:.1f}% (max {max_exc:.1f} kW)"
            else:
                ax_viol.scatter([0.8], [y], color=v_col, s=42 if is_ours else 30, zorder=4)
                v_txt = "0.0%"
            tv = ax_viol.text(
                (vr if vr > 0.0 else 2.6) + 1.5,
                y,
                v_txt,
                ha="left",
                va="center",
                fontsize=txt_fs,
                fontweight="bold" if (is_ours or vr > 0.0) else "normal",
                color=v_col,
            )
            texts_all.append(tv)

            ax_cost.barh(y, cost, height=bar_h, color=m_col, edgecolor="#FFFFFF", lw=0.6, zorder=3)
            c_txt = f"€{cost / 1000.0:.2f}k | CVaR €{cvar95:.0f}"
            is_noisy_mpc = c_key == "forecast_noise" and "MPCPolicy" in m_key
            tc = ax_cost.text(
                cost + max_wc * 0.015,
                y,
                c_txt,
                ha="left",
                va="center",
                fontsize=txt_fs,
                fontweight="bold" if (is_ours or (c_key == "forecast_noise" and m_key != "PSM'")) else "normal",
                color=PALETTE["PPO-Diff-Shield"] if is_ours else (PALETTE["Gate"] if is_noisy_mpc else "#1E293B"),
            )
            texts_all.append(tc)

    legend_handles = [Patch(facecolor=col, label=lbl) for _k, lbl, col in methods]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.875),
        ncol=min(5, len(methods)),
        fontsize=8.6 if len(methods) > 3 else 9.6,
        frameon=False,
        columnspacing=1.2 if len(methods) > 3 else 1.8,
    )

    _ = _pick_ood_row("forecast_noise", "PSM'")
    _ = _pick_ood_row("forecast_noise", "MPCPolicy(causal_noisy, H=24)")
    _ = _pick_ood_row("forecast_noise", "clamped_ppo_diff_shield")
    _ = _pick_ood_row("grid_cap_reduction", "PSM'")

    add_slide_header(
        fig,
        f"OOD Stress ({week_label}): Diff-Shield Holds 0.00% Violations Across All Faults & Beats MPC Under Noise",
        "",
    )
    fig.subplots_adjust(left=0.16, right=0.975, top=0.78, bottom=0.13)
    _assert_no_overlaps(fig, texts_all, "fig_06")
    return save_slide_fig(fig, _dest_dir(out_dir), "06_ood_stress_2019w04", save_svg=save_svg)


def _simulate_val_week(
    env: Any,
    start_idx: int,
    policy: Any = None,
    agent: Any = None,
) -> Dict[str, Any]:
    """Deterministically roll out one 168-hour Val week and collect hourly traces."""
    obs, _ = env.reset(options={"start_idx": int(start_idx)})
    base = getattr(env, "unwrapped", env)
    ep_len = int(base.episode_length)
    dt = float(base.dt_hours)
    grid_import, soc_pct, unserved_kwh, exc_kw, costs = [], [], [], [], []
    for _ in range(ep_len):
        act = agent.select_action(obs, evaluate=True)[0] if agent is not None else policy.get_action(obs)
        obs, _, terminated, truncated, info = env.step(act)
        grid_import.append(float(info["grid_import_kw"]))
        soc_pct.append(float(info["soc"]) * 100.0)
        unserved_kwh.append(float(info["total_unserved_kw"]) * dt)
        exc_kw.append(float(info["grid_exceedance_kw"]))
        costs.append(float(info["energy_cost"]) + float(info["unserved_cost"]) + float(info["battery_wear_cost"]))
        if terminated or truncated:
            break
    return {
        "grid_import_kw": np.asarray(grid_import, dtype=float),
        "soc_pct": np.asarray(soc_pct, dtype=float),
        "unserved_kwh": np.asarray(unserved_kwh, dtype=float),
        "exceedance_kw": np.asarray(exc_kw, dtype=float),
        "cost_eur": float(np.sum(costs)),
        "cap_kw": float(base.grid_import_cap_kw),
        "soc_min_pct": float(base.bat_soc_min) * 100.0,
        "soc_max_pct": float(base.bat_soc_max) * 100.0,
    }


def _resolve_seed101_ckpt(exp_id: str) -> Optional[str]:
    """Resolve seed-101 final checkpoint weights path for ``exp_id``."""
    cfg = load_config_with_base(exp_id)
    folder = agent_name(cfg["method"], cfg)
    run_json = os.path.join(REPO_ROOT, "results", "runs", folder, f"{exp_id}_seed101.json")
    if os.path.exists(run_json):
        with open(run_json, encoding="utf-8") as f:
            data = json.load(f)
        rel = data.get("final_checkpoint", {}).get("weights")
        if rel:
            cand = os.path.join(REPO_ROOT, rel)
            if os.path.exists(cand):
                return cand
    fallback = os.path.join(REPO_ROOT, "checkpoints", folder, f"{exp_id}_seed101_final.pt")
    if os.path.exists(fallback):
        return fallback
    return None


def fig_07_dispatch_case_2019w04(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot 168-hour dispatch trajectory on the Val stress week (2019-W04)."""
    fig = _new_slide()
    week_label, start_idx = _resolve_val_stress_week_idx()

    env_base = SmartGridEnv({"split": "val", "charge_clamp": True, "seed": 42})
    cap_kw = float(env_base.grid_import_cap_kw)
    soc_min_pct = float(env_base.bat_soc_min) * 100.0
    soc_max_pct = float(env_base.bat_soc_max) * 100.0

    traces = [
        ("Grid-Only", "Grid-Only", PALETTE["Grid-Only"], 1.5, _simulate_val_week(env_base, start_idx, policy=GridOnlyPolicy())),
        ("PSM'", "PSM'", PALETTE["PSM'"], 1.9, _simulate_val_week(env_base, start_idx, policy=PeakShaveMinPolicy({"cap_kw": cap_kw}))),
    ]
    missing_ckpts: List[str] = []
    for exp_id, disp_lbl, col, lw in (
        ("clamped_ppo_shielded", "Shielded PPO (seed 101)", PALETTE["Shielded PPO"], 2.0),
        ("clamped_ppo_diff_shield", "PPO-Diff-Shield (seed 101)", PALETTE["PPO-Diff-Shield"], 2.6),
    ):
        ckpt = _resolve_seed101_ckpt(exp_id)
        if ckpt is None:
            missing_ckpts.append(exp_id)
            continue
        cfg = load_config_with_base(exp_id)
        cfg["seed"] = 101
        env_rl = build_env(cfg, split="val", seed=201)
        if is_shielded_run(cfg["method"], cfg):
            env_rl = CapShield(env_rl)
        agent = load_agent_for_method(cfg["method"], cfg)
        agent.load_models(ckpt)
        traces.append((exp_id, disp_lbl, col, lw, _simulate_val_week(env_rl, start_idx, agent=agent)))

    wk_hours = len(traces[0][4]["grid_import_kw"])
    h_end = wk_hours - 1
    hrs = np.arange(wk_hours)

    axes = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0, 1.0], "hspace": 0.38})
    ax_grid, ax_soc, ax_uns = axes[0], axes[1], axes[2]

    # Row 1: Grid Import
    style_clean_axes(ax_grid, grid_axis="both")
    ax_grid.axhline(cap_kw, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=3, label=f"Contract Cap ({cap_kw:.1f} kW)")
    for exp_id, disp_lbl, col, lw, tr in traces:
        ls = "--" if exp_id == "Grid-Only" else ("-." if exp_id == "clamped_ppo_shielded" else "-")
        gi = tr["grid_import_kw"]
        ax_grid.plot(hrs, gi, color=col, lw=lw, ls=ls, label=disp_lbl, zorder=5 if exp_id in OURS_KEYS else 4)
        if exp_id == "Grid-Only":
            ax_grid.fill_between(hrs, cap_kw, gi, where=gi > cap_kw, color=PALETTE["Gate"], alpha=0.18, zorder=2, label="Grid-Only Cap Overruns")
    ax_grid.set_ylabel("Grid Import (kW)")
    ax_grid.set_title("Row 1. Net Grid Import vs. Contract Cap (Overruns Shaded)", loc="left", pad=8, fontsize=11.0)

    # Row 2: Battery SoC
    style_clean_axes(ax_soc, grid_axis="both")
    ax_soc.axhline(soc_min_pct, color="#94A3B8", ls=":", lw=1.2, zorder=2)
    ax_soc.axhline(soc_max_pct, color="#94A3B8", ls=":", lw=1.2, zorder=2)
    soc_endpoints: List[Tuple[str, float, str, bool]] = []
    for exp_id, disp_lbl, col, lw, tr in traces:
        if exp_id == "Grid-Only":
            continue
        ls = "-." if exp_id == "clamped_ppo_shielded" else "-"
        ax_soc.plot(hrs, tr["soc_pct"], color=col, lw=lw, ls=ls, zorder=5 if exp_id in OURS_KEYS else 4)
        end_soc = float(tr["soc_pct"][h_end])
        if exp_id == "clamped_ppo_shielded":
            lbl = f"Shielded: parked {end_soc:.0f}% (wk min {tr['soc_pct'].min():.1f}%)"
        elif exp_id == "clamped_ppo_diff_shield":
            lbl = f"Diff-Shield: {end_soc:.0f}% (wk [{tr['soc_pct'].min():.0f}%, {tr['soc_pct'].max():.0f}%])"
        else:
            lbl = f"{disp_lbl}: {end_soc:.0f}% (wk [{tr['soc_pct'].min():.0f}%, {tr['soc_pct'].max():.0f}%])"
        soc_endpoints.append((lbl, end_soc, col, exp_id in OURS_KEYS))

    ax_soc.set_ylim(max(0.0, soc_min_pct - 6.0), min(100.0, soc_max_pct + 7.0))
    ax_soc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_soc.set_ylabel("Battery SoC (%)")
    ax_soc.set_title(
        f"Row 2. Battery SoC Dynamics (Full {wk_hours}h Cycling [{soc_min_pct:.0f}%, {soc_max_pct:.0f}%] vs. Parking)",
        loc="left", pad=8, fontsize=11.0
    )

    _ = next((tr for eid, _, _, _, tr in traces if eid == "clamped_ppo_shielded"), None)
    _ = next((tr for eid, _, _, _, tr in traces if eid == "clamped_ppo_diff_shield"), None)
    row2_texts: List[Text] = []
    soc_sorted = sorted(enumerate(soc_endpoints), key=lambda item: item[1][1])
    repelled_soc = repel_y_positions(
        [item[1][1] for item in soc_sorted], min_sep=17.0, y_min=soc_min_pct + 2.0, y_max=soc_max_pct
    )
    for (_, (lbl, end_soc, col, is_ours)), ry in zip(soc_sorted, repelled_soc):
        ax_soc.plot([h_end, h_end + 1.8], [end_soc, ry], color=col, lw=0.9, alpha=0.7, clip_on=False)
        t = ax_soc.text(
            h_end + 2.2,
            ry,
            lbl,
            ha="left",
            va="center",
            fontsize=8.3,
            fontweight="bold" if (is_ours or "parked" in lbl) else "normal",
            color=col,
        )
        row2_texts.append(t)

    # Row 3: Cumulative Unserved
    style_clean_axes(ax_uns, grid_axis="both")
    cum_endpoints: List[Tuple[str, str, float, str, bool]] = []
    for exp_id, disp_lbl, col, lw, tr in traces:
        ls = "--" if exp_id == "Grid-Only" else ("-." if exp_id == "clamped_ppo_shielded" else "-")
        cum = np.cumsum(tr["unserved_kwh"])
        ax_uns.plot(hrs, cum, color=col, lw=lw, ls=ls, zorder=5 if exp_id in OURS_KEYS else 4)
        short = disp_lbl.split(" (")[0]
        cum_endpoints.append((short, col, float(cum[h_end]), exp_id, exp_id in OURS_KEYS))

    max_cum = max(ep[2] for ep in cum_endpoints)
    ax_uns.set_ylim(-max_cum * 0.06, max_cum * 1.14)
    ax_uns.set_xlim(0, wk_hours)
    xticks_24 = list(range(0, wk_hours + 1, 24))
    xtick_lbls = [f"{h} (D{h // 24 + 1})" if h < wk_hours else str(h) for h in xticks_24]
    ax_uns.set_xticks(xticks_24)
    ax_uns.set_xticklabels(xtick_lbls)
    ax_uns.set_xlabel(f"Hour Index Within {week_label} (Full {wk_hours}h Week: Day 1–Day {wk_hours // 24})")
    ax_uns.set_ylabel("Cum. Unserved (kWh)")
    ax_uns.set_title(f"Row 3. Cumulative Unserved Load (Full {wk_hours}h Week)", loc="left", pad=8, fontsize=11.0)

    cum_sorted = sorted(enumerate(cum_endpoints), key=lambda item: item[1][2])
    repelled_uns = repel_y_positions(
        [item[1][2] for item in cum_sorted], min_sep=max(520.0, max_cum * 0.14), y_min=0.0, y_max=max_cum * 1.08
    )
    row3_texts: List[Text] = []
    for (_, ep), ry in zip(cum_sorted, repelled_uns):
        short, col, wk_tot, _eid, is_ours = ep
        t = ax_uns.text(
            h_end + 2.2,
            ry,
            f"{short}: {wk_tot:,.0f} kWh",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="bold" if is_ours else "normal",
            color=col,
        )
        row3_texts.append(t)

    handles, labels = ax_grid.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.48, 0.865),
        ncol=min(6, len(handles)),
        fontsize=8.7,
        frameon=False,
        columnspacing=1.1,
    )

    subtitle_note = (
        f"[Missing checkpoints skipped: {', '.join(missing_ckpts)}]"
        if missing_ckpts
        else ""
    )
    add_slide_header(
        fig,
        f"Case Study ({week_label}): Diff-Shield Cycles Full Battery Capacity to Cut Shedding Under Peak Demand",
        subtitle_note,
    )
    fig.subplots_adjust(left=0.075, right=0.805, top=0.77, bottom=0.12)
    _assert_no_overlaps(fig, row2_texts + row3_texts, "fig_07")
    return save_slide_fig(fig, _dest_dir(out_dir), "07_dispatch_case_2019w04", save_svg=save_svg)


def fig_08_seasonal_test_context(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot Val H1 vs. Locked Test v0 H2 seasonal arbitrage collapse and W49 stress."""
    fig = _new_slide()
    constants = load_project_constants()
    baselines = load_baselines()
    per_seed = load_per_seed_table()
    regime_rows = load_regime_benchmarks()
    test_sum = load_test_evaluation_summary()
    stress_data = load_test_stress_weeks_data()
    total_hours = _val_total_hours(regime_rows, baselines)

    val_ds = _per_seed_summary(per_seed, "clamped_ppo_diff_shield", total_hours)
    val_sh = _per_seed_summary(per_seed, "clamped_ppo_shielded", total_hours)

    comp_rows = [
        ("LP' Oracle", PALETTE["LP'"], float(constants["lp_prime_val_eur"]), 0.0, float(constants["lp_prime_test_eur"]), 0.0, False),
        ("PSM'", PALETTE["PSM'"], float(baselines["PSM'"]["cost_eur"]), 0.0, float(test_sum["PSM'"]["cost_mean"]), 0.0, False),
        ("PPO-Diff-Shield\n(Val post-fix | Test v0)", PALETTE["PPO-Diff-Shield"], val_ds["cost_mean"], val_ds["cost_std"], float(test_sum["clamped_ppo_diff_shield"]["cost_mean"]), float(test_sum["clamped_ppo_diff_shield"]["cost_std"]), True),
        ("Shielded PPO (FB)", PALETTE["Shielded PPO"], val_sh["cost_mean"], val_sh["cost_std"], float(test_sum["clamped_ppo_shielded"]["cost_mean"]), float(test_sum["clamped_ppo_shielded"]["cost_std"]), False),
        ("Grid-Only (Unsafe)", PALETTE["Grid-Only"], float(baselines["Grid-Only"]["cost_eur"]), 0.0, float(test_sum["Grid-Only"]["cost_mean"]), 0.0, False),
    ]

    val_headroom = float(baselines["PSM'"]["cost_eur"]) - float(constants["lp_prime_val_eur"])
    test_headroom = float(test_sum["PSM'"]["cost_mean"]) - float(constants["lp_prime_test_eur"])

    raw_stress = stress_data.get("raw_rows", [])
    stress_specs = [
        ("Grid-Only", lambda r: r["model"] == "Grid-Only", _color("Grid-Only"), False),
        ("Vanilla PPO (MB)", lambda r: r["model"].startswith("clamped_mb_ppo_seed"), _color("clamped_mb_ppo"), False),
        ("PSM'", lambda r: r["model"] == "PSM'", _color("PSM'"), False),
        ("PPO-Diff-Shield (v0)", lambda r: r["model"].startswith("clamped_ppo_diff_shield_seed"), PALETTE["PPO-Diff-Shield"], True),
        ("Shielded PPO (MB)", lambda r: r["model"].startswith("clamped_mb_ppo_shielded_seed"), _color("clamped_mb_ppo_shielded"), False),
    ]
    stress_entries: List[Dict[str, Any]] = []
    for disp, pred, col, is_ours in stress_specs:
        mild_costs = [float(r["cost_eur"]) for r in raw_stress if pred(r) and r["stress_criterion"] != "highest_exceedance"]
        w49_costs = [float(r["cost_eur"]) for r in raw_stress if pred(r) and r["stress_criterion"] == "highest_exceedance"]
        w49_vhs = [float(r["violation_hours"]) for r in raw_stress if pred(r) and r["stress_criterion"] == "highest_exceedance"]
        mild_m, mild_s = _mean_std(mild_costs)
        w49_m, w49_s = _mean_std(w49_costs)
        vh_m, _ = _mean_std(w49_vhs)
        stress_entries.append({
            "label": disp,
            "color": col,
            "mild_mean": mild_m,
            "mild_std": mild_s,
            "w49_mean": w49_m,
            "w49_std": w49_s,
            "w49_vh": vh_m,
            "ours": is_ours,
        })

    ax_gap, ax_w49 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.15], "wspace": 0.34})
    texts_all: List[Text] = []

    # Panel A: Seasonal Arbitrage Collapse
    style_clean_axes(ax_gap, grid_axis="x")
    y_a = np.arange(len(comp_rows), dtype=float)
    bar_h = 0.34
    for idx, (lbl, col, vc, vs, tc, ts, is_ours) in enumerate(comp_rows):
        if is_ours:
            ax_gap.axhspan(idx - 0.44, idx + 0.44, color=col, alpha=0.07, zorder=0)
        ax_gap.barh(idx - bar_h / 2, vc, height=bar_h, color=col, alpha=0.92, edgecolor="#FFFFFF", lw=0.7, zorder=3)
        ax_gap.barh(idx + bar_h / 2, tc, height=bar_h, color=col, alpha=0.45, hatch="///", edgecolor=col, lw=0.8, zorder=3)
        if vs > 1.0:
            ax_gap.errorbar(vc, idx - bar_h / 2, xerr=vs, fmt="none", ecolor="#0F172A", elinewidth=1.2, capsize=2.5, zorder=5)
        if ts > 1.0:
            ax_gap.errorbar(tc, idx + bar_h / 2, xerr=ts, fmt="none", ecolor="#0F172A", elinewidth=1.2, capsize=2.5, zorder=5)
        t_v = ax_gap.text(
            vc + vs + 750.0, idx - bar_h / 2, f"H1: €{vc / 1000.0:.1f}k",
            ha="left", va="center", fontsize=8.3, fontweight="bold" if is_ours else "normal", color="#0F172A"
        )
        t_t = ax_gap.text(
            tc + ts + 750.0, idx + bar_h / 2, f"H2 v0: €{tc / 1000.0:.1f}k",
            ha="left", va="center", fontsize=8.3, fontweight="bold" if is_ours else "normal", color="#475569"
        )
        texts_all.extend([t_v, t_t])

    max_a = max(max(r[2] + r[3], r[4] + r[5]) for r in comp_rows)
    ax_gap.set_xlim(0.0, max_a * 1.32)
    ax_gap.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.0f}k"))
    ax_gap.set_xlabel("Split Operational Cost (€k)")
    ax_gap.set_yticks(y_a)
    ax_gap.set_yticklabels([r[0] for r in comp_rows], fontsize=9.5)
    ax_gap.invert_yaxis()
    ax_gap.set_ylim(len(comp_rows) + 0.95, -0.55)
    _highlight_row_band_and_tick(ax_gap, 2, comp_rows[2][0], PALETTE["PPO-Diff-Shield"])
    ax_gap.set_title("A. Seasonal Arbitrage Collapse (Val H1 vs. Test v0 H2)", loc="left", pad=22, fontsize=11.3)

    val_weeks = round(total_hours / max(1, _episode_hours_from_split_report()))
    sr_path = os.path.join(REPO_ROOT, "data", "processed", "split_report.json")
    test_weeks = int(round(float(test_sum.get("n_weeks", 24))))
    if os.path.exists(sr_path):
        with open(sr_path, encoding="utf-8") as f:
            test_weeks = int(json.load(f)["splits"]["test"]["n_weeks"])

    gap_legend = [
        Patch(facecolor="#475569", alpha=0.92, label=f"Val 2019 H1 ({val_weeks}w, post-fix)"),
        Patch(facecolor="#475569", alpha=0.45, hatch="///", edgecolor="#475569", label=f"Locked Test v0 2019 H2 ({test_weeks}w)"),
    ]
    ax_gap.legend(handles=gap_legend, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, fontsize=8.2, frameon=False, columnspacing=1.0)
    t_headroom = ax_gap.text(
        max_a * 1.29,
        len(comp_rows) + 0.15,
        f"LP'→PSM' headroom: H1 = €{val_headroom:,.0f} vs H2 = €{test_headroom:,.0f} ({val_headroom / test_headroom:.1f}× narrower)",
        ha="right",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#334155",
    )
    texts_all.append(t_headroom)

    # Panel B: Locked Test Stress Weeks
    style_clean_axes(ax_w49, grid_axis="x")
    y_b = np.arange(len(stress_entries), dtype=float)
    for idx, se in enumerate(stress_entries):
        col = se["color"]
        if se["ours"]:
            ax_w49.axhspan(idx - 0.42, idx + 0.42, color=col, alpha=0.07, zorder=0)
        ax_w49.plot([se["mild_mean"], se["w49_mean"]], [idx, idx], color=col, lw=2.4, alpha=0.8, zorder=3)
        ax_w49.scatter([se["mild_mean"]], [idx], facecolors="#FFFFFF", edgecolors=col, s=75, lw=1.8, zorder=4)
        if se["w49_std"] > 1.0:
            ax_w49.errorbar(se["w49_mean"], idx, xerr=se["w49_std"], fmt="none", ecolor=col, elinewidth=1.4, capsize=3.0, zorder=4)
        ax_w49.scatter([se["w49_mean"]], [idx], color=col, s=88 if se["ours"] else 75, edgecolors="#FFFFFF", lw=1.0, zorder=5)
        t_m = ax_w49.text(se["mild_mean"] - 70.0, idx, f"€{se['mild_mean'] / 1000.0:.2f}k", ha="right", va="center", fontsize=8.2, color="#475569")
        viol_tag = f"{se['w49_vh']:.0f}h viol" if se["w49_vh"] > 0.0 else "0h viol"
        w49_col = PALETTE["Gate"] if se["w49_vh"] > 0.0 else (PALETTE["PPO-Diff-Shield"] if se["ours"] else "#1E293B")
        t_w = ax_w49.text(
            se["w49_mean"] + se["w49_std"] + 85.0,
            idx,
            f"€{se['w49_mean'] / 1000.0:.2f}k ({viol_tag})",
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="bold" if (se["ours"] or se["w49_vh"] > 0.0) else "normal",
            color=w49_col,
        )
        texts_all.extend([t_m, t_w])

    mild_weeks = sorted({r["week"].split("-")[-1] for r in raw_stress if r["stress_criterion"] != "highest_exceedance"})
    stress_week = next((r["week"] for r in raw_stress if r["stress_criterion"] == "highest_exceedance"), "2019-W49")
    stress_short = stress_week.split("-")[-1]
    min_b = min(se["mild_mean"] for se in stress_entries)
    max_b = max(se["w49_mean"] + se["w49_std"] for se in stress_entries)
    ax_w49.set_xlim(min_b - 650.0, max_b + 1250.0)
    ax_w49.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.1f}k"))
    ax_w49.set_xlabel(f"Weekly Cost (€k): Mild Weeks ({'/'.join(mild_weeks)}) → {stress_short} Cap-Binding Stress")
    ax_w49.set_yticks(y_b)
    ax_w49.set_yticklabels([se["label"] for se in stress_entries], fontsize=9.8)
    ax_w49.invert_yaxis()
    _highlight_row_band_and_tick(ax_w49, 3, stress_entries[3]["label"], PALETTE["PPO-Diff-Shield"])
    ax_w49.set_title(f"B. Locked Test Stress Weeks (Mild Mean → {stress_week})", loc="left", pad=22, fontsize=11.3)

    n_mild = len(mild_weeks)
    w49_legend = [
        Line2D([0], [0], marker="o", color="#475569", markerfacecolor="#FFFFFF", markeredgewidth=1.8, markersize=7, lw=0, label=f"{n_mild} Mild Weeks Mean"),
        Line2D([0], [0], marker="o", color="#475569", markerfacecolor="#475569", markersize=7, lw=0, label=f"{stress_week} Cap-Binding Week"),
    ]
    ax_w49.legend(handles=w49_legend, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, fontsize=8.3, frameon=False, columnspacing=1.2)

    grid_w49 = next(se for se in stress_entries if se["label"] == "Grid-Only")
    psm_w49 = next(se for se in stress_entries if se["label"] == "PSM'")
    ds_w49 = next(se for se in stress_entries if se["label"].startswith("PPO-Diff-Shield"))

    add_slide_header(
        fig,
        f"Seasonal Shift Shrinks H2 Arbitrage {val_headroom / test_headroom:.1f}× While {stress_short} Drives All Safe Curtailment Cost",
        (
            f"Locked Test v0 (2019 H2) remains deliberately un-unlocked: LP'→PSM' headroom narrows from €{val_headroom:,.0f} (H1) "
            f"to €{test_headroom:,.0f} (H2), and {stress_week} drives safe curtailment "
            f"(€{psm_w49['w49_mean']:,.0f}–€{ds_w49['w49_mean']:,.0f}, 0h viol vs Grid-Only {grid_w49['w49_vh']:.0f}h viol)."
        ),
        "Source: Read-only CSVs — per_seed_metrics.csv, frozen_baselines.csv, test_evaluation.csv, and test_stress_weeks.csv (v0-prototype Test snapshot).",
    )
    fig.subplots_adjust(left=0.145, right=0.975, top=0.79, bottom=0.13)
    _assert_no_overlaps(fig, texts_all, "fig_08")
    return save_slide_fig(fig, _dest_dir(out_dir), "08_seasonal_test_context", save_svg=save_svg)


def _load_split_report() -> Dict[str, Any]:
    path = os.path.join(REPO_ROOT, "data", "processed", "split_report.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_yaml_config() -> Dict[str, Any]:
    path = os.path.join(REPO_ROOT, "configs", "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_physics_context() -> Dict[str, Any]:
    env_cfg = default_config()
    sr = _load_split_report()
    yaml_cfg = _load_yaml_config()
    constants = load_project_constants()
    causal_cfg = load_config_with_base("causal23_ppo_diff_shield")
    ci_cfg = load_config_with_base("ci_ppo_diff_shield")

    prep_cfg = yaml_cfg.get("data_preparation", {})
    raw_shares = sr.get("zone_shares") or {}
    zone_shares = {z: float(raw_shares.get(z, prep_cfg["zones"][z]["share"])) for z in ZONE_NAMES}
    pv_scale = (
        float(env_cfg["pv_area_m2"])
        * float(env_cfg["pv_efficiency"])
        * float(env_cfg["pv_performance_ratio"])
        / 1000.0
    )
    cap_meta = sr.get("grid_import_cap") or {}
    cap_kw = float(cap_meta.get("value_kw", constants["contract_cap_kw"]))
    cap_quantile_pct = int(round(float(cap_meta.get("quantile", prep_cfg.get("cap_quantile", 0.9))) * 100.0))
    motor_min_raw = float(env_cfg["motor_min_pct"])
    motor_min_frac = motor_min_raw / 100.0 if motor_min_raw > 1.0 else motor_min_raw

    return {
        "env_cfg": env_cfg,
        "split_report": sr,
        "yaml_cfg": yaml_cfg,
        "constants": constants,
        "causal_cfg": causal_cfg,
        "ci_cfg": ci_cfg,
        "zone_shares": zone_shares,
        "pv_scale": pv_scale,
        "pv_peak_kw": pv_scale * 1000.0,
        "microgrid_peak_kw": float(sr.get("microgrid_peak_kw", prep_cfg.get("microgrid_peak_kw", 500.0))),
        "cap_kw": cap_kw,
        "cap_quantile_pct": cap_quantile_pct,
        "motor_min_frac": motor_min_frac,
    }


def _draw_diagram_box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str = "",
    lines: Sequence[str] = (),
    facecolor: str = "#F8FAFC",
    edgecolor: str = "#CBD5E1",
    title_color: str = "#0F172A",
    body_color: str = "#334155",
    lw: float = 1.2,
    ls: str = "-",
    rounding: float = 1.0,
    title_size: float = 9.2,
    body_size: float = 8.1,
    align: str = "left",
    title_align: Optional[str] = None,
    zorder: int = 3,
) -> Tuple[FancyBboxPatch, List[Text]]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)

    texts: List[Text] = []
    t_align = title_align or align
    pad_x = 1.3
    tx = x + pad_x if t_align == "left" else x + 0.5 * w
    bx = x + pad_x if align == "left" else x + 0.5 * w

    if title and lines:
        title_y = y + h - 2.6
        t_head = ax.text(
            tx, title_y, title, ha=t_align, va="top",
            fontsize=title_size, fontweight="bold", color=title_color, zorder=zorder + 2
        )
        texts.append(t_head)
        top_body = title_y - 4.4
        bot_body = y + 2.0
        ys = np.linspace(top_body, bot_body, len(lines)) if len(lines) > 1 else [0.5 * (top_body + bot_body)]
        for ly, line in zip(ys, lines):
            tb = ax.text(
                bx, float(ly), line, ha=align, va="center",
                fontsize=body_size, color=body_color, zorder=zorder + 2
            )
            texts.append(tb)
    elif title:
        t_head = ax.text(
            tx, y + 0.5 * h, title, ha=t_align, va="center",
            fontsize=title_size, fontweight="bold", color=title_color, zorder=zorder + 2
        )
        texts.append(t_head)
    elif lines:
        ys = (
            np.linspace(y + h - 2.6, y + 2.6, len(lines))
            if len(lines) > 1
            else [y + 0.5 * h]
        )
        for ly, line in zip(ys, lines):
            tb = ax.text(
                bx, float(ly), line, ha=align, va="center",
                fontsize=body_size, color=body_color, zorder=zorder + 2
            )
            texts.append(tb)
    return patch, texts


def _draw_arrow(
    ax: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = "#475569",
    lw: float = 1.6,
    style: str = "-|>",
    ls: str = "-",
    rad: float = 0.0,
    mutation: float = 11.5,
    zorder: int = 4,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        linestyle=ls,
        mutation_scale=mutation,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _new_schematic_axes(fig: Figure) -> Axes:
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.80])
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_axis_off()
    return ax


def _assert_diagram_clean(
    fig: Figure,
    box_text_pairs: Sequence[Tuple[FancyBboxPatch, Sequence[Text]]],
    all_texts: Sequence[Text],
    tag: str,
) -> None:
    _assert_no_overlaps(fig, all_texts, tag)
    renderer = fig.canvas.get_renderer()
    for patch, texts in box_text_pairs:
        pb = patch.get_window_extent(renderer=renderer)
        for t in texts:
            if not (t.get_visible() and t.get_text().strip()):
                continue
            tb = t.get_window_extent(renderer=renderer)
            if (
                tb.x0 < pb.x0 - 1.5
                or tb.x1 > pb.x1 + 1.5
                or tb.y0 < pb.y0 - 1.5
                or tb.y1 > pb.y1 + 1.5
            ):
                raise RuntimeError(
                    f"[{tag}] Text {t.get_text()!r} overflows its diagram box "
                    f"(text=[{tb.x0:.1f},{tb.x1:.1f}]x[{tb.y0:.1f},{tb.y1:.1f}], "
                    f"box=[{pb.x0:.1f},{pb.x1:.1f}]x[{pb.y0:.1f},{pb.y1:.1f}])."
                )


def fig_09_microgrid_architecture(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot clean schematic of the 5-zone microgrid CMDP architecture."""
    fig = _new_slide()
    ctx = _load_physics_context()
    env_cfg = ctx["env_cfg"]
    ci_cfg = ctx["ci_cfg"]
    causal_cfg = ctx["causal_cfg"]
    consts = ctx["constants"]
    shares = ctx["zone_shares"]
    zone_voll = ci_cfg["zone_voll_eur_kwh"]

    ax = _new_schematic_axes(fig)
    texts_all: List[Text] = []
    box_pairs: List[Tuple[FancyBboxPatch, Sequence[Text]]] = []

    h_horizon = int(causal_cfg["forecast_horizon_hours"])
    causal_dim = int(causal_cfg["state_dim"])
    phys_obs_dim = 2 + len(ZONE_NAMES) + 1
    exog_obs_dim = BASE_OBS_DIM - phys_obs_dim
    cap_kw = ctx["cap_kw"]
    cap_q = ctx["cap_quantile_pct"]
    bat_kwh = float(env_cfg["battery_capacity_kwh"])
    bat_kw = float(env_cfg["battery_max_power_kw"])
    eta_c = float(env_cfg["battery_eta_charge"]) * 100.0
    soc_lo = float(env_cfg["battery_soc_min"]) * 100.0
    soc_hi = float(env_cfg["battery_soc_max"]) * 100.0
    pv_peak = ctx["pv_peak_kw"]
    pv_area = float(env_cfg["pv_area_m2"])
    pv_eff = float(env_cfg["pv_efficiency"]) * 100.0
    pv_pr = float(env_cfg["pv_performance_ratio"])
    mg_peak = ctx["microgrid_peak_kw"]
    motor_lo = ctx["motor_min_frac"]
    wear_eur = float(env_cfg["cost_wear_eur_kwh"])
    voll_r1 = float(env_cfg["cost_unserved_eur_kwh"])
    dc_rate = float(ci_cfg["demand_charge_eur_per_kw_week"])
    trip_eur = float(ci_cfg["industrial_trip_penalty_eur"])
    g1_kw = float(consts["g1_exceedance_kw"])
    g2_eur = float(consts["g2_cost_eur"])
    g2_mult = float(consts["g2_cost_multiplier"])
    n_shed_steps = len(SHED_ORDER)

    # Top closed-loop policy arc
    ax.plot([13.25, 13.25, 86.75, 86.75], [94.5, 98.2, 98.2, 95.3], color=PALETTE["PPO-Diff-Shield"], lw=1.6, ls="--", zorder=2)
    _draw_arrow(ax, (86.75, 96.2), (86.75, 94.5), color=PALETTE["PPO-Diff-Shield"], lw=1.6, mutation=11.0, zorder=4)
    t_arc = ax.text(
        50.0,
        98.2,
        f"Closed-Loop Policy π_θ(s_t) → a_t ({ACTION_DIM}-D Continuous Control)",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["PPO-Diff-Shield"],
        bbox=dict(facecolor="#FFFFFF", edgecolor="none", pad=2.2),
        zorder=6,
    )
    texts_all.append(t_arc)

    # Column headers
    for cx, cw, ctitle, cfc, cec, ctc in (
        (0.5, 25.5, f"1. State Space s_t ({BASE_OBS_DIM}-D / {causal_dim}-D)", "#EFF6FF", "#93C5FD", "#1E3A8A"),
        (30.5, 39.0, f"2. Physical Microgrid & CMDP Step ({mg_peak:.0f} kW Peak)", "#F1F5F9", "#94A3B8", "#0F172A"),
        (74.0, 25.5, f"3. {ACTION_DIM}-D Action, Safety & Gates", "#F0FDFA", "#5EEAD4", PALETTE["PPO-Diff-Shield"]),
    ):
        p_c, t_c = _draw_diagram_box(
            ax, cx, 88.5, cw, 6.0, title=ctitle, facecolor=cfc, edgecolor=cec,
            title_color=ctc, title_size=9.2, align="center", rounding=0.9
        )
        box_pairs.append((p_c, t_c))
        texts_all.extend(t_c)

    # Left column: State boxes
    for by, bh, btitle, blines, bfc, bec, btc, bls, blw in (
        (
            59.5, 26.5, f"Physical & Market State ({phys_obs_dim}-D)",
            [
                f"• BESS SoC_t ∈ [{soc_lo:.0f}%, {soc_hi:.0f}%]",
                "• Rooftop PV P_pv,t (kW)",
                f"• {len(ZONE_NAMES)} Zone Loads L_1..{len(ZONE_NAMES)},t (kW)",
                "• Spot Price λ_t (€/MWh)",
            ],
            "#FFFFFF", "#BFDBFE", "#1E3A8A", "-", 1.2,
        ),
        (
            34.0, 23.0, f"Weather & Exogenous ({exog_obs_dim}-D)",
            [
                "• Weather: Temp, RH, Pressure",
                "• Calendar: sin/cos(hour, dow)",
                "• Stadium Event: active & t+1",
            ],
            "#FFFFFF", "#BFDBFE", "#1E3A8A", "-", 1.2,
        ),
        (
            2.5, 29.0, f"+{h_horizon}h Forecasts (+{FORECAST_FEATURE_DIM}-D = {causal_dim}-D)",
            [
                f"• causal_noisy (H={h_horizon}h lookahead)",
                "• Net load: next, max, mean",
                "• D-1 price: next, min, max",
                "• PV/load √(h/H) noise; exact D-1",
            ],
            "#ECFEFF", PALETTE["Causal-Diff-Shield"], "#0E7490", "--", 1.3,
        ),
    ):
        p_s, t_s = _draw_diagram_box(
            ax, 0.5, by, 25.5, bh, title=btitle, lines=blines, facecolor=bfc,
            edgecolor=bec, title_color=btc, ls=bls, lw=blw, title_size=8.8, body_size=7.9
        )
        box_pairs.append((p_s, t_s))
        texts_all.extend(t_s)

    # Center column top: Grid, PV, BESS
    for ax_x, aw, atitle, alines, afc, aec, atc in (
        (30.8, 12.2, "Utility Grid PCC", [f"≤ {cap_kw:.2f} kW", f"(P{cap_q} cap)", "No export (≥ 0)"], "#FEF2F2", "#FCA5A5", "#991B1B"),
        (43.9, 12.2, "Rooftop PV", [f"{pv_peak:.0f} kWp peak", f"{pv_area:,.0f} m² · {pv_eff:.0f}%", f"PR={pv_pr:.2f} (curt≥0)"], "#FFFBEB", "#FCD34D", "#92400E"),
        (57.0, 12.2, "BESS (0.5C)", [f"{bat_kwh:.0f} kWh / ±{bat_kw:.0f} kW", f"η_c, η_d = {eta_c:.0f}%", f"SoC [{soc_lo:.0f}%, {soc_hi:.0f}%]"], "#F0FDFA", "#5EEAD4", PALETTE["PPO-Diff-Shield"]),
    ):
        p_a, t_a = _draw_diagram_box(
            ax, ax_x, 63.0, aw, 23.0, title=atitle, lines=alines, facecolor=afc,
            edgecolor=aec, title_color=atc, title_size=8.4, body_size=7.7
        )
        box_pairs.append((p_a, t_a))
        texts_all.extend(t_a)

    # Center column middle: Power Balance Bus
    p_bus, t_bus = _draw_diagram_box(
        ax, 30.8, 50.2, 38.4, 9.2,
        lines=[
            f"Microgrid Power Balance (Δt = {float(env_cfg['dt_hours']):.0f}h):",
            "P_grid = max(0, Σ L_z(1 - s_z) + P_bat - P_pv)",
        ],
        facecolor="#1E293B", edgecolor="#0F172A", body_color="#FFFFFF",
        body_size=8.1, align="center", rounding=0.7
    )
    for tb in t_bus:
        tb.set_fontweight("bold")
    box_pairs.append((p_bus, t_bus))
    texts_all.extend(t_bus)

    _draw_arrow(ax, (36.9, 63.0), (36.9, 59.4), color="#991B1B", lw=1.4)
    _draw_arrow(ax, (50.0, 63.0), (50.0, 59.4), color="#B45309", lw=1.4)
    _draw_arrow(ax, (63.1, 63.0), (63.1, 59.4), color=PALETTE["PPO-Diff-Shield"], lw=1.4, style="<|-|>")

    # Center column bottom: 5 Zones
    frame_patch = FancyBboxPatch(
        (30.8, 2.5), 38.4, 44.2, boxstyle="round,pad=0.0,rounding_size=1.0",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2, zorder=2
    )
    ax.add_patch(frame_patch)
    t_zhead = ax.text(
        50.0, 44.6, f"{len(ZONE_NAMES)} Heterogeneous Demand Zones (Shares & Physics)",
        ha="center", va="center", fontsize=8.7, fontweight="bold", color="#0F172A", zorder=4
    )
    box_pairs.append((frame_patch, [t_zhead]))
    texts_all.append(t_zhead)
    _draw_arrow(ax, (50.0, 50.2), (50.0, 46.7), color="#334155", lw=1.5)

    zone_rows = [
        ("residential", f"Residential ({shares['residential'] * 100:.0f}%)", f"Bimodal peaks | R2 VoLL €{float(zone_voll['residential']):.2f}/kWh", "#F8FAFC", "#CBD5E1", "#1E293B"),
        ("commercial", f"Commercial ({shares['commercial'] * 100:.0f}%)", f"08–18h business | R2 VoLL €{float(zone_voll['commercial']):.2f}/kWh", "#F8FAFC", "#CBD5E1", "#1E293B"),
        ("industrial", f"Industrial ({shares['industrial'] * 100:.0f}%)", f"Motor u ∈ {{0}} ∪ [{motor_lo:.2f}, 1.00] | VoLL €{float(zone_voll['industrial']):.2f}", "#FEF3C7", "#D97706", "#92400E"),
        ("stadium", f"Stadium ({shares['stadium'] * 100:.0f}%)", f"Event-driven peaks | R2 VoLL €{float(zone_voll['stadium']):.2f}/kWh", "#F8FAFC", "#CBD5E1", "#1E293B"),
        ("street", f"Street Light ({shares['street'] * 100:.0f}%)", f"Irradiance-driven | R2 VoLL €{float(zone_voll['street']):.2f}/kWh", "#F8FAFC", "#CBD5E1", "#1E293B"),
    ]
    z_ys = np.linspace(35.3, 3.9, len(zone_rows))
    for zy, (zk, z_left, z_right, z_fc, z_ec, z_tc) in zip(z_ys, zone_rows):
        pz, tz = _draw_diagram_box(
            ax, 31.8, float(zy), 36.4, 6.7, title=f"{z_left}:  {z_right}",
            facecolor=z_fc, edgecolor=z_ec, title_color=z_tc, title_size=7.65,
            align="left", rounding=0.6, lw=1.3 if zk == "industrial" else 0.9, zorder=4
        )
        box_pairs.append((pz, tz))
        texts_all.extend(tz)

    # Right column: Action, Safety, Cost, Gates
    for cy, ch, ctitle, clines, cfc, cec, ctc, clw in (
        (
            68.5, 17.5, f"{ACTION_DIM}-D Control Action a_t",
            [
                f"• a_bat ∈ [-1, 1] (±{bat_kw:.0f} kW)",
                f"• a_shed[1..{len(ZONE_NAMES)}] ∈ [0, 1]^{len(ZONE_NAMES)}",
            ],
            "#FFFFFF", "#CBD5E1", "#0F172A", 1.2,
        ),
        (
            45.5, 19.5, "Safety Projection & Guard",
            [
                "• DiffSafetyLayer (T3 STE)",
                f"• + {n_shed_steps}-step CapShield motor guard",
                "• Hard cap: P_grid ≤ P_cap (0% viol)",
            ],
            "#F0FDFA", PALETTE["PPO-Diff-Shield"], PALETTE["PPO-Diff-Shield"], 1.6,
        ),
        (
            22.0, 20.0, "Economic Cost & C&I Tariffs",
            [
                f"• R1: Spot + €{wear_eur:.2f} wear + €{voll_r1:.2f} VoLL",
                f"• R2: +{dc_rate:.2f} €/kW-wk Demand Charge",
                f"• Zone VoLL + €{trip_eur:.0f} Industrial trip",
            ],
            "#FFFBEB", "#F59E0B", "#92400E", 1.2,
        ),
        (
            2.5, 16.0, "Evaluation Gates (G1 / G2)",
            [
                f"• G1 Safety: Mean Exc ≤ {g1_kw:.2f} kW",
                f"• G2 Econ: Cost ≤ {g2_mult:.2f}×PSM' (€{g2_eur / 1000.0:.1f}k)",
            ],
            "#FEF2F2", PALETTE["Gate"], PALETTE["Gate"], 1.4,
        ),
    ):
        pc, tc = _draw_diagram_box(
            ax, 74.0, cy, 25.5, ch, title=ctitle, lines=clines, facecolor=cfc,
            edgecolor=cec, title_color=ctc, lw=clw, title_size=8.8, body_size=7.75
        )
        box_pairs.append((pc, tc))
        texts_all.extend(tc)

    _draw_arrow(ax, (86.75, 68.5), (86.75, 65.0), color="#334155", lw=1.5)
    _draw_arrow(ax, (86.75, 45.5), (86.75, 42.0), color=PALETTE["PPO-Diff-Shield"], lw=1.5)
    _draw_arrow(ax, (86.75, 22.0), (86.75, 18.5), color="#92400E", lw=1.5)
    _draw_arrow(ax, (26.0, 48.0), (30.8, 48.0), color="#1E3A8A", lw=1.6)
    _draw_arrow(ax, (74.0, 54.8), (69.2, 54.8), color=PALETTE["PPO-Diff-Shield"], lw=1.8)
    _draw_arrow(ax, (69.2, 32.0), (74.0, 32.0), color="#92400E", lw=1.6)

    t_in = ax.text(28.4, 50.2, "s_t", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#1E3A8A")
    t_out = ax.text(71.6, 57.0, "u_t*", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=PALETTE["PPO-Diff-Shield"])
    t_fb = ax.text(71.6, 34.2, "r_t, g_t", ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="#92400E")
    texts_all.extend([t_in, t_out, t_fb])

    add_slide_header(
        fig,
        "Microgrid CMDP Architecture: 5 Heterogeneous Zones, Discontinuous Motor & Dual-Regime Safety",
        (
            f"Hourly dispatch coordinates {bat_kwh:.0f} kWh / ±{bat_kw:.0f} kW BESS and {pv_peak:.0f} kWp PV under a "
            f"{cap_kw:.2f} kW (P{cap_q}) import cap, {int(round(voll_r1 / wear_eur))}:1 VoLL-to-wear asymmetry, and discrete motor bounds."
        ),
        "Source: env/smartgrid_env.py, env/cap_shield.py, configs/config.yaml, and data/processed/split_report.json.",
    )
    _assert_diagram_clean(fig, box_pairs, texts_all, "fig_09")
    return save_slide_fig(fig, _dest_dir(out_dir), "09_microgrid_architecture", save_svg=save_svg)


def fig_10_dataset_split_regimes(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot chronological weekly peak net demand and cap violation hours across Train/Val/Test."""
    fig = _new_slide()
    ctx = _load_physics_context()
    env_cfg = ctx["env_cfg"]
    sr = ctx["split_report"]
    _ = ctx["ci_cfg"]

    cap_kw = ctx["cap_kw"]
    cap_q = ctx["cap_quantile_pct"]
    pv_scale = ctx["pv_scale"]
    viol_tol = float(env_cfg["violation_tolerance_kw"])
    ep_len = int(env_cfg["episode_length_hours"])

    split_defs = [
        ("train", "Train ('17–'18)", "#CBD5E1", "#475569"),
        ("val", "Val (H1 '19)", "#CCFBF1", PALETTE["PPO-Diff-Shield"]),
        ("test", "Test (H2 '19)", "#FEF3C7", "#B45309"),
    ]
    split_records: List[Dict[str, Any]] = []
    all_peaks, all_viols, all_week_labels = [], [], []
    week_offset = 0

    for split_key, split_title, bg_col, fg_col in split_defs:
        pq_path = os.path.join(REPO_ROOT, "data", "processed", f"{split_key}.parquet")
        df = pd.read_parquet(pq_path)
        pv_kw = df["irradiance_wm2"].to_numpy(dtype=float) * pv_scale
        wind_kw = df["wind_kw"].to_numpy(dtype=float) if "wind_kw" in df.columns else np.zeros(len(df))
        net_kw = np.maximum(0.0, df["load_kw"].to_numpy(dtype=float) - pv_kw - wind_kw)

        n_w = len(df) // ep_len
        net_mat = net_kw[: n_w * ep_len].reshape(n_w, ep_len)
        peaks = net_mat.max(axis=1)
        viols = (net_mat > (cap_kw + viol_tol)).sum(axis=1)
        w_labels = []
        for lw in range(n_w):
            iso = df.index[lw * ep_len].isocalendar()
            w_labels.append(f"{iso.year}-W{iso.week:02d}")

        split_records.append({
            "key": split_key,
            "title": split_title,
            "bg": bg_col,
            "fg": fg_col,
            "start_w": week_offset,
            "end_w": week_offset + n_w - 1,
            "n_w": n_w,
            "total_viol_h": int(viols.sum()),
            "max_peak_kw": float(peaks.max()),
        })
        all_peaks.append(peaks)
        all_viols.append(viols)
        all_week_labels.extend(w_labels)
        week_offset += n_w

    weekly_peaks = np.concatenate(all_peaks)
    weekly_viols = np.concatenate(all_viols)
    total_weeks = len(weekly_peaks)
    weeks_x = np.arange(total_weeks, dtype=float)

    val_stress_week = sr["stress_weeks"]["val"]["highest_exceedance"]["week"]
    test_stress_week = sr["stress_weeks"]["test"]["highest_exceedance"]["week"]
    w04_idx = all_week_labels.index(val_stress_week)
    w49_idx = all_week_labels.index(test_stress_week)

    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.25, 1.0], hspace=0.14)
    ax_peak = fig.add_subplot(gs[0])
    ax_viol = fig.add_subplot(gs[1], sharex=ax_peak)

    texts_all: List[Text] = []
    style_clean_axes(ax_peak, grid_axis="y")
    for srec in split_records:
        ax_peak.axvspan(srec["start_w"] - 0.5, srec["end_w"] + 0.5, facecolor=srec["bg"], alpha=0.35, zorder=0)
        ax_viol.axvspan(srec["start_w"] - 0.5, srec["end_w"] + 0.5, facecolor=srec["bg"], alpha=0.35, zorder=0)
    for b_idx in (split_records[0]["end_w"] + 0.5, split_records[1]["end_w"] + 0.5):
        ax_peak.axvline(b_idx, color="#475569", ls=":", lw=1.4, zorder=2)
        ax_viol.axvline(b_idx, color="#475569", ls=":", lw=1.4, zorder=2)

    ax_peak.plot(weeks_x, weekly_peaks, color="#334155", lw=1.55, zorder=3, label="Weekly Peak Net Load")
    ax_peak.fill_between(
        weeks_x, cap_kw, weekly_peaks, where=weekly_peaks > cap_kw, interpolate=True,
        color=PALETTE["Gate"], alpha=0.24, zorder=2, label="Cap Exceedance Region"
    )
    ax_peak.axhline(cap_kw, color=PALETTE["Gate"], ls="--", lw=1.8, zorder=4, label=f"P_cap = {cap_kw:.2f} kW (P{cap_q})")

    y_pk_lo = max(100.0, float(weekly_peaks.min()) - 25.0)
    y_pk_hi = float(weekly_peaks.max()) * 1.18
    ax_peak.set_ylim(y_pk_lo, y_pk_hi)
    ax_peak.set_xlim(-1.0, total_weeks + 1.0)
    ax_peak.set_ylabel("Weekly Peak Net Load (kW)")
    ax_peak.tick_params(labelbottom=False)
    ax_peak.set_title(f"Chronological {total_weeks}-Week Net Demand & Exceedance Timeline", loc="left", pad=22, fontsize=12.0)

    for srec in split_records:
        cx = 0.5 * (srec["start_w"] + srec["end_w"])
        ts = ax_peak.text(
            cx,
            y_pk_hi - 8.0,
            f"{srec['title']}\n{srec['n_w']} weeks · {srec['total_viol_h']:,}h viol",
            ha="center",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color=srec["fg"],
            path_effects=_halo(),
            zorder=6,
        )
        texts_all.append(ts)

    for w_idx, w_lbl, col, dx_off, ha_dir in (
        (w04_idx, val_stress_week, PALETTE["PPO-Diff-Shield"], 3.0, "left"),
        (w49_idx, test_stress_week, "#B45309", -3.0, "right"),
    ):
        pk_val = float(weekly_peaks[w_idx])
        ax_peak.scatter([w_idx], [pk_val], color=col, s=64, edgecolors="#FFFFFF", lw=1.1, zorder=5)
        tw = ax_peak.text(
            w_idx + dx_off,
            pk_val + 14.0,
            f"{w_lbl}\n{pk_val:.1f} kW",
            ha=ha_dir,
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=col,
            path_effects=_halo(),
            zorder=6,
        )
        ax_peak.plot([w_idx, w_idx + dx_off * 0.7], [pk_val, pk_val + 12.0], color=col, lw=1.0, zorder=4)
        texts_all.append(tw)

    ax_peak.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, fontsize=9.2, frameon=False, columnspacing=1.4)

    style_clean_axes(ax_viol, grid_axis="y")
    bar_cols = [PALETTE["Gate"] if vh > 0 else "#94A3B8" for vh in weekly_viols]
    ax_viol.bar(weeks_x, weekly_viols, width=0.82, color=bar_cols, alpha=0.82, zorder=3)
    ax_viol.set_ylim(0.0, float(weekly_viols.max()) * 1.24)
    ax_viol.set_ylabel("Viol (h/wk)")
    ax_viol.set_xlabel("Chronological Weekly Index (2017–2019, Strictly Non-Overlapping Splits)")

    tick_candidates = [0, 26, 52, 78, split_records[1]["start_w"], split_records[2]["start_w"], total_weeks - 1]
    ax_viol.set_xticks(tick_candidates)
    ax_viol.set_xticklabels([all_week_labels[i] for i in tick_candidates], fontsize=9.2)

    for w_idx, w_lbl, col, ha_dir, dx in (
        (w04_idx, val_stress_week, PALETTE["PPO-Diff-Shield"], "left", 2.0),
        (w49_idx, test_stress_week, "#B45309", "right", -2.0),
    ):
        vh_val = int(weekly_viols[w_idx])
        tv = ax_viol.text(
            w_idx + dx,
            vh_val + 3.0,
            f"{w_lbl}: {vh_val}h",
            ha=ha_dir,
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=col,
            path_effects=_halo(),
            zorder=5,
        )
        texts_all.append(tv)

    add_slide_header(
        fig,
        f"Chronological {total_weeks}-Week Dataset Splits (Train / Val / Test)",
        "",
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.81, bottom=0.12)
    _assert_no_overlaps(fig, texts_all, "fig_10")
    return save_slide_fig(fig, _dest_dir(out_dir), "10_dataset_split_regimes", save_svg=save_svg)


def fig_11_safety_mechanisms(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot 3-track mechanistic comparison of gradient flow in safe RL."""
    fig = _new_slide()
    ctx = _load_physics_context()
    per_seed = load_per_seed_table()
    regime_rows = load_regime_benchmarks()
    baselines = load_baselines()
    total_hours = _val_total_hours(regime_rows, baselines)
    consts = ctx["constants"]
    env_cfg = ctx["env_cfg"]

    sh_fb = _per_seed_summary(per_seed, "clamped_ppo_shielded", total_hours)
    aux_fb = _per_seed_summary(per_seed, "clamped_ppo_shield_aux", total_hours)
    ds_fin_m = _find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "mean", "final")
    ds_best_m = _find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "mean", "best")
    ci_ds_m = _find_regime_row(regime_rows, "regime2_val", "ci_ppo_diff_shield", "mean", "final")
    ci_d6_m = _find_regime_row(regime_rows, "regime2_val", "ci_ppo_diff_6d", "mean", "final")

    g2_eur = float(consts["g2_cost_eur"])
    voll_r1 = float(env_cfg["cost_unserved_eur_kwh"])
    soc_lo = float(env_cfg["battery_soc_min"]) * 100.0
    soc_hi = float(env_cfg["battery_soc_max"]) * 100.0
    motor_lo = ctx["motor_min_frac"]
    ste_alpha = float(DifferentiableSafetyLayer().surrogate_grad_scale)
    n_shed_steps = len(SHED_ORDER)

    ax = _new_schematic_axes(fig)
    texts_all: List[Text] = []
    box_pairs: List[Tuple[FancyBboxPatch, Sequence[Text]]] = []

    tracks = [
        {
            "y": 68.5,
            "h": 29.0,
            "banner": "Track A: Post-Hoc Shielded PPO — External Filter Disconnects Gradient → Battery Parking",
            "bg": "#FDF2F8",
            "border": "#F9A8D4",
            "head_col": "#9D174D",
            "blocks": [
                ("Actor π_θ(s_t)", ["Proposes raw a_bat ∈ [-1, 1]", "No internal cap awareness"], "#FFFFFF", "#CBD5E1", "#0F172A"),
                ("External CapShield", ["Env wrapper intercepts a_bat", f"Sheds load at €{voll_r1:.2f}/kWh"], "#FCE7F3", PALETTE["Shielded PPO"], "#831843"),
                ("Environment Step", ["Observes safe P_grid ≤ P_cap", "Zero penalty for idling BESS"], "#FFFFFF", "#CBD5E1", "#0F172A"),
                ("Observed Pathology", [f"Active SoC: {sh_fb['active_soc_mean']:.1f}% (parked)", f"Cost €{sh_fb['cost_mean'] / 1000.0:.1f}k (> G2) | {sh_fb['viol_rate_mean']:.2f}% viol"], "#FEF2F2", PALETTE["Gate"], PALETTE["Gate"]),
            ],
            "grad_start": (37.0, 72.8),
            "grad_end": (12.5, 72.8),
            "grad_col": PALETTE["Gate"],
            "grad_ls": "--",
            "grad_text": "✖  Gradient stops at CapShield boundary (∂L/∂θ = 0 for shedding): policy idles BESS to dodge wear",
            "ours": False,
        },
        {
            "y": 35.5,
            "h": 29.0,
            "banner": "Track B: v0 Analytical 1-D Projection — Vanishing Active Jacobian & Double-Projection Bias",
            "bg": "#FFFBEB",
            "border": "#FCD34D",
            "head_col": "#92400E",
            "blocks": [
                ("Actor π_θ(s_t)", ["Outputs raw μ_θ(s_t)", "Log-prob mismatch if projected"], "#FFFFFF", "#CBD5E1", "#0F172A"),
                ("1-D Analytical Clip (v0)", ["ã₀ = clip(a₀ - ReLU(g)/P_max)", "Active cap: ∂ã₀/∂a₀ = 1 - 1 = 0"], "#FEF3C7", "#D97706", "#92400E"),
                ("External CapShield", ["5-zone curtailment & SoC clamp", "Fixes residual cap breach"], "#FFFFFF", "#CBD5E1", "#0F172A"),
                ("Observed Pathology", [f"Stalled > G2 (€{g2_eur / 1000.0:.1f}k) on 1-D clip", f"Aux-Shield: €{aux_fb['cost_mean'] / 1000.0:.1f}k ({aux_fb['viol_rate_mean']:.2f}% viol)"], "#FEF2F2", "#D97706", "#92400E"),
            ],
            "grad_start": (37.0, 39.8),
            "grad_end": (12.5, 39.8),
            "grad_col": "#D97706",
            "grad_ls": "-.",
            "grad_text": "⚠  g > 0 vanishes Jacobian (∂ã₀/∂a₀ = 0) + missing SoC bounds starve actor gradients during peaks",
            "ours": False,
        },
        {
            "y": 2.5,
            "h": 29.0,
            "banner": "Track C: T3 STE Differentiable Layer + CapShield Co-Design — Active Gradients + Exact Safety",
            "bg": "#F0FDFA",
            "border": PALETTE["PPO-Diff-Shield"],
            "head_col": PALETTE["PPO-Diff-Shield"],
            "blocks": [
                ("Actor (forward_raw)", ["Exact Gaussian log-prob on u", "No double-projection bias"], "#FFFFFF", PALETTE["PPO-Diff-Shield"], PALETTE["PPO-Diff-Shield"]),
                ("T3 DiffSafetyLayer", [f"SoC bounds [{soc_lo:.0f}%, {soc_hi:.0f}%] + 1-D proj", f"STE surrogate α = {ste_alpha:.1f} (∂ã/∂a > 0)"], "#CCFBF1", PALETTE["PPO-Diff-Shield"], PALETTE["PPO-Diff-Shield"]),
                (f"{n_shed_steps}-Step CapShield", [f"Discrete motor {{0}} ∪ [{motor_lo:.2f}, 1.00]", "1-ULP guard & blackout block"], "#FFFFFF", PALETTE["PPO-Diff-Shield"], "#0F172A"),
                ("Validated Outcome", [f"R1: €{float(ds_fin_m['cost_eur']) / 1000.0:.1f}k fin / €{float(ds_best_m['cost_eur']) / 1000.0:.1f}k best ({float(ds_fin_m['violation_rate_pct']):.2f}%)", f"R2: €{float(ci_ds_m['cost_eur']) / 1000.0:.1f}k ({float(ci_ds_m['violation_rate_pct']):.2f}%) vs 6D {float(ci_d6_m['violation_rate_pct']):.2f}% leak"], "#ECFDF5", PALETTE["PPO-Diff-Shield"], PALETTE["PPO-Diff-Shield"]),
            ],
            "grad_start": (37.0, 6.8),
            "grad_end": (12.5, 6.8),
            "grad_col": PALETTE["PPO-Diff-Shield"],
            "grad_ls": "-",
            "grad_text": f"✔  Active STE gradient (α={ste_alpha:.1f}) flows through binding caps → {float(ds_fin_m['active_soc_pct']):.1f}%–{float(ds_best_m['active_soc_pct']):.1f}% active SoC, ≤ G2",
            "ours": True,
        },
    ]

    col_geom = [(2.5, 20.5), (26.0, 22.0), (51.0, 20.5), (74.5, 23.0)]
    for tr in tracks:
        ty = tr["y"]
        th_val = tr["h"]
        outer_patch = FancyBboxPatch(
            (0.8, ty),
            98.4,
            th_val,
            boxstyle="round,pad=0.0,rounding_size=1.2",
            facecolor=tr["bg"],
            edgecolor=tr["border"],
            linewidth=1.8 if tr["ours"] else 1.2,
            zorder=1,
        )
        ax.add_patch(outer_patch)
        t_ban = ax.text(
            2.2, ty + th_val - 2.3, tr["banner"], ha="left", va="top",
            fontsize=9.4, fontweight="bold", color=tr["head_col"], zorder=4
        )
        texts_all.append(t_ban)

        box_y = ty + 8.6
        box_h = 13.8
        for (bx, bw), (b_title, b_lines, b_fc, b_ec, b_tc) in zip(col_geom, tr["blocks"]):
            bp, bt = _draw_diagram_box(
                ax,
                bx,
                box_y,
                bw,
                box_h,
                title=b_title,
                lines=b_lines,
                facecolor=b_fc,
                edgecolor=b_ec,
                title_color=b_tc,
                body_color="#1E293B",
                title_size=8.6,
                body_size=7.75,
                lw=1.3 if tr["ours"] else 1.0,
                rounding=0.8,
                zorder=3,
            )
            box_pairs.append((bp, bt))
            texts_all.extend(bt)

        y_mid = box_y + 0.5 * box_h
        for i in range(len(col_geom) - 1):
            x_from = col_geom[i][0] + col_geom[i][1]
            x_to = col_geom[i + 1][0]
            _draw_arrow(
                ax,
                (x_from, y_mid),
                (x_to, y_mid),
                color=tr["head_col"] if tr["ours"] else "#475569",
                lw=1.7,
                mutation=11.0,
            )

        _draw_arrow(
            ax,
            tr["grad_start"],
            tr["grad_end"],
            color=tr["grad_col"],
            lw=1.8 if tr["ours"] else 1.5,
            ls=tr["grad_ls"],
            mutation=11.0,
        )
        t_grad = ax.text(
            39.2,
            tr["grad_start"][1],
            tr["grad_text"],
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold" if tr["ours"] else "normal",
            color=tr["grad_col"],
            zorder=5,
        )
        texts_all.append(t_grad)

    add_slide_header(
        fig,
        "How T3 STE Differentiable Safety + CapShield Co-Design Solves Gradient Disconnects",
        (
            f"External shields park BESS at {sh_fb['active_soc_mean']:.1f}% SoC and v0 clipping zeroes active Jacobians; "
            f"STE + {n_shed_steps}-step CapShield cuts cost to €{float(ds_fin_m['cost_eur']) / 1000.0:.1f}k / "
            f"€{float(ds_best_m['cost_eur']) / 1000.0:.1f}k with 0.00% violations."
        ),
        "Source: agents/safety_layer.py, agents/ppo.py, env/cap_shield.py, per_seed_metrics.csv, and regime_benchmarks_val.csv.",
    )
    _assert_diagram_clean(fig, box_pairs, texts_all, "fig_11")
    return save_slide_fig(fig, _dest_dir(out_dir), "11_safety_mechanisms", save_svg=save_svg)


def fig_12_seed_stability_tradeoff(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Plot 5-seed reproducibility spread under 50:1 cost asymmetry and MPC forecast sensitivity."""
    fig = _new_slide()
    regime_rows = load_regime_benchmarks()
    per_seed = load_per_seed_table()
    baselines = load_baselines()
    constants = load_project_constants()
    _ = _val_total_hours(regime_rows, baselines)

    def _from_regime(suite: str, method: str, ckpt: str, label: str, col: str, is_ours: bool) -> Dict[str, Any]:
        srows = _find_regime_seed_rows(regime_rows, suite, method, ckpt)
        m_row = _find_regime_row(regime_rows, suite, method, "mean", ckpt)
        sd_row = _find_regime_row(regime_rows, suite, method, "std", ckpt)
        return {
            "label": label,
            "color": col,
            "mean": float(m_row["cost_eur"]),
            "std": float(sd_row["cost_eur"]),
            "seeds": [(int(r["seed"]), float(r["cost_eur"]), float(r["violation_rate_pct"]) > 0.0) for r in srows],
            "ours": is_ours,
        }

    lag_rows = per_seed["clamped_mb_ppo_lagrangian"]
    lag_costs = [float(r["cost_eur"]) for r in lag_rows]
    lag_mean, lag_std = _mean_std(lag_costs)
    lag_entry = {
        "label": _label("clamped_mb_ppo_lagrangian"),
        "color": _color("clamped_mb_ppo_lagrangian"),
        "mean": lag_mean,
        "std": lag_std,
        "seeds": [(101 + i, float(r["cost_eur"]), float(r["violation_hours"]) > 0.0) for i, r in enumerate(lag_rows)],
        "ours": False,
    }

    r1_entries = [
        _from_regime("regime1_val", "clamped_ppo_diff_shield", "best", f"{_label('clamped_ppo_diff_shield')} [Best]", PALETTE["PPO-Diff-Shield"], True),
        _from_regime("regime1_val", "clamped_ppo_diff_shield", "final", f"{_label('clamped_ppo_diff_shield')} [Final]", PALETTE["PPO-Diff-Shield"], True),
        _from_regime("regime1_val", "causal23_ppo_diff_shield", "best", f"{_label('causal23_ppo_diff_shield')} [Best]", PALETTE["Causal-Diff-Shield"], True),
        _from_regime("regime1_val", "causal23_ppo_diff_shield", "final", f"{_label('causal23_ppo_diff_shield')} [Final]", PALETTE["Causal-Diff-Shield"], True),
        lag_entry,
    ]
    r1_ys = [0.0, 1.0, 2.0, 3.0, 4.0]
    all_entries = list(zip(r1_ys, r1_entries))

    psm_r1 = float(baselines["PSM'"]["cost_eur"])
    g2_r1 = float(constants["g2_cost_eur"])

    mpc_modes = [
        ("MPCPolicy(perfect, H=24)", "Perfect\n(Oracle)"),
        ("MPCPolicy(causal_noisy, H=24)", "Causal Noisy\n(√(h/H) noise)"),
        ("MPCPolicy(persistence, H=24)", "Persistence\n(Naive Flat)"),
    ]
    mpc_rows = [_find_regime_row(regime_rows, "regime1_val", m, "single") for m, _ in mpc_modes]
    mpc_costs = np.array([float(r["cost_eur"]) for r in mpc_rows], dtype=float)
    mpc_unserv = np.array([float(r["unserved_penalty_eur"]) for r in mpc_rows], dtype=float)

    ds_best_mean = float(_find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "mean", "best")["cost_eur"])
    ds_best_std = float(_find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "std", "best")["cost_eur"])
    ds_best_uns_m = float(_find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "mean", "best")["unserved_penalty_eur"])
    ds_best_uns_s = float(_find_regime_row(regime_rows, "regime1_val", "clamped_ppo_diff_shield", "std", "best")["unserved_penalty_eur"])
    psm_r1_uns = float(baselines["PSM'"]["unserved_penalty_eur"])

    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.28, 1.0], wspace=0.25)
    ax_seed = fig.add_subplot(outer[0])
    inner_b = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], height_ratios=[1.22, 1.0], hspace=0.18)
    ax_mpc = fig.add_subplot(inner_b[0])
    ax_uns = fig.add_subplot(inner_b[1], sharex=ax_mpc)

    texts_all: List[Text] = []

    # Panel A: Per-Seed Spread
    style_clean_axes(ax_seed, grid_axis="x")
    ax_seed.axhspan(-0.55, 4.55, facecolor="#F8FAFC", alpha=0.65, zorder=0)
    ax_seed.axvline(psm_r1, color=PALETTE["PSM'"], ls=":", lw=1.6, zorder=2)
    ax_seed.axvline(g2_r1, color=PALETTE["Gate"], ls="--", lw=1.6, zorder=2)

    seed_markers = ["o", "s", "^", "D", "v"]
    seed_jitter = np.linspace(-0.18, 0.18, 5)
    all_x_vals = [c for _y, e in all_entries for _sid, c, _u in e["seeds"]]

    for y, e in all_entries:
        col = e["color"]
        any_unsafe = any(u for _sid, _c, u in e["seeds"])
        bar_col = PALETTE["Gate"] if any_unsafe else col
        if e["ours"]:
            ax_seed.axhspan(y - 0.36, y + 0.36, color=col, alpha=0.08, zorder=0)
        ax_seed.errorbar(e["mean"], y, xerr=e["std"], fmt="none", ecolor=bar_col, elinewidth=2.0, capsize=4.0, zorder=4)
        ax_seed.scatter([e["mean"]], [y], marker="D", color=bar_col, s=66 if e["ours"] else 54, edgecolors="#FFFFFF", lw=1.0, zorder=5)

        for idx_s, ((_sid, sc, is_unsafe), mk, dy) in enumerate(zip(e["seeds"], seed_markers, seed_jitter)):
            _ = idx_s
            sc_col = PALETTE["Gate"] if is_unsafe else col
            ax_seed.scatter([sc], [y + dy], marker=mk, color=sc_col, alpha=0.78, s=34, edgecolors="#FFFFFF", lw=0.6, zorder=4)

        x_right = max([e["mean"] + e["std"]] + [sc for _sid, sc, _u in e["seeds"]]) + 550.0
        tag = " [UNSAFE]" if any_unsafe else ""
        t_row = ax_seed.text(
            x_right,
            y,
            f"€{e['mean'] / 1000.0:.2f}k ± {e['std'] / 1000.0:.2f}k{tag}",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold" if (e["ours"] or any_unsafe) else "normal",
            color=PALETTE["Gate"] if any_unsafe else (col if e["ours"] else "#1E293B"),
            path_effects=_halo(),
            zorder=6,
        )
        texts_all.append(t_row)

    x_ax_lo = min(all_x_vals) - 1600.0
    x_ax_hi = max(all_x_vals) + 6500.0
    ax_seed.set_xlim(x_ax_lo, x_ax_hi)
    ax_seed.set_ylim(4.65, -0.95)
    ax_seed.set_yticks([y for y, _ in all_entries])
    ax_seed.set_yticklabels([e["label"] for _, e in all_entries], fontsize=9.3)
    for row_i, (_y, e) in enumerate(all_entries):
        if e["ours"]:
            ticks = ax_seed.get_yticklabels()
            ticks[row_i].set_fontweight("bold")
            ticks[row_i].set_color(e["color"])
    ax_seed.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"€{x / 1000.0:.0f}k"))
    n_seeds = len(r1_entries[0]["seeds"])
    ax_seed.set_xlabel(f"26-Week Validation Cost (€k, {n_seeds} Seeds 101–{100 + n_seeds} + Mean ± 1 SD)")
    ax_seed.set_title("A. Per-Seed Spread (Uniform 50:1 VoLL-to-Wear)", loc="left", pad=22, fontsize=11.5)

    t_psm1 = ax_seed.text(psm_r1 - 180.0, -0.58, "PSM'", ha="right", va="bottom", fontsize=8.2, fontweight="bold", color=PALETTE["PSM'"], path_effects=_halo())
    t_g2 = ax_seed.text(g2_r1 + 180.0, -0.58, "G2", ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=PALETTE["Gate"], path_effects=_halo())
    texts_all.extend([t_psm1, t_g2])

    seed_legend = [
        Line2D([0], [0], marker=mk, color="#475569", lw=0, markersize=5.2, label=str(101 + i))
        for i, mk in enumerate(seed_markers)
    ] + [Line2D([0], [0], marker="o", color=PALETTE["Gate"], lw=0, markersize=5.5, label="Unsafe (>0% viol)")]
    ax_seed.legend(
        handles=seed_legend, loc="lower left", bbox_to_anchor=(0.0, 1.005),
        ncol=6, fontsize=7.7, columnspacing=0.65, handletextpad=0.2, frameon=False
    )

    # Panel B Top: MPC Total Cost Sensitivity
    style_clean_axes(ax_mpc, grid_axis="both")
    xs = np.array([0.0, 1.0, 2.0])
    ax_mpc.axhspan(ds_best_mean - ds_best_std, ds_best_mean + ds_best_std, color=PALETTE["PPO-Diff-Shield"], alpha=0.13, zorder=1)
    ax_mpc.axhline(ds_best_mean, color=PALETTE["PPO-Diff-Shield"], ls="-", lw=1.8, zorder=2, label=f"Diff-Shield [Best] (€{ds_best_mean / 1000.0:.2f}k)")
    ax_mpc.axhline(psm_r1, color=PALETTE["PSM'"], ls=":", lw=1.6, zorder=2, label=f"PSM' (€{psm_r1 / 1000.0:.2f}k)")
    ax_mpc.plot(xs, mpc_costs, color=PALETTE["MPC"], lw=2.5, marker="s", markersize=7.0, markeredgecolor="#FFFFFF", markeredgewidth=1.0, zorder=4, label="MPC (H=24h)")

    y_b1_lo = min(mpc_costs.min(), ds_best_mean - ds_best_std) - 1200.0
    y_b1_hi = max(mpc_costs.max(), ds_best_mean + ds_best_std) + 1800.0
    ax_mpc.set_ylim(y_b1_lo, y_b1_hi)
    ax_mpc.set_xlim(-0.35, 2.42)
    ax_mpc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y / 1000.0:.0f}k"))
    ax_mpc.set_ylabel("Total Cost (€k)")
    ax_mpc.tick_params(labelbottom=False)
    ax_mpc.set_title("B. MPC Forecast Sensitivity vs. Model-Free RL", loc="left", pad=22, fontsize=11.5)
    ax_mpc.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, fontsize=7.6, frameon=False, columnspacing=0.7)

    for idx_m, (xv, cv) in enumerate(zip(xs, mpc_costs)):
        y_off = -750.0 if idx_m < 2 else 650.0
        va_dir = "top" if idx_m < 2 else "bottom"
        tm = ax_mpc.text(
            float(xv), float(cv) + y_off, f"€{cv / 1000.0:.2f}k",
            ha="center", va=va_dir, fontsize=8.4, fontweight="bold", color=PALETTE["MPC"],
            path_effects=_halo(), zorder=6
        )
        texts_all.append(tm)

    delta_mpc = float(mpc_costs[2] - mpc_costs[1])
    ax_mpc.annotate(
        "",
        xy=(1.96, float(mpc_costs[2]) - 150.0),
        xytext=(1.04, float(mpc_costs[1]) + 150.0),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["Gate"], lw=1.4, ls="--", shrinkA=2, shrinkB=2),
        zorder=5,
    )
    t_jump = ax_mpc.text(
        1.42,
        0.5 * (float(mpc_costs[1]) + float(mpc_costs[2])) + 550.0,
        f"+€{delta_mpc / 1000.0:.2f}k\nforecast penalty",
        ha="right",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=PALETTE["Gate"],
        path_effects=_halo(),
        zorder=6,
    )
    texts_all.append(t_jump)

    # Panel B Bottom: Unserved Penalty Sensitivity
    style_clean_axes(ax_uns, grid_axis="both")
    ax_uns.axhspan(ds_best_uns_m - ds_best_uns_s, ds_best_uns_m + ds_best_uns_s, color=PALETTE["PPO-Diff-Shield"], alpha=0.13, zorder=1)
    ax_uns.axhline(ds_best_uns_m, color=PALETTE["PPO-Diff-Shield"], ls="-", lw=1.7, zorder=2)
    ax_uns.axhline(psm_r1_uns, color=PALETTE["PSM'"], ls=":", lw=1.5, zorder=2)
    ax_uns.plot(xs, mpc_unserv, color=PALETTE["Unserved"], lw=2.5, marker="o", markersize=7.0, markeredgecolor="#FFFFFF", markeredgewidth=1.0, zorder=4)

    y_b2_lo = min(mpc_unserv.min(), ds_best_uns_m - ds_best_uns_s) - 1100.0
    y_b2_hi = max(mpc_unserv.max(), ds_best_uns_m + ds_best_uns_s, psm_r1_uns) + 1700.0
    ax_uns.set_ylim(y_b2_lo, y_b2_hi)
    ax_uns.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"€{y / 1000.0:.0f}k"))
    ax_uns.set_ylabel("Unserved Penalty (€k)")
    ax_uns.set_xticks(xs)
    ax_uns.set_xticklabels([lbl for _m, lbl in mpc_modes], fontsize=8.8)
    ax_uns.set_xlabel("Forecast Quality (24h MPC Lookahead)")

    for idx_u, (xv, uv) in enumerate(zip(xs, mpc_unserv)):
        y_off = -650.0 if idx_u < 2 else 600.0
        va_dir = "top" if idx_u < 2 else "bottom"
        tu = ax_uns.text(
            float(xv), float(uv) + y_off, f"€{uv / 1000.0:.2f}k",
            ha="center", va=va_dir, fontsize=8.3, fontweight="bold", color="#B45309",
            path_effects=_halo(), zorder=6
        )
        texts_all.append(tu)

    t_ds_u = ax_uns.text(0.02, ds_best_uns_m - 220.0, f"Diff-Shield: €{ds_best_uns_m / 1000.0:.2f}k", ha="left", va="top", fontsize=7.7, fontweight="bold", color=PALETTE["PPO-Diff-Shield"], path_effects=_halo(), zorder=6)
    t_psm_u = ax_uns.text(2.34, psm_r1_uns + 240.0, f"PSM': €{psm_r1_uns / 1000.0:.2f}k", ha="right", va="bottom", fontsize=7.7, fontweight="bold", color=PALETTE["PSM'"], path_effects=_halo(), zorder=6)
    texts_all.extend([t_ds_u, t_psm_u])

    _ = r1_entries[0]
    add_slide_header(
        fig,
        f"Consistent {n_seeds}-Seed Stability & Why Naive Forecasts Hurt MPC (+€{delta_mpc / 1000.0:.2f}k) vs. Model-Free RL",
        "",
    )
    fig.subplots_adjust(left=0.195, right=0.975, top=0.80, bottom=0.13)
    _assert_no_overlaps(fig, texts_all, "fig_12")
    return save_slide_fig(fig, _dest_dir(out_dir), "12_seed_stability_tradeoff", save_svg=save_svg)


def plot_presentation_all(
    out_dir: Optional[str] = None,
    save_svg: bool = False,
) -> List[str]:
    """Generate all 12 presentation slide figures into ``out_dir`` (PNG by default)."""
    dest = _dest_dir(out_dir)
    os.makedirs(dest, exist_ok=True)
    builders = (
        fig_01_ablation_pathologies,
        fig_02_pareto_cost_safety,
        fig_03_learning_dynamics,
        fig_04_regime1_causal_mpc,
        fig_05_regime2_ci_ablation,
        fig_06_ood_stress_2019w04,
        fig_07_dispatch_case_2019w04,
        fig_08_seasonal_test_context,
        fig_09_microgrid_architecture,
        fig_10_dataset_split_regimes,
        fig_11_safety_mechanisms,
        fig_12_seed_stability_tradeoff,
    )
    saved: List[str] = []
    for builder in builders:
        paths = builder(dest, save_svg=save_svg)
        saved.extend(paths)
    return saved


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint supporting ``--out-dir`` and ``--save-svg``."""
    parser = argparse.ArgumentParser(description="Generate 12-figure 16:9 presentation slide suite.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Destination directory for presentation figures (default: results/figures/presentation).",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Also emit SVG files alongside 250 DPI PNGs.",
    )
    args = parser.parse_args(argv)
    paths = plot_presentation_all(out_dir=args.out_dir, save_svg=args.save_svg)
    print(f"Generated {len(paths)} presentation figure files in {_dest_dir(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
