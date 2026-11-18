"""Single source of truth for presentation slide and report styling.

Provides 16:9 slide geometry, an Okabe-Ito-derived colorblind-safe palette
(highlighting Ours in saturated teal #0F766E), concise presentation method
labels, clean Matplotlib styling helpers, and deterministic 1-D label
collision avoidance / overlap verification utilities.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text

SLIDE_FIGSIZE: Tuple[float, float] = (13.333, 7.5)
SAVE_DPI: int = 250
SLIDE_FACECOLOR: str = "#FFFFFF"

PALETTE: Dict[str, str] = {
    # Primary / Ours
    "PPO-Diff-Shield": "#0F766E",
    "Causal-Diff-Shield": "#0891B2",
    "Diff-Shield-6D": "#B45309",
    # RL variants
    "PPO": "#0072B2",
    "PPO_FB": "#2B8CBE",
    "PPO_UNC": "#56B4E9",
    "PPO-Lagrangian": "#D55E00",
    "PPO-Lagrangian_FB": "#E67E22",
    "PPO-Lagrangian_UNC": "#E69F00",
    "PPO-Residual": "#65A30D",
    "Shielded PPO": "#CC79A7",
    "Shielded PPO_MB": "#884969",
    "PPO-Aux-Shield": "#7C3AED",
    # Baselines & MPC
    "Grid-Only": "#64748B",
    "PSM'": "#334155",
    "LP'": "#4D7C0F",
    "MILP'": "#3F6212",
    "Shield + Idle": "#94A3B8",
    "Shield + PSM'": "#475569",
    "MPC": "#2563EB",
    "MPC-Perfect": "#1E3A8A",
    "MPC-Persistence": "#60A5FA",
    # Gates & cost decomposition
    "Gate": "#C0392B",
    "Feasible": "#22C55E",
    "Energy": "#0072B2",
    "Wear": "#64748B",
    "Unserved": "#E69F00",
    "DemandCharge": "#7C3AED",
    "Trip": "#C0392B",
}

METHOD_LABELS: Dict[str, str] = {
    "Grid-Only": "Grid-Only",
    "PSM'": "PSM'",
    "LP'": "LP' Oracle",
    "MILP'": "MILP' Oracle",
    "Shield + Idle": "Shield + Idle",
    "Shield + PSM'": "Shield + PSM'",
    "MPCPolicy(perfect, H=24)": "MPC (Perfect, 24h)",
    "MPCPolicy(causal_noisy, H=24)": "MPC (Causal, 24h)",
    "MPCPolicy(persistence, H=24)": "MPC (Persist., 24h)",
    "MPC-Perfect": "MPC (Perfect, 24h)",
    "MPC": "MPC (Causal, 24h)",
    "MPC-Persistence": "MPC (Persist., 24h)",
    "s3_ppo": "PPO (Unclamped)",
    "unclamped_ppo": "PPO (Unclamped)",
    "clamped_ppo": "PPO (Full-Batch)",
    "clamped_mb_ppo": "PPO (Minibatch)",
    "s3_ppo_lag": "PPO-Lag (Unclamped)",
    "unclamped_ppo_lagrangian": "PPO-Lag (Unclamped)",
    "clamped_ppo_lagrangian": "PPO-Lag (Full-Batch)",
    "clamped_mb_ppo_lagrangian": "PPO-Lag (Minibatch)",
    "s3_ppo_res": "PPO-Residual (Unclamped)",
    "unclamped_ppo_residual": "PPO-Residual (Unclamped)",
    "clamped_ppo_residual": "PPO-Residual (Full-Batch)",
    "clamped_mb_ppo_residual": "PPO-Residual (Minibatch)",
    "s3_sh_ppo": "Shielded PPO (Unclamped)",
    "unclamped_ppo_shielded": "Shielded PPO (Unclamped)",
    "clamped_ppo_shielded": "Shielded PPO (Full-Batch)",
    "clamped_mb_ppo_shielded": "Shielded PPO (Minibatch)",
    "clamped_ppo_shield_aux": "PPO-Aux-Shield",
    "clamped_ppo_diff_shield": "PPO-Diff-Shield",
    "causal23_ppo_shielded": "Shielded PPO (Causal 23D)",
    "causal23_ppo_residual": "PPO-Residual (Causal 23D)",
    "causal23_ppo_diff_shield": "Causal-Diff-Shield",
    "ci_ppo_shielded": "Shielded PPO",
    "ci_ppo_diff_shield": "PPO-Diff-Shield",
    "ci_ppo_diff_6d": "Diff-Shield-6D (Abl.)",
    "PPO-Diff-Shield": "PPO-Diff-Shield",
    "Causal-Diff-Shield": "Causal-Diff-Shield",
    "Diff-Shield-6D": "Diff-Shield-6D (Abl.)",
    "PPO-Aux-Shield": "PPO-Aux-Shield",
    "Shielded PPO": "Shielded PPO",
    "PPO-Residual": "PPO-Residual",
    "PPO-Lagrangian": "PPO-Lagrangian",
    "PPO": "Vanilla PPO",
}


def apply_presentation_rc() -> None:
    """Apply clean, high-contrast Matplotlib rcParams for 16:9 slides."""
    plt.rcParams.update({
        "figure.facecolor": SLIDE_FACECOLOR,
        "axes.facecolor": SLIDE_FACECOLOR,
        "savefig.facecolor": SLIDE_FACECOLOR,
        "figure.dpi": 140,
        "savefig.dpi": SAVE_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 11.5,
        "text.color": "#0F172A",
        "axes.labelsize": 12.0,
        "axes.labelweight": "medium",
        "axes.labelcolor": "#1E293B",
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#0F172A",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.alpha": 0.85,
        "grid.linewidth": 0.75,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#E2E8F0",
        "legend.fontsize": 10.5,
        "svg.fonttype": "none",
    })


def style_clean_axes(ax: Axes, grid_axis: Optional[str] = "x") -> Axes:
    """Remove top/right spines and apply subtle gridlines along ``grid_axis``."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid_axis in ("x", "y", "both"):
        ax.grid(
            True,
            axis=grid_axis,
            color="#E2E8F0",
            linewidth=0.75,
            alpha=0.85,
            linestyle="-",
        )
    ax.tick_params(axis="both", which="both", length=3.5, color="#94A3B8", labelcolor="#334155")
    return ax


def add_slide_header(
    fig: Figure,
    title: str,
    subtitle: str,
    source_note: Optional[str] = None,
) -> Tuple[Text, Optional[Text], Optional[Text]]:
    """Add left-aligned slide title and optional subtitle (source notes suppressed)."""
    del source_note
    title_art = fig.text(
        0.05,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=17.5,
        fontweight="bold",
        color="#0F172A",
    )
    subtitle_art: Optional[Text] = None
    if subtitle:
        subtitle_art = fig.text(
            0.05,
            0.91,
            subtitle,
            ha="left",
            va="top",
            fontsize=11.5,
            color="#475569",
        )
    return title_art, subtitle_art, None


def save_slide_fig(
    fig: Figure,
    out_dir: str,
    stem: str,
    save_svg: bool = False,
) -> List[str]:
    """Save figure as high-DPI PNG (and optional SVG) on a white slide canvas."""
    os.makedirs(out_dir, exist_ok=True)
    clean_stem = stem
    for ext in (".png", ".svg"):
        if clean_stem.endswith(ext):
            clean_stem = clean_stem[: -len(ext)]

    saved: List[str] = []
    exts = ("png", "svg") if save_svg else ("png",)
    for ext in exts:
        path = os.path.join(out_dir, f"{clean_stem}.{ext}")
        fig.savefig(
            path,
            dpi=SAVE_DPI,
            facecolor=SLIDE_FACECOLOR,
            bbox_inches="tight",
            pad_inches=0.06,
        )
        saved.append(path)
    plt.close(fig)
    return saved


def repel_y_positions(
    ys: Sequence[float],
    min_sep: float,
    y_min: float,
    y_max: float,
) -> List[float]:
    """Deterministically relax 1-D y-coordinates to enforce >= min_sep inside [y_min, y_max].

    Preserves the sorted value order (breaking ties by original index) and returns
    displaced y-coordinates aligned with the input sequence order. When
    ``(len(ys) - 1) * min_sep > (y_max - y_min)``, falls back to maximum feasible
    uniform spacing inside ``[y_min, y_max]``.
    """
    n = len(ys)
    if n == 0:
        return []
    y_lo = float(y_min)
    y_hi = float(y_max)
    sep_req = float(min_sep)
    if y_lo > y_hi:
        raise ValueError(f"y_min ({y_lo}) must be <= y_max ({y_hi})")
    if sep_req < 0.0:
        raise ValueError(f"min_sep ({sep_req}) must be >= 0")

    if n == 1:
        return [min(max(float(ys[0]), y_lo), y_hi)]

    span = y_hi - y_lo
    feasible = (n - 1) * sep_req <= span
    order = sorted(range(n), key=lambda idx: (float(ys[idx]), idx))

    if not feasible:
        pos = [y_lo + k * span / (n - 1) for k in range(n)]
        pos[0] = y_lo
        pos[-1] = y_hi
        pos = [min(max(v, y_lo), y_hi) for v in pos]
    else:
        pos = [min(max(float(ys[idx]), y_lo), y_hi) for idx in order]
        if sep_req > 0.0:
            for _ in range(300):
                max_move = 0.0
                for k in range(n - 1):
                    gap = pos[k + 1] - pos[k]
                    if gap < sep_req:
                        half = 0.5 * (sep_req - gap)
                        pos[k] = max(pos[k] - half, y_lo)
                        pos[k + 1] = min(pos[k + 1] + half, y_hi)
                        if half > max_move:
                            max_move = half
                if max_move < 1e-11:
                    break

            has_slack = (n - 1) * sep_req < span * 0.999999999999
            pos[0] = max(pos[0], y_lo)
            for k in range(1, n):
                target = pos[k - 1] + sep_req
                if has_slack and (target - pos[k - 1]) < sep_req and target < y_hi:
                    target = math.nextafter(target, math.inf)
                if pos[k] < target:
                    pos[k] = min(target, y_hi)

            pos[-1] = min(pos[-1], y_hi)
            for k in range(n - 2, -1, -1):
                limit = pos[k + 1] - sep_req
                if has_slack and (pos[k + 1] - limit) < sep_req and limit > y_lo:
                    limit = math.nextafter(limit, -math.inf)
                limit = max(limit, y_lo)
                if pos[k] > limit:
                    pos[k] = limit

            pos = [min(max(v, y_lo), y_hi) for v in pos]

    out = [0.0] * n
    for idx, val in zip(order, pos):
        out[idx] = val
    return out


def check_axes_text_overlaps(
    fig: Figure,
    texts: Iterable[Text],
) -> List[Tuple[Text, Text]]:
    """Render ``fig`` and return all pairs of visible Text objects whose bboxes overlap."""
    text_list = [t for t in texts if t.get_visible() and t.get_text().strip()]
    if len(text_list) < 2:
        return []

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer=renderer) for t in text_list]

    overlaps: List[Tuple[Text, Text]] = []
    for i in range(len(text_list)):
        for j in range(i + 1, len(text_list)):
            if bboxes[i].overlaps(bboxes[j]):
                overlaps.append((text_list[i], text_list[j]))
    return overlaps
