import io
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES = os.path.join(ROOT, "results", "tables")
RUNS = os.path.join(ROOT, "results", "runs")
FIGS = os.path.join(ROOT, "results", "figures")


def load_project_constants():
    """Load evaluation gates and physical constants from results/tables/project_constants.csv."""
    import csv
    constants_path = os.path.join(TABLES, "project_constants.csv")
    constants = {}
    if os.path.exists(constants_path):
        with open(constants_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                constants[row["name"]] = float(row["value"])
    exceedance_gate_kw = constants.get("d_main_kw_per_step", 0.05)
    cost_gate_eur = constants.get("g2_cost_eur", 58306.73)
    return exceedance_gate_kw, cost_gate_eur


# Experiment groups: (label, agent folder under results/runs/, colour)
GROUPS = [
    ("PPO", "ppo", "#0072B2"),
    ("PPO-Lagrangian", "ppo_lagrangian", "#D55E00"),
    ("PPO-Residual", "ppo_residual", "#009E73"),
    ("Shielded PPO", "ppo_shielded", "#CC79A7"),
]
COLOURS = dict((g[0], g[2]) for g in GROUPS)

G1_EXC_KW, G2_COST_EUR = load_project_constants()
FLOOR = 1e-4             # floor used when a log axis has to show a zero

GATE = "#C0392B"         # gate lines
BASE = "#5A5A5A"         # frozen-baseline reference lines

COST_YLIM = (43500, 61000)
STEP_TICKS = [50, 100, 150, 200, 250, 300]
EXC_TICKS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.8,
    "figure.constrained_layout.use": False,
    "axes.prop_cycle": cycler(color=[g[2] for g in GROUPS]),
})


def fmt_num(v, _pos=None):
    """Plain number for a log axis: 0.0001, 0.001, 0.01, 0.1, 1, 10."""
    return "%g" % v


def fmt_k(v, _pos=None):
    """Thousands for a cost axis: 45k, 50k, 55k."""
    return "%dk" % round(v / 1000.0)


def log_axis(ax, which, ticks, formatter=fmt_num):
    """Give a log axis explicit, in-range ticks and readable labels."""
    target = ax.xaxis if which == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(formatter))
    target.set_minor_locator(FixedLocator([]))
    target.set_minor_formatter(NullFormatter())


# Loading helpers
def group_of(agent):
    """Map an agent folder name to its experiment group."""
    for label, folder, _colour in GROUPS:
        if agent == folder:
            return label
    return None


def load_runs(family: str = "unclamped"):
    """Load runs filtered by experiment family."""
    out = dict((g[0], []) for g in GROUPS)
    target_clamped = family != "unclamped"
    target_forecast = family in ("clamped_forecast", "forecast")
    if target_forecast:
        prefix_filter = ("clamped_forecast_",)
    elif family in ("clamped_mb", "mb"):
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
            has_forecast = int(rc.get("forecast_horizon_hours", 0)) > 0
            if is_clamped == target_clamped and has_forecast == target_forecast:
                out[label].append(run_data)
    return out


def load_baselines():
    """Load baseline metrics from results/tables/frozen_baselines.csv."""
    import csv
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
                "cost_eur": row.get("cost_eur") or row.get("val_cost_eur"),
                "exceedance_kw": row.get("exceedance_kw") or row.get("mean_exceedance_kw"),
            }
    return out


def final_metrics(run):
    """The metrics dict of the run's final checkpoint."""
    return run.get("final_checkpoint", {}).get("metrics", {})


def eval_curve(runs):
    """(steps, per-seed cost series, per-seed exceedance series) for one group."""
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
    """Column-wise mean and sample sd of a list of equal-length series."""
    n = len(rows[0])
    mean = [statistics.fmean([r[i] for r in rows]) for i in range(n)]
    sd = [statistics.stdev([r[i] for r in rows]) if len(rows) > 1 else 0.0
          for i in range(n)]
    return mean, sd


def group_handle(label, extra=""):
    return Line2D([], [], marker="o", ls="", color=COLOURS[label], markersize=8,
                  label=label + extra)


# Figure generators
def fig_learning_curves(runs, baselines):
    """Fig 1 - Val cost and Val exceedance over training, mean +/- sd of 5 seeds."""
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
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
        ax.plot(x, mean, color=colour, marker="o", markersize=4.5, label=label,
                zorder=3)
        ax.fill_between(x, [m - s for m, s in zip(mean, sd)],
                        upper, color=colour, alpha=0.18, linewidth=0)
    ax.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.4, zorder=2,
               label="G2 gate  %.0f EUR" % G2_COST_EUR)
    for name, style, col in (("Grid-Only", "--", BASE), ("PSM'", ":", BASE), ("LP'", "-.", "#196F3D")):
        if name in baselines:
            ax.axhline(float(baselines[name]["cost_eur"]), ls=style, color=col,
                       lw=1.3, zorder=2)

    ymax = 61500 if max_cost_upper <= 60500 else int(math.ceil((max_cost_upper + 2500) / 5000.0) * 5000)
    tick_step = 5000 if ymax <= 70000 else 10000
    ax.set_ylim(COST_YLIM[0], ymax)
    ax.yaxis.set_major_locator(FixedLocator(list(range(45000, ymax + 1, tick_step))))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.set_ylabel("Val cost (EUR)")
    ax.set_title("A. Validation Cost over Training")
    ax.margins(x=0.02)

    ax = axes[1]
    for label, _prefix, colour in GROUPS:
        steps, _costs, excs = eval_curve(runs[label])
        if not steps:
            continue
        mean, sd = mean_sd(excs)
        x = [s / 1000 for s in steps]
        y = [max(v, FLOOR) for v in mean]
        lo = [max(v - s, FLOOR) for v, s in zip(mean, sd)]
        hi = [max(v + s, FLOOR) for v, s in zip(mean, sd)]
        ax.plot(x, y, color=colour, marker="o", markersize=4.5, label=label,
                zorder=3)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0)
    ax.axhline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2,
               label="G1 gate  %.2f kW/step" % G1_EXC_KW)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 30)
    log_axis(ax, "y", EXC_TICKS)
    ax.set_ylabel("Mean exceedance above cap\n(kW/step, log scale)")
    ax.set_xlabel("Training steps (thousands, evaluated every 50k)")
    ax.set_title("B. Mean Grid Exceedance over Training")
    ax.set_xticks(STEP_TICKS)
    ax.margins(x=0.02)

    handles = [group_handle(g[0]) for g in GROUPS]
    handles += [
        Line2D([], [], ls="--", color=GATE, lw=1.4,
               label="G2 gate  %.0f EUR" % G2_COST_EUR),
        Line2D([], [], ls="--", color=GATE, lw=1.5,
               label="G1 gate  %.2f kW/step" % G1_EXC_KW),
    ]
    for name, style, col in (("Grid-Only", "--", BASE), ("PSM'", ":", BASE), ("LP'", "-.", "#196F3D")):
        if name in baselines:
            handles.append(Line2D([], [], ls=style, color=col, lw=1.4,
                                  label="%s (frozen) %.0f EUR"
                                        % (name,
                                           float(baselines[name]["cost_eur"]))))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.20, hspace=0.20)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=True)
    fig.savefig(os.path.join(FIGS, "learning_curves.png"), bbox_inches="tight")
    plt.close(fig)


def fig_cost_vs_exceedance(runs, baselines):
    """Fig 2 - the cost / exceedance frontier, one marker per (method, seed)."""
    fig, ax = plt.subplots(figsize=(9.8, 6.8))

    max_y = G2_COST_EUR
    handles = []
    for label, _prefix, colour in GROUPS:
        xs, ys = [], []
        for r in runs[label]:
            m = final_metrics(r)
            xs.append(max(float(m["mean_exceedance_kw"]), FLOOR))
            ys.append(float(m["total_cost_eur"]))
        if not xs:
            continue
        max_y = max(max_y, max(ys))
        ax.scatter(xs, ys, s=80, color=colour, edgecolor="white", linewidth=1.1,
                   zorder=4)
        handles.append(group_handle(label, "  (5 seeds)"))

    for name, marker, colour in (("Grid-Only", "s", "#444444"),
                                 ("PSM'", "D", "#1A5276"),
                                 ("LP'", "*", "#196F3D"),
                                 ("PS'", "^", "#7D3C98")):
        if name not in baselines:
            continue
        b = baselines[name]
        by = float(b["cost_eur"])
        max_y = max(max_y, by)
        ax.scatter([max(float(b["exceedance_kw"]), FLOOR)],
                   [by], marker=marker, s=175 if marker != "*" else 220, color=colour,
                   edgecolor="white", linewidth=1.2, zorder=5)
        handles.append(Line2D([], [], marker=marker, ls="", color=colour,
                              markersize=9 if marker != "*" else 11,
                              label="%s (frozen)  %.0f EUR, %.2f kW"
                                    % (name, by, float(b["exceedance_kw"]))))

    ymax = 62000 if max_y <= 60500 else int(math.ceil((max_y + 2500) / 5000.0) * 5000)
    tick_step = 5000 if ymax <= 75000 else 10000

    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.65, 12)
    ax.set_ylim(COST_YLIM[0], ymax)
    log_axis(ax, "x", EXC_TICKS)
    ax.yaxis.set_major_locator(FixedLocator(list(range(45000, ymax + 1, tick_step))))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))

    # Feasible region: below G2 cost and left of G1 exceedance
    ax.add_patch(plt.Rectangle((FLOOR * 0.65, COST_YLIM[0]),
                                G1_EXC_KW - FLOOR * 0.65,
                                G2_COST_EUR - COST_YLIM[0],
                                color="#009E73", alpha=0.09, zorder=0))
    ax.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2)
    ax.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2)
    handles += [
        Patch(facecolor="#009E73", alpha=0.15, label="Feasible Region (G1 & G2)"),
        Line2D([], [], ls="--", color=GATE, lw=1.5,
               label="G2 gate  %.0f EUR" % G2_COST_EUR),
        Line2D([], [], ls="--", color=GATE, lw=1.5,
               label="G1 gate  %.2f kW/step" % G1_EXC_KW),
    ]

    ax.set_xlabel("Mean exceedance above cap (kW/step, log scale; 0 floored at %g)"
                  % FLOOR)
    ax.set_ylabel("Val cost (EUR)")
    ax.set_title("Validation Cost vs. Mean Grid Exceedance")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.22)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=True)
    fig.savefig(os.path.join(FIGS, "cost_vs_exceedance.png"), bbox_inches="tight")
    plt.close(fig)


def fig_lambda(runs):
    """Fig 3 - lambda multiplier over training for the two constrained variants."""
    fig, ax = plt.subplots(figsize=(9.8, 6.2))

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
                    color=colour, alpha=0.55, lw=1.1, zorder=2)
            ends.append(float(traj[-1]["lambda"]))
        if ends:
            handles.append(Line2D([], [], color=colour, lw=2.4,
                                  label="%s   lambda at 300k: %s"
                                        % (label,
                                           ", ".join("%.2f" % e
                                                     for e in sorted(ends)))))

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 100)
    ax.set_xlim(-5, 305)
    log_axis(ax, "y", [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    ax.set_xticks(STEP_TICKS)
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Lagrange multiplier lambda (log scale)")
    ax.set_title("Lagrange Multiplier Trajectory over Training")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.16)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=1, frameon=True)
    fig.savefig(os.path.join(FIGS, "lambda_trajectory.png"), bbox_inches="tight")
    plt.close(fig)


def fig_per_seed(runs, baselines):
    """Fig 4 - per-seed cost and exceedance, with the two gates."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.12})

    labels = [g[0] for g in GROUPS]
    ypos = list(range(len(labels)))[::-1]

    ax = axes[0]
    max_cost = G2_COST_EUR
    for y, label in zip(ypos, labels):
        vals = sorted(float(final_metrics(r)["total_cost_eur"])
                      for r in runs[label])
        if not vals:
            continue
        max_cost = max(max_cost, max(vals))
        ax.scatter(vals, [y] * len(vals), s=85, color="#0072B2", zorder=4,
                   edgecolor="white", linewidth=1.1)
        ax.scatter([sum(vals) / len(vals)], [y], marker="D", s=95,
                   color="#1A5276", zorder=5)
    ax.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2)
    if "Grid-Only" in baselines:
        ax.axvline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE,
                   lw=1.3, zorder=2)
    if "PSM'" in baselines:
        ax.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=BASE,
                   lw=1.3, zorder=2)
    if "LP'" in baselines:
        ax.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color="#196F3D",
                   lw=1.3, zorder=2)

    xmax = 61000 if max_cost <= 60000 else int(math.ceil((max_cost + 2500) / 5000.0) * 5000)
    tick_step = 5000 if xmax <= 70000 else 10000
    ax.set_xlim(44000, xmax)
    ax.xaxis.set_major_locator(FixedLocator(list(range(45000, xmax + 1, tick_step))))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Val cost (EUR)")
    ax.set_title("A. Validation Cost per Seed")
    ax.margins(y=0.20)

    ax = axes[1]
    for y, label in zip(ypos, labels):
        vals = sorted(max(float(final_metrics(r)["mean_exceedance_kw"]), FLOOR)
                      for r in runs[label])
        if not vals:
            continue
        ax.scatter(vals, [y] * len(vals), s=85, color="#009E73", zorder=4,
                   edgecolor="white", linewidth=1.1)
        ax.scatter([sum(vals) / len(vals)], [y], marker="D", s=95,
                   color="#145A32", zorder=5)
    ax.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.65, 12)
    log_axis(ax, "x", EXC_TICKS)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean exceedance above cap (kW/step, log scale; 0 floored)")
    ax.set_title("B. Mean Grid Exceedance per Seed")
    ax.margins(y=0.20)

    handles = [
        Line2D([], [], marker="o", ls="", color="#5A5A5A", markersize=8,
               label="one seed"),
        Line2D([], [], marker="D", ls="", color="#5A5A5A", markersize=8,
               label="mean of 5 seeds"),
        Line2D([], [], ls="--", color=GATE, lw=1.5,
               label="G2 gate  %.0f EUR" % G2_COST_EUR),
        Line2D([], [], ls="--", color=GATE, lw=1.5,
               label="G1 gate  %.2f kW/step" % G1_EXC_KW),
    ]
    if "Grid-Only" in baselines:
        handles.append(Line2D([], [], ls="--", color=BASE, lw=1.3,
                              label="Grid-Only  %.0f"
                                    % float(baselines["Grid-Only"]["cost_eur"])))
    if "PSM'" in baselines:
        handles.append(Line2D([], [], ls=":", color=BASE, lw=1.3,
                              label="PSM'  %.0f"
                                    % float(baselines["PSM'"]["cost_eur"])))
    if "LP'" in baselines:
        handles.append(Line2D([], [], ls="-.", color="#196F3D", lw=1.3,
                              label="LP'  %.0f"
                                    % float(baselines["LP'"]["cost_eur"])))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.18)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True)
    fig.savefig(os.path.join(FIGS, "per_seed_results.png"), bbox_inches="tight")
    plt.close(fig)


def fig_cost_breakdown(runs, baselines):
    """Fig 5 - where the money goes: energy purchase vs unserved penalty vs wear."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.08})
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    parts = ("energy_purchase_eur", "battery_wear_eur", "unserved_penalty_eur")
    part_names = ("Grid energy bought", "Battery wear", "Unserved penalty")
    part_colours = ("#0072B2", "#7F8C8D", "#E67E22")

    ypos = list(range(len(GROUPS)))[::-1]
    labels = [g[0] for g in GROUPS]
    bar_h = 0.55

    all_tots = []
    all_excs = []
    all_tps = []
    group_data = []

    for y, (label, _prefix, colour) in zip(ypos, GROUPS):
        totals = []
        for i, part in enumerate(parts):
            vals = [float(final_metrics(r).get(part, 0.0)) for r in runs[label]]
            v = sum(vals) / len(vals) if vals else 0.0
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
            ax1.barh(y, v, left=left, height=bar_h, color=part_colours[i],
                     edgecolor="white", linewidth=0.8)
            left += v

        # Panel B: Exceedance bar
        ax2.barh(y, mean_exc, height=bar_h, color=colour, edgecolor="white", linewidth=0.8)

        # Panel C: Battery Throughput bar
        ax3.barh(y, mean_tp, xerr=std_tp, height=bar_h, color=colour, edgecolor="white", linewidth=0.8, capsize=2)

    # Panel A limits and decor
    ax1.set_xlim(0, max_tot * 1.06)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax1.set_xlabel("Validation Cost (EUR, mean over 5 seeds)")
    ax1.set_title("A. Cost Breakdown vs. Grid-Only", fontsize=11, fontweight="bold")
    ax1.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.3, zorder=2)
    if "Grid-Only" in baselines:
        ax1.axvline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.2, zorder=2)
    if "PSM'" in baselines:
        ax1.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=BASE, lw=1.2, zorder=2)
    if "LP'" in baselines:
        ax1.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color="#196F3D", lw=1.2, zorder=2)

    # Panel B limits and decor
    ax2.set_xlim(0, max_exc * 1.08)
    ax2.axvline(0.0, ls=":", lw=1.8, color="#1A5276", zorder=2)
    ax2.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.3, zorder=2)
    if "Grid-Only" in baselines:
        ax2.axvline(float(baselines["Grid-Only"]["exceedance_kw"]), ls="--", color=BASE, lw=1.2, zorder=2)
    ax2.set_xlabel("Mean Exceedance (kW/step)")
    ax2.set_title("B. Exceedance vs. PSM' & Grid-Only", fontsize=11, fontweight="bold")

    # Panel C limits and decor
    ax3.set_xlim(0, max_tp * 1.08)
    ax3.axvline(5.84, ls=":", color=BASE, lw=1.4, zorder=2)
    ax3.axvline(24.99, ls="-.", color="#196F3D", lw=1.4, zorder=2)
    ax3.set_xlabel("Battery Throughput (MWh, 26 Val Weeks)")
    ax3.set_title("C. Battery Usage Across Methods", fontsize=11, fontweight="bold")

    # Handles
    cost_handles = [Patch(facecolor=part_colours[i], label=part_names[i]) for i in range(len(parts))]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.3, label="G1 / G2 Gates"),
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only"),
        Line2D([], [], ls=":", color=BASE, lw=1.2, label="PSM' Target"),
    ]
    if "LP'" in baselines:
        ref_handles.append(Line2D([], [], ls="-.", color="#196F3D", lw=1.2, label="LP' Oracle"))

    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
    fig.legend(handles=cost_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True)
    fig.savefig(os.path.join(FIGS, "cost_breakdown.png"), bbox_inches="tight")
    plt.close(fig)


# Cross-family comparison figures
def fig_unified_pareto(all_families, baselines, out_dir):
    """Unified Pareto scatter plot comparing Cost vs Exceedance across environments."""
    fig, ax = plt.subplots(figsize=(11.0, 7.5))

    fam_styles = [
        ("unclamped", "Unclamped", "x", 7.0),
        ("clamped", "Clamped", "o", 6.5),
        ("clamped_mb", "Clamped Minibatch", "^", 7.5),
        ("clamped_forecast", "Clamped + Forecast", "*", 9.5),
    ]

    # Shaded feasible quadrant (x <= G1, y <= G2)
    ax.fill_between([FLOOR * 0.65, G1_EXC_KW], [42000, 42000], [G2_COST_EUR, G2_COST_EUR],
                    color="#2ECC71", alpha=0.10, zorder=1)

    max_cost = G2_COST_EUR
    for label, _prefix, colour in GROUPS:
        for fam_key, _fam_name, marker, msize in fam_styles:
            runs = all_families.get(fam_key, {}).get(label, [])
            if not runs:
                continue
            costs = [float(final_metrics(r).get("total_cost_eur", 0.0)) for r in runs]
            excs = [float(final_metrics(r).get("mean_exceedance_kw", 0.0)) for r in runs]
            if costs:
                max_cost = max(max_cost, max(costs))

            x_seeds = [max(e, FLOOR) for e in excs]
            ax.scatter(x_seeds, costs, marker=marker, color=colour, alpha=0.45,
                       s=msize ** 2 * 0.5, zorder=3)

            mean_cost = statistics.fmean(costs)
            mean_exc = statistics.fmean(excs)
            if marker == "x":
                ax.scatter([max(mean_exc, FLOOR)], [mean_cost], marker=marker,
                           color=colour, linewidths=2.0, s=msize ** 2 * 1.8, zorder=5)
            else:
                ax.scatter([max(mean_exc, FLOOR)], [mean_cost], marker=marker,
                           facecolors=colour, edgecolors="black", linewidths=1.5,
                           s=msize ** 2 * 1.8, zorder=5)

    ax.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.4, zorder=2)
    ax.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.4, zorder=2)
    if "Grid-Only" in baselines:
        ax.axhline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.2, zorder=2)
    if "PSM'" in baselines:
        ax.axhline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=BASE, lw=1.2, zorder=2)
    if "LP'" in baselines:
        ax.axhline(float(baselines["LP'"]["cost_eur"]), ls="-.", color="#196F3D", lw=1.2, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.65, 12)
    log_axis(ax, "x", EXC_TICKS)
    ymax = int(math.ceil((max_cost + 4000) / 5000.0) * 5000)
    ax.set_ylim(44000, ymax)
    ax.yaxis.set_major_locator(FixedLocator(list(range(45000, ymax + 1, 5000))))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.set_xlabel("Mean grid exceedance (kW/step, log scale; 0 floored at 10⁻⁴)")
    ax.set_ylabel("Validation Cost (EUR)")
    ax.set_title("Unified Pareto Frontier: Cost vs. Exceedance across 4 Environments", fontsize=11.5, fontweight="bold")

    handles_m = [Line2D([], [], color=colour, marker="s", ls="", markersize=7, label=label)
                 for label, _p, colour in GROUPS]
    handles_f = [Line2D([], [], color="#333333", marker=marker, ls="", markersize=msize,
                        label=fam_name) for _, fam_name, marker, msize in fam_styles]
    handles_ref = [
        Patch(facecolor="#2ECC71", alpha=0.20, label="Feasible Region (G1 & G2)"),
        Line2D([], [], ls="--", color=GATE, lw=1.3, label="G1 / G2 Gates"),
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only"),
        Line2D([], [], ls=":", color=BASE, lw=1.2, label="PSM'"),
    ]
    if "LP'" in baselines:
        handles_ref.append(Line2D([], [], ls="-.", color="#196F3D", lw=1.2, label="LP' Oracle"))
    handles_ref.extend([
        Line2D([], [], color="#555555", marker="o", ls="", markersize=4.5, alpha=0.5, label="Single seed"),
        Line2D([], [], color="#555555", marker="o", ls="", markersize=8, markeredgecolor="black", label="Mean of 5 seeds"),
    ])
    fig.subplots_adjust(left=0.09, right=0.96, top=0.92, bottom=0.22)
    fig.legend(handles=handles_m + handles_f + handles_ref, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=True)
    fig.savefig(os.path.join(out_dir, "unified_pareto.png"), bbox_inches="tight")
    plt.close(fig)


def fig_grouped_cost_breakdown(all_families, baselines, out_dir):
    """Grouped bar plots comparing Cost Breakdown, Exceedance, and Battery Usage across environments."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 8.5), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.08})
    ax_cost, ax_exc, ax_batt = axes[0], axes[1], axes[2]

    fam_keys = [
        ("unclamped", "Unclamped"),
        ("clamped", "Clamped"),
        ("clamped_mb", "Clamped Minibatch"),
        ("clamped_forecast", "Clamped + Forecast"),
    ]
    parts = ("energy_purchase_eur", "battery_wear_eur", "unserved_penalty_eur")
    part_names = ("Grid energy bought", "Battery wear", "Unserved penalty")
    part_colours = ("#0072B2", "#7F8C8D", "#E67E22")

    y_ticks = []
    y_labels = []
    bar_h = 0.72

    all_tots = []
    all_excs = []
    all_tps = []
    row_data = []

    for g_idx, (label, _prefix, colour) in enumerate(GROUPS):
        for f_idx, (f_key, f_name) in enumerate(fam_keys):
            y = -(g_idx * 5.2 + f_idx * 1.0)
            y_ticks.append(y)
            y_labels.append(f"{label}  [{f_name}]")

            runs = all_families.get(f_key, {}).get(label, [])
            if not runs:
                row_data.append((y, label, f_key, colour, [0, 0, 0], 0.0, 0.0, 0.0, 0.0))
                continue

            totals = []
            for i, part in enumerate(parts):
                vals = [float(final_metrics(r).get(part, 0.0)) for r in runs]
                v = statistics.fmean(vals) if vals else 0.0
                totals.append(v)
            tot = sum(totals)
            all_tots.append(tot)

            excs = [float(final_metrics(r).get("mean_exceedance_kw", 0.0)) for r in runs]
            mean_exc = statistics.fmean(excs) if excs else 0.0
            all_excs.append(mean_exc)

            tps = [float(final_metrics(r).get("battery_wear_eur", 0.0)) / 20.0 for r in runs]
            mean_tp = statistics.fmean(tps) if tps else 0.0
            std_tp = statistics.stdev(tps) if len(tps) > 1 else 0.0
            all_tps.append(mean_tp + std_tp)

            row_data.append((y, label, f_key, colour, totals, tot, mean_exc, mean_tp, std_tp))

    max_tot = max([G2_COST_EUR] + all_tots + [float(b["cost_eur"]) for b in baselines.values() if "cost_eur" in b])
    max_exc = max([G1_EXC_KW, 3.26] + all_excs)
    max_tp = max([25.0] + all_tps)

    for y, label, f_key, colour, totals, tot, mean_exc, mean_tp, std_tp in row_data:
        if tot == 0.0 and mean_exc == 0.0:
            continue
        alpha_fam = 0.45 if f_key == "unclamped" else (0.65 if f_key == "clamped" else (0.85 if f_key == "clamped_mb" else 1.0))

        # Panel A: Stacked cost bars
        left = 0.0
        for i, part in enumerate(parts):
            v = totals[i]
            ax_cost.barh(y, v, left=left, height=bar_h, color=part_colours[i],
                         edgecolor="white", linewidth=0.8)
            left += v

        # Panel B: Exceedance bar
        ax_exc.barh(y, mean_exc, height=bar_h, color=colour, alpha=alpha_fam,
                    edgecolor="white", linewidth=0.8)

        # Panel C: Battery Throughput bar
        ax_batt.barh(y, mean_tp, xerr=std_tp, height=bar_h, color=colour, alpha=alpha_fam,
                     edgecolor="white", linewidth=0.8, capsize=2)

    # Panel A limits and decor
    ax_cost.set_yticks(y_ticks)
    ax_cost.set_yticklabels(y_labels, fontsize=8.5)
    ax_cost.set_ylim(min(y_ticks) - 1.0, max(y_ticks) + 1.0)
    ax_cost.set_xlim(0, max_tot * 1.06)
    ax_cost.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_xlabel("Validation Cost (EUR, mean of 5 seeds)")
    ax_cost.set_title("A. Cost Breakdown vs. Grid-Only", fontsize=11, fontweight="bold")
    if "Grid-Only" in baselines:
        ax_cost.axvline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.2, zorder=2)
    if "LP'" in baselines:
        ax_cost.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color="#196F3D", lw=1.2, zorder=2)
    if "PSM'" in baselines:
        ax_cost.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=BASE, lw=1.2, zorder=2)
    ax_cost.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.3, zorder=2)

    # Panel B limits and decor
    ax_exc.set_ylim(min(y_ticks) - 1.0, max(y_ticks) + 1.0)
    ax_exc.set_xlim(0, max_exc * 1.08)
    ax_exc.axvline(0.0, ls=":", lw=1.8, color="#1A5276", zorder=2)
    ax_exc.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.3, zorder=2)
    if "Grid-Only" in baselines:
        ax_exc.axvline(float(baselines["Grid-Only"]["exceedance_kw"]), ls="--", color=BASE, lw=1.2, zorder=2)
    ax_exc.set_xlabel("Mean Exceedance (kW/step)")
    ax_exc.set_title("B. Exceedance vs. PSM' & Grid-Only", fontsize=11, fontweight="bold")

    # Panel C limits and decor
    ax_batt.set_ylim(min(y_ticks) - 1.0, max(y_ticks) + 1.0)
    ax_batt.set_xlim(0, max_tp * 1.08)
    ax_batt.axvline(5.84, ls=":", color=BASE, lw=1.4, zorder=2)
    ax_batt.axvline(24.99, ls="-.", color="#196F3D", lw=1.4, zorder=2)
    ax_batt.set_xlabel("Battery Throughput (MWh, 26 Val Weeks)")
    ax_batt.set_title("C. Battery Usage Across Methods", fontsize=11, fontweight="bold")

    cost_handles = [Patch(facecolor=part_colours[i], label=part_names[i]) for i in range(len(parts))]
    fam_handles = [
        Patch(facecolor="#777777", alpha=0.45 if fk == "unclamped" else (0.65 if fk == "clamped" else (0.85 if fk == "clamped_mb" else 1.0)),
              label=fn) for fk, fn in fam_keys
    ]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.3, label="G1 / G2 Gates"),
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only"),
        Line2D([], [], ls=":", color=BASE, lw=1.2, label="PSM' Target"),
        Line2D([], [], ls="-.", color="#196F3D", lw=1.2, label="LP' Oracle"),
    ]

    fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.16)
    fig.legend(handles=cost_handles + fam_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, frameon=True)
    fig.savefig(os.path.join(out_dir, "grouped_cost_breakdown.png"), bbox_inches="tight")
    plt.close(fig)


def fig_battery_usage(all_families, baselines, out_dir):
    """Symmetric comparison of Battery Energy Throughput and Active Operating Time across environments."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.5), sharey=True, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    ax_tp, ax_soc = axes[0], axes[1]

    fam_keys = [
        ("unclamped", "Unclamped"),
        ("clamped", "Clamped"),
        ("clamped_mb", "Clamped Minibatch"),
        ("clamped_forecast", "Clamped + Forecast"),
    ]
    bar_h = 0.72
    y_ticks = []
    y_labels = []

    all_tps = []
    row_data = []

    for g_idx, (label, _prefix, colour) in enumerate(GROUPS):
        for f_idx, (f_key, f_name) in enumerate(fam_keys):
            y = -(g_idx * 5.2 + f_idx * 1.0)
            y_ticks.append(y)
            y_labels.append(f"{label}  [{f_name}]")

            runs = all_families.get(f_key, {}).get(label, [])
            if not runs:
                row_data.append((y, label, f_key, colour, 0.0, 0.0, 0.0, 0.0))
                continue

            tps = [float(final_metrics(r).get("battery_wear_eur", 0.0)) / 20.0 for r in runs]
            mean_tp = statistics.fmean(tps) if tps else 0.0
            std_tp = statistics.stdev(tps) if len(tps) > 1 else 0.0
            all_tps.append(mean_tp + std_tp)

            socs = [100.0 - float(final_metrics(r).get("soc_pinned_pct", 100.0)) for r in runs]
            mean_soc = statistics.fmean(socs) if socs else 0.0
            std_soc = statistics.stdev(socs) if len(socs) > 1 else 0.0

            row_data.append((y, label, f_key, colour, mean_tp, std_tp, mean_soc, std_soc))

    max_tp = max([25.0] + all_tps)

    for y, label, f_key, colour, mean_tp, std_tp, mean_soc, std_soc in row_data:
        if mean_tp == 0.0 and mean_soc == 0.0:
            continue
        alpha_fam = 0.45 if f_key == "unclamped" else (0.65 if f_key == "clamped" else (0.85 if f_key == "clamped_mb" else 1.0))

        # Panel A: Throughput
        ax_tp.barh(y, mean_tp, xerr=std_tp, height=bar_h, color=colour, alpha=alpha_fam,
                   edgecolor="white", linewidth=0.8, capsize=2)

        # Panel B: Active SoC %
        ax_soc.barh(y, mean_soc, xerr=std_soc, height=bar_h, color=colour, alpha=alpha_fam,
                    edgecolor="white", linewidth=0.8, capsize=2)

    # Panel A limits & decor
    ax_tp.set_yticks(y_ticks)
    ax_tp.set_yticklabels(y_labels, fontsize=8.5)
    ax_tp.set_ylim(min(y_ticks) - 1.0, max(y_ticks) + 1.0)
    ax_tp.set_xlim(0, max_tp * 1.08)
    ax_tp.axvline(0.0, ls="--", color=BASE, lw=1.2, zorder=2)
    ax_tp.axvline(5.84, ls=":", color=BASE, lw=1.4, zorder=2)
    ax_tp.axvline(24.99, ls="-.", color="#196F3D", lw=1.4, zorder=2)
    ax_tp.set_xlabel("Battery Throughput (MWh, 26 Val Weeks)")
    ax_tp.set_title("A. Battery Energy Throughput (MWh) Across Environments", fontsize=11, fontweight="bold")

    # Panel B limits & decor
    ax_soc.set_ylim(min(y_ticks) - 1.0, max(y_ticks) + 1.0)
    ax_soc.set_xlim(0, 100)
    ax_soc.set_xlabel("Active Unpinned SoC Time (% of 4,368 Val Hours)")
    ax_soc.set_title("B. Active Battery Dynamic Operating Time (% Hours Unpinned)", fontsize=11, fontweight="bold")

    # Bottom legend
    method_handles = [Patch(facecolor=c, label=m) for m, _p, c in GROUPS]
    fam_handles = [
        Patch(facecolor="#777777", alpha=0.45 if fk == "unclamped" else (0.65 if fk == "clamped" else (0.85 if fk == "clamped_mb" else 1.0)),
              label=fn) for fk, fn in fam_keys
    ]
    ref_handles = [
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only (0.0 MWh)"),
        Line2D([], [], ls=":", color=BASE, lw=1.4, label="PSM' (5.84 MWh)"),
        Line2D([], [], ls="-.", color="#196F3D", lw=1.4, label="LP' Oracle (24.99 MWh)"),
    ]

    fig.subplots_adjust(left=0.18, right=0.96, top=0.90, bottom=0.18)
    fig.legend(handles=method_handles + fam_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True)
    fig.savefig(os.path.join(out_dir, "battery_usage_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def fig_targeted_learning_curves(all_families, baselines, out_dir):
    """Learning curves for competitive methods (Residual and Shielded) across environments."""
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    comp_methods = [
        ("PPO-Residual", "#009E73"),
        ("Shielded PPO", "#CC79A7"),
    ]
    fam_styles = [
        ("unclamped", "Unclamped", ":", "x", 1.5),
        ("clamped", "Clamped", "--", "o", 1.8),
        ("clamped_mb", "Clamped Minibatch", "-.", "^", 1.8),
        ("clamped_forecast", "Clamped + Forecast", "-", "*", 2.0),
    ]

    ax_cost, ax_exc = axes[0], axes[1]
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

    ax_cost.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.3, zorder=2)
    if "Grid-Only" in baselines:
        ax_cost.axhline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.2, zorder=2)
    if "PSM'" in baselines:
        ax_cost.axhline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=BASE, lw=1.2, zorder=2)
    if "LP'" in baselines:
        ax_cost.axhline(float(baselines["LP'"]["cost_eur"]), ls="-.", color="#196F3D", lw=1.2, zorder=2)
    ymax = int(math.ceil((max_cost_upper + 2500) / 2500.0) * 2500)
    ax_cost.set_ylim(44000, ymax)
    ax_cost.yaxis.set_major_locator(FixedLocator(list(range(45000, ymax + 1, 5000))))
    ax_cost.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_cost.set_ylabel("Val cost (EUR)")
    ax_cost.set_title("A. Validation Cost over Training (Targeted Competitive Methods)")

    ax_exc.axhline(G1_EXC_KW, ls="--", color=GATE, lw=1.3, zorder=2)
    ax_exc.set_yscale("log")
    ax_exc.set_ylim(FLOOR * 0.65, 10)
    log_axis(ax_exc, "y", EXC_TICKS)
    ax_exc.set_ylabel("Mean exceedance (kW/step, log scale)")
    ax_exc.set_xlabel("Training steps (thousands)")
    ax_exc.set_title("B. Mean Grid Exceedance over Training (Targeted Competitive Methods)")
    ax_exc.set_xticks(STEP_TICKS)

    method_handles = [Line2D([], [], color=c, lw=2.5, label=m) for m, c in comp_methods]
    env_handles = [
        Line2D([], [], color="#444444", ls=ls, lw=lw, marker=marker,
               markersize=6, label=fn) for _, fn, ls, marker, lw in fam_styles
    ]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.3, label="G1 / G2 Gates"),
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only"),
        Line2D([], [], ls=":", color=BASE, lw=1.2, label="PSM'"),
    ]
    if "LP'" in baselines:
        ref_handles.append(Line2D([], [], ls="-.", color="#196F3D", lw=1.2, label="LP' Oracle"))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.20, hspace=0.20)
    fig.legend(handles=method_handles + env_handles + ref_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=True)
    fig.savefig(os.path.join(out_dir, "targeted_learning_curves.png"), bbox_inches="tight")
    plt.close(fig)


def fig_stress_weeks(out_dir: str):
    """Stress week comparison highlighting the true cap-binding week 2019-W49."""
    import csv
    import numpy as np

    path = os.path.join(TABLES, "test_stress_weeks.csv")
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    crits = ["lowest_pv", "highest_price", "hottest", "stormiest", "highest_exceedance"]
    crit_labels = [
        "Lowest PV\n(2019-W48)",
        "Highest Price\n(2019-W45)",
        "Hottest\n(2019-W30)",
        "Stormiest\n(2019-W51)",
        "Highest Exceedance\n(2019-W49)",
    ]
    methods = ["Grid-Only", "PSM'", "PPO", "PPO-Lagrangian", "PPO-Residual", "Shielded PPO"]
    colors = ["#555555", "#1A5276", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    def get_group(m):
        if m in ("Grid-Only", "PSM'"):
            return m
        if "shielded" in m:
            return "Shielded PPO"
        if "residual" in m:
            return "PPO-Residual"
        if "lagrangian" in m:
            return "PPO-Lagrangian"
        return "PPO"

    # Panel A: filter to clamped_mb + baselines
    data_cost = {m: {c: [] for c in crits} for m in methods}
    for r in rows:
        fam = r["family"]
        g = get_group(r["model"])
        c = r["stress_criterion"]
        if fam == "baseline" and g in ("Grid-Only", "PSM'"):
            if c in data_cost[g]:
                data_cost[g][c].append(float(r["cost_eur"]))
        elif fam == "clamped_mb" and g in methods:
            if c in data_cost[g]:
                data_cost[g][c].append(float(r["cost_eur"]))

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.16})
    ax1, ax2 = axes[0], axes[1]

    # Panel A: Grouped vertical bars across 5 stress weeks
    x1 = np.arange(len(crits))
    n_m = len(methods)
    bar_w1 = 0.13

    for i, m in enumerate(methods):
        c_means = [float(np.mean(data_cost[m][c])) for c in crits]
        c_errs = [float(np.std(data_cost[m][c], ddof=1)) if len(data_cost[m][c]) > 1 else 0.0 for c in crits]
        pos = x1 - (n_m - 1) * bar_w1 / 2 + i * bar_w1
        ax1.bar(pos, c_means, bar_w1, yerr=c_errs, color=colors[i], edgecolor="white", linewidth=0.8, capsize=2)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(crit_labels, fontsize=9.5)
    ax1.set_ylabel("Total Cost (EUR)")
    ax1.set_title("A. Test Cost vs. Grid-Only Across All 5 Stress Weeks", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 5400)
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: filter to highest_exceedance (2019-W49)
    rl_methods = ["PPO", "PPO-Lagrangian", "PPO-Residual", "Shielded PPO"]
    x2 = np.arange(len(rl_methods))
    bar_w2 = 0.32

    w49_clamped = {m: [] for m in rl_methods}
    w49_clamped_mb = {m: [] for m in rl_methods}

    for r in rows:
        if r["stress_criterion"] != "highest_exceedance":
            continue
        fam = r["family"]
        g = get_group(r["model"])
        if fam == "clamped" and g in rl_methods:
            w49_clamped[g].append(float(r["violation_hours"]))
        elif fam == "clamped_mb" and g in rl_methods:
            w49_clamped_mb[g].append(float(r["violation_hours"]))

    for j, m in enumerate(rl_methods):
        pos_fb = x2[j] - bar_w2 / 2
        pos_mb = x2[j] + bar_w2 / 2
        fb_mean = float(np.mean(w49_clamped[m])) if w49_clamped[m] else 0.0
        fb_std = float(np.std(w49_clamped[m], ddof=1)) if len(w49_clamped[m]) > 1 else 0.0
        mb_mean = float(np.mean(w49_clamped_mb[m])) if w49_clamped_mb[m] else 0.0
        mb_std = float(np.std(w49_clamped_mb[m], ddof=1)) if len(w49_clamped_mb[m]) > 1 else 0.0

        ax2.bar(pos_fb, fb_mean, bar_w2, yerr=fb_std, color=COLOURS[m], alpha=0.60, hatch="//", edgecolor="#333333", linewidth=0.8, capsize=3)
        ax2.bar(pos_mb, mb_mean, bar_w2, yerr=mb_std, color=COLOURS[m], alpha=1.0, edgecolor="white", linewidth=0.8, capsize=3)

    ax2.axhline(66.0, ls="--", color="#555555", lw=1.5, label="Grid-Only (66h / 10.3 kW)")
    ax2.axhline(0.0, ls=":", color="#1A5276", lw=2.0, label="PSM' Target (0h / 0.0 kW)")

    ax2.set_ylim(0, 75)
    ax2.set_ylabel("Violation Hours on 2019-W49 (out of 168h)")
    ax2.set_title("B. Cap-Binding Stress Week (2019-W49): Violation Hours by Family", fontsize=11, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rl_methods, fontsize=9.5)
    ax2.grid(axis="y", alpha=0.3)

    # Shared bottom legend (ncol=5, 10 items)
    legend_handles = [
        Patch(facecolor="#555555", label="Grid-Only"),
        Patch(facecolor="#1A5276", label="PSM'"),
        Patch(facecolor="#0072B2", label="PPO"),
        Patch(facecolor="#D55E00", label="PPO-Lagrangian"),
        Patch(facecolor="#009E73", label="PPO-Residual"),
        Patch(facecolor="#CC79A7", label="Shielded PPO"),
        Patch(facecolor="#777777", alpha=0.60, hatch="//", edgecolor="#333333", label="Clamped (Full-Batch)"),
        Patch(facecolor="#777777", alpha=1.0, edgecolor="white", label="Clamped Minibatch"),
        Line2D([], [], ls="--", color="#555555", lw=1.5, label="Grid-Only (66h / 10.3 kW)"),
        Line2D([], [], ls=":", color="#1A5276", lw=2.0, label="PSM' Target (0h / 0.0 kW)"),
    ]

    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.20, wspace=0.16)
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=True)
    fig.savefig(os.path.join(out_dir, "stress_weeks_comparison.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_family(family: str, baselines: dict):
    global FIGS
    if family in ("clamped_forecast", "forecast"):
        FIGS = os.path.join(ROOT, "results", "figures", "clamped_forecast")
    elif family in ("clamped_mb", "mb"):
        FIGS = os.path.join(ROOT, "results", "figures", "clamped_mb")
    elif family in ("clamped", "s3v2", "v2"):
        FIGS = os.path.join(ROOT, "results", "figures", "clamped")
    else:
        FIGS = os.path.join(ROOT, "results", "figures", "unclamped")
    os.makedirs(FIGS, exist_ok=True)

    runs = load_runs(family=family)
    total = sum(len(v) for v in runs.values())
    if total == 0:
        print("no run JSONs found in %s for family %r - skipping" % (RUNS, family))
        return

    print("loaded %d runs in %d groups (family: %s)" % (total, len(runs), family))
    for fn, args in ((fig_learning_curves, (runs, baselines)),
                     (fig_cost_vs_exceedance, (runs, baselines)),
                     (fig_lambda, (runs,)),
                     (fig_per_seed, (runs, baselines)),
                     (fig_cost_breakdown, (runs, baselines))):
        try:
            fn(*args)
            print("  ok  %s" % fn.__name__)
        except Exception as exc:  # noqa: BLE001
            print("  FAIL %s: %s" % (fn.__name__, exc))
    print("figures written to %s" % FIGS)


def plot_comparison(baselines: dict):
    comp_dir = os.path.join(ROOT, "results", "figures", "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    all_families = {
        "unclamped": load_runs("unclamped"),
        "clamped": load_runs("clamped"),
        "clamped_mb": load_runs("clamped_mb"),
        "clamped_forecast": load_runs("clamped_forecast"),
    }

    print("generating cross-family comparison figures in %s" % comp_dir)
    for fn, args in (
        (fig_unified_pareto, (all_families, baselines, comp_dir)),
        (fig_grouped_cost_breakdown, (all_families, baselines, comp_dir)),
        (fig_battery_usage, (all_families, baselines, comp_dir)),
        (fig_targeted_learning_curves, (all_families, baselines, comp_dir)),
        (fig_stress_weeks, (comp_dir,)),
    ):
        try:
            fn(*args)
            print("  ok  %s" % fn.__name__)
        except Exception as exc:  # noqa: BLE001
            print("  FAIL %s: %s" % (fn.__name__, exc))


def main():
    family = "unclamped"
    if "--family" in sys.argv:
        idx = sys.argv.index("--family")
        if idx + 1 < len(sys.argv):
            family = sys.argv[idx + 1]

    baselines = load_baselines()

    if family in ("comparison", "compare"):
        plot_comparison(baselines)
    elif family == "all":
        for fam in ("unclamped", "clamped", "clamped_mb", "clamped_forecast"):
            plot_family(fam, baselines)
        plot_comparison(baselines)
    else:
        plot_family(family, baselines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
