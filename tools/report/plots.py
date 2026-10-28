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
    "figure.constrained_layout.use": True,
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
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.4), sharex=True)

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
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.savefig(os.path.join(FIGS, "learning_curves.png"))
    plt.close(fig)


def fig_cost_vs_exceedance(runs, baselines):
    """Fig 2 - the cost / exceedance frontier, one marker per (method, seed)."""
    fig, ax = plt.subplots(figsize=(9.4, 6.6))

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
    ax.axhline(G2_COST_EUR, ls="--", color=GATE, lw=1.5, zorder=2,
               label="G2 gate  %.0f EUR (1.05 x PSM')" % G2_COST_EUR)
    ax.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.5, zorder=2,
               label="G1 gate  %.2f kW/step" % G1_EXC_KW)
    handles += [
        Line2D([], [], ls="--", color=GATE, lw=1.5,
               label="G2 gate  %.0f EUR" % G2_COST_EUR),
        Line2D([], [], ls="--", color=GATE, lw=1.5,
               label="G1 gate  %.2f kW/step" % G1_EXC_KW),
    ]

    ax.set_xlabel("Mean exceedance above cap (kW/step, log scale; 0 floored at %g)"
                  % FLOOR)
    ax.set_ylabel("Val cost (EUR)")
    ax.set_title("Validation Cost vs. Mean Grid Exceedance")
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.savefig(os.path.join(FIGS, "cost_vs_exceedance.png"))
    plt.close(fig)


def fig_lambda(runs):
    """Fig 3 - lambda multiplier over training for the two constrained variants."""
    fig, ax = plt.subplots(figsize=(9.4, 6.0))

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
    fig.legend(handles=handles, loc="outside lower center", ncol=1)
    fig.savefig(os.path.join(FIGS, "lambda_trajectory.png"))
    plt.close(fig)


def fig_per_seed(runs, baselines):
    """Fig 4 - per-seed cost and exceedance, with the two gates."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))

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
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.savefig(os.path.join(FIGS, "per_seed_results.png"))
    plt.close(fig)


def fig_cost_breakdown(runs, baselines):
    """Fig 5 - where the money goes: energy purchase vs unserved penalty vs wear."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    parts = ("energy_purchase_eur", "unserved_penalty_eur", "battery_wear_eur")
    names = ("energy purchase", "unserved penalty", "battery wear")
    colours = ("#0072B2", "#E67E22", "#7F8C8D")

    ypos = list(range(len(GROUPS)))[::-1]
    max_total = G2_COST_EUR
    for y, (label, _prefix, _c) in zip(ypos, GROUPS):
        left = 0.0
        totals = []
        for i, part in enumerate(parts):
            vals = [float(final_metrics(r).get(part, 0.0)) for r in runs[label]]
            if not vals:
                continue
            v = sum(vals) / len(vals)
            totals.append(v)
            ax.barh(y, v, left=left, height=0.52, color=colours[i],
                    edgecolor="white", linewidth=0.8)
            if v >= 4000:
                ax.text(left + v / 2, y, "%.0f" % v, ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += v
        if totals and sum(totals) > 0:
            max_total = max(max_total, left)
            ax.text(left + 600, y, "wear %.2f%%" % (100 * totals[2] / sum(totals)),
                    va="center", fontsize=8.5, color="#555555")

    handles = [Patch(facecolor=c, label=n) for c, n in zip(colours, names)]
    for name, style, col in (("Grid-Only", "--", BASE), ("PSM'", ":", BASE), ("LP'", "-.", "#196F3D")):
        if name in baselines:
            ax.axvline(float(baselines[name]["cost_eur"]), ls=style, color=col,
                       lw=1.4, zorder=3)
            handles.append(Line2D([], [], ls=style, color=col, lw=1.4,
                                  label="%s (frozen) %.0f"
                                        % (name,
                                           float(baselines[name]["cost_eur"]))))

    xmax = max(68000, int(math.ceil((max_total + 9000) / 10000.0) * 10000))
    ax.set_yticks(ypos)
    ax.set_yticklabels([g[0] for g in GROUPS])
    ax.set_xlabel("Val cost (EUR, mean over 5 seeds)")
    ax.set_title("Validation Cost Breakdown by Component")
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(FixedLocator(list(range(0, xmax + 1, 20000))))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.savefig(os.path.join(FIGS, "cost_breakdown.png"))
    plt.close(fig)


# Cross-family comparison figures
def fig_unified_pareto(all_families, baselines, out_dir):
    """Unified Pareto scatter plot comparing Cost vs Exceedance across environments."""
    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    fam_styles = [
        ("unclamped", "Unclamped", "x", 7.0),
        ("clamped", "Clamped", "o", 6.5),
        ("clamped_mb", "Clamped Minibatch", "^", 7.5),
        ("clamped_forecast", "Clamped + Forecast", "*", 9.5),
    ]

    # Shaded feasible quadrant (x <= G1, y <= G2)
    ax.fill_between([FLOOR * 0.65, G1_EXC_KW], [42000, 42000], [G2_COST_EUR, G2_COST_EUR],
                    color="#2ECC71", alpha=0.10, zorder=1)
    ax.text(FLOOR * 1.5, 56500, "Feasible Region\n(G1 & G2)", color="#1E8449",
            fontsize=9.5, fontweight="bold", va="top", zorder=2)

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
    ax.set_title("Unified Pareto Frontier: Cost vs. Exceedance across 3 Environments")

    handles_m = [Line2D([], [], color=colour, marker="s", ls="", markersize=7, label=label)
                 for label, _p, colour in GROUPS]
    handles_f = [Line2D([], [], color="#333333", marker=marker, ls="", markersize=msize,
                        label=fam_name) for _, fam_name, marker, msize in fam_styles]
    handles_ref = [
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
    fig.legend(handles=handles_m + handles_f + handles_ref, loc="outside lower center", ncol=4)
    fig.savefig(os.path.join(out_dir, "unified_pareto.png"))
    plt.close(fig)


def fig_grouped_cost_breakdown(all_families, baselines, out_dir):
    """Grouped bar plots comparing Cost Breakdown and Exceedance across environments."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 11.2), width_ratios=[1.25, 0.75])

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

    ax_cost, ax_exc = axes[0], axes[1]
    max_total = G2_COST_EUR

    for g_idx, (label, _prefix, colour) in enumerate(GROUPS):
        for f_idx, (f_key, f_name) in enumerate(fam_keys):
            y = -(g_idx * 5.2 + f_idx * 1.0)
            y_ticks.append(y)
            y_labels.append(f"{label}  [{f_name}]")

            runs = all_families.get(f_key, {}).get(label, [])
            if not runs:
                continue

            left = 0.0
            totals = []
            for i, part in enumerate(parts):
                vals = [float(final_metrics(r).get(part, 0.0)) for r in runs]
                v = statistics.fmean(vals) if vals else 0.0
                totals.append(v)
                ax_cost.barh(y, v, left=left, height=bar_h, color=part_colours[i],
                             edgecolor="white", linewidth=0.8)
                if v >= 6000:
                    ax_cost.text(left + v / 2, y, "%.0fk" % (v / 1000), ha="center", va="center",
                                 color="white", fontsize=8.5, fontweight="bold")
                left += v
            max_total = max(max_total, left)

            if label == "Shielded PPO" and totals[2] > 0:
                ax_cost.text(left + 600, y, "unserved %.0f EUR" % totals[2], va="center",
                             fontsize=8.5, color="#D35400", fontweight="bold")

            excs = [float(final_metrics(r).get("mean_exceedance_kw", 0.0)) for r in runs]
            mean_exc = statistics.fmean(excs) if excs else 0.0
            exc_val = max(mean_exc, FLOOR)
            alpha_fam = 0.55 if f_key == "unclamped" else (0.8 if f_key == "clamped" else 1.0)
            ax_exc.barh(y, exc_val, height=bar_h, color=colour, alpha=alpha_fam,
                        edgecolor="white", linewidth=0.8)
            if exc_val > 0.01:
                ax_exc.text(exc_val * 1.15, y, "%.2f kW" % exc_val, va="center",
                            fontsize=8, color="#333333")
            else:
                ax_exc.text(FLOOR * 1.25, y, "0.00 kW", va="center",
                            fontsize=8, color="#27AE60", fontweight="bold")

    xmax = max(68000, int(math.ceil((max_total + 12000) / 10000.0) * 10000))
    ax_cost.set_yticks(y_ticks)
    ax_cost.set_yticklabels(y_labels, fontsize=9.5)
    ax_cost.set_ylim(min(y_ticks) - 1.0, max(y_ticks) + 1.0)
    ax_cost.set_xlabel("Validation Cost (EUR, mean of 5 seeds)")
    ax_cost.set_title("A. Stacked Cost Breakdown by Model & Environment", fontsize=11, fontweight="bold")
    ax_cost.set_xlim(0, xmax)
    ax_cost.xaxis.set_major_locator(FixedLocator(list(range(0, xmax + 1, 20000))))
    ax_cost.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    if "Grid-Only" in baselines:
        ax_cost.axvline(float(baselines["Grid-Only"]["cost_eur"]), ls="--", color=BASE, lw=1.2, zorder=2)
    if "PSM'" in baselines:
        ax_cost.axvline(float(baselines["PSM'"]["cost_eur"]), ls=":", color=BASE, lw=1.2, zorder=2)
    if "LP'" in baselines:
        ax_cost.axvline(float(baselines["LP'"]["cost_eur"]), ls="-.", color="#196F3D", lw=1.2, zorder=2)
    ax_cost.axvline(G2_COST_EUR, ls="--", color=GATE, lw=1.3, zorder=2)

    ax_exc.set_yticks(y_ticks)
    ax_exc.set_yticklabels([])
    ax_exc.set_ylim(min(y_ticks) - 1.0, max(y_ticks) + 1.0)
    ax_exc.set_xscale("log")
    ax_exc.set_xlim(FLOOR * 0.65, 15)
    log_axis(ax_exc, "x", EXC_TICKS)
    ax_exc.axvline(G1_EXC_KW, ls="--", color=GATE, lw=1.3, zorder=2)
    ax_exc.set_xlabel("Mean Exceedance (kW/step, log scale; 0 floored)")
    ax_exc.set_title("B. Mean Grid Exceedance by Model & Environment", fontsize=11, fontweight="bold")

    cost_handles = [Patch(facecolor=part_colours[i], label=part_names[i]) for i in range(len(parts))]
    fam_handles = [
        Patch(facecolor="#777777", alpha=0.45 if fk == "unclamped" else (0.65 if fk == "clamped" else (0.85 if fk == "clamped_mb" else 1.0)),
              label=fn) for fk, fn in fam_keys
    ]
    ref_handles = [
        Line2D([], [], ls="--", color=GATE, lw=1.3, label="G1 / G2 Gates"),
        Line2D([], [], ls="--", color=BASE, lw=1.2, label="Grid-Only"),
        Line2D([], [], ls=":", color=BASE, lw=1.2, label="PSM'"),
    ]
    if "LP'" in baselines:
        ref_handles.append(Line2D([], [], ls="-.", color="#196F3D", lw=1.2, label="LP' Oracle"))
    fig.legend(handles=cost_handles + fam_handles + ref_handles, loc="outside lower center", ncol=5)
    fig.savefig(os.path.join(out_dir, "grouped_cost_breakdown.png"))
    plt.close(fig)


def fig_targeted_learning_curves(all_families, baselines, out_dir):
    """Learning curves for competitive methods (Residual and Shielded) across environments."""
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 8.4), sharex=True)

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
    fig.legend(handles=method_handles + env_handles + ref_handles, loc="outside lower center", ncol=4)
    fig.savefig(os.path.join(out_dir, "targeted_learning_curves.png"))
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

    data_cost = {m: {c: [] for c in crits} for m in methods}
    data_viol = {m: {c: [] for c in crits} for m in methods}

    for r in rows:
        g = get_group(r["model"])
        c = r["stress_criterion"]
        if g in data_cost and c in data_cost[g]:
            data_cost[g][c].append(float(r["cost_eur"]))
            data_viol[g][c].append(float(r["violation_hours"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    x = np.arange(len(crits))
    n_m = len(methods)
    bar_w = 0.13

    for i, m in enumerate(methods):
        c_means = [float(np.mean(data_cost[m][c])) for c in crits]
        c_errs = [float(np.std(data_cost[m][c])) if len(data_cost[m][c]) > 1 else 0.0 for c in crits]
        v_means = [float(np.mean(data_viol[m][c])) for c in crits]
        v_errs = [float(np.std(data_viol[m][c])) if len(data_viol[m][c]) > 1 else 0.0 for c in crits]

        pos = x - (n_m - 1) * bar_w / 2 + i * bar_w
        axes[0].bar(pos, c_means, bar_w, yerr=c_errs, label=m, color=colors[i], edgecolor="white", capsize=2)
        axes[1].bar(pos, v_means, bar_w, yerr=v_errs, label=m, color=colors[i], edgecolor="white", capsize=2)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(crit_labels, fontsize=9)
    axes[0].set_ylabel("Total Cost (EUR)")
    axes[0].set_title("A. Test Cost across Stress Weeks", fontweight="bold")
    axes[0].set_ylim(0, 4300)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(crit_labels, fontsize=9)
    axes[1].set_ylabel("Violation Hours (h out of 168h)")
    axes[1].set_title("B. Grid Violation Hours across Stress Weeks", fontweight="bold")
    axes[1].set_ylim(0, 80)
    axes[1].grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper left", ncol=2, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "stress_weeks_comparison.png"))
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
