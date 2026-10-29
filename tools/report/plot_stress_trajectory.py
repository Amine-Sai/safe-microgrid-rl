#!/usr/bin/env python3
"""Plot hour-by-hour dispatch trajectory on the cap-binding stress week (2019-W49).

Visualizes Grid-Only, PSM', PPO-Residual, and Shielded PPO over 168 hours:
1. Grid import power vs. the 368.15 kW contract cap (highlighting violations).
2. Battery state of charge (SoC) dynamics and reserve floors.
3. Load shedding profiles illustrating the safety vs. penalty trade-off.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
from env.smartgrid_env import SmartGridEnv
from env.cap_shield import CapShield
from experiments.train_core import load_agent_for_method, load_config_with_base

W49_START_IDX = 3360  # 2019-12-02 00:00:00 to 2019-12-08 23:00:00 (168 hours)
EP_LEN = 168
CAP_KW = 368.153918

COLORS = {
    "Grid-Only": "#7F8C8D",
    "PSM'": "#2C3E50",
    "PPO-Residual": "#009E73",
    "Shielded PPO": "#CC79A7",
    "Cap": "#C0392B",
}


def simulate_policy(env, policy=None, agent_wrapper=None):
    """Run a 168-hour episode and record time series."""
    obs, _ = env.reset(options={"start_idx": W49_START_IDX})
    history = {
        "grid_import": [],
        "grid_exceedance": [],
        "battery_power": [],
        "soc": [],
        "unserved": [],
        "price": [],
    }
    for _ in range(EP_LEN):
        if agent_wrapper is not None:
            action = agent_wrapper.select_action(obs, evaluate=True)[0]
        else:
            action = policy.get_action(obs)
        obs, _, _, _, info = env.step(action)
        history["grid_import"].append(info["grid_import_kw"])
        history["grid_exceedance"].append(info["grid_exceedance_kw"])
        history["battery_power"].append(info["battery_power_kw"])
        history["soc"].append(info["soc"])
        history["unserved"].append(info["total_unserved_kw"])
        history["price"].append(info["price_eur_mwh"])
    return {k: np.array(v) for k, v in history.items()}


def main():
    print("Simulating 2019-W49 stress week dispatch trajectories...")

    # 1. Grid-Only
    env_base = SmartGridEnv({"split": "test", "charge_clamp": True})
    hist_grid = simulate_policy(env_base, policy=GridOnlyPolicy())

    # 2. PSM'
    hist_psm = simulate_policy(env_base, policy=PeakShaveMinPolicy())

    # 3. Residual PPO (clamped_mb_ppo_residual_seed101)
    cfg_r = load_config_with_base("clamped_mb_ppo_residual")
    cfg_r["seed"] = 101
    agent_r = load_agent_for_method(cfg_r["method"], cfg_r)
    agent_r.load_models(os.path.join(REPO_ROOT, "checkpoints", "ppo_residual", "clamped_mb_ppo_residual_seed101_final.pt"))
    hist_res = simulate_policy(env_base, agent_wrapper=agent_r)

    # 4. Shielded PPO (clamped_mb_ppo_shielded_seed101)
    cfg_s = load_config_with_base("clamped_mb_ppo_shielded")
    cfg_s["seed"] = 101
    agent_s = load_agent_for_method(cfg_s["method"], cfg_s)
    agent_s.load_models(os.path.join(REPO_ROOT, "checkpoints", "ppo_shielded", "clamped_mb_ppo_shielded_seed101_final.pt"))
    env_shielded = CapShield(SmartGridEnv({"split": "test", "charge_clamp": True}))
    hist_shi = simulate_policy(env_shielded, agent_wrapper=agent_s)

    hours = np.arange(EP_LEN)

    # Setup 3-panel figure
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})

    # Panel 1: Grid Import Power & Cap Exceedance
    ax1 = axes[0]
    ax1.plot(hours, hist_grid["grid_import"], label="Grid-Only", color=COLORS["Grid-Only"], linestyle=":", alpha=0.75, linewidth=1.4)
    ax1.plot(hours, hist_psm["grid_import"], label="PSM'", color=COLORS["PSM'"], linewidth=1.6)
    ax1.plot(hours, hist_res["grid_import"], label="PPO-Residual", color=COLORS["PPO-Residual"], linewidth=1.6)
    ax1.plot(hours, hist_shi["grid_import"], label="Shielded PPO", color=COLORS["Shielded PPO"], linewidth=1.8)

    ax1.axhline(CAP_KW, color=COLORS["Cap"], linestyle="--", linewidth=1.5, label="Contract Cap")

    # Highlight residual exceedances
    res_exc_mask = hist_res["grid_import"] > CAP_KW
    ax1.fill_between(hours, CAP_KW, hist_res["grid_import"], where=res_exc_mask, color="#E74C3C", alpha=0.45, label="Cap Exceedance")

    ax1.set_ylabel("Grid Import (kW)")
    ax1.set_title("A. Grid Import Trajectory vs. Contract Import Cap (2019-W49, Peak Exceedance Week)", fontweight="bold")
    ax1.set_ylim(0, 460)
    ax1.legend(loc="upper right", ncol=3, fontsize=9, frameon=True, framealpha=0.92, facecolor="white", edgecolor="#DDD")
    ax1.grid(True)

    # Panel 2: Battery State of Charge (SoC)
    ax2 = axes[1]
    ax2.plot(hours, hist_psm["soc"] * 100.0, label="PSM'", color=COLORS["PSM'"], linewidth=1.6)
    ax2.plot(hours, hist_res["soc"] * 100.0, label="PPO-Residual", color=COLORS["PPO-Residual"], linewidth=1.6)
    ax2.plot(hours, hist_shi["soc"] * 100.0, label="Shielded PPO", color=COLORS["Shielded PPO"], linewidth=1.8)

    ax2.axhline(90.0, color="#888888", linestyle=":", linewidth=1.0)
    ax2.axhline(10.0, color="#888888", linestyle=":", linewidth=1.0)
    ax2.fill_between(hours, 10.0, 90.0, color="#2ECC71", alpha=0.06, label="Safe Band [10%, 90%]")

    ax2.set_ylabel("Battery SoC (%)")
    ax2.set_title("B. Battery State of Charge (SoC) Dynamics Across Daily Cycles", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", ncol=4, fontsize=9, frameon=True, framealpha=0.92, facecolor="white", edgecolor="#DDD")
    ax2.grid(True)

    # Panel 3: Load Shedding Profiles
    ax3 = axes[2]
    ax3.plot(hours, hist_shi["unserved"], label="Shielded PPO", color=COLORS["Shielded PPO"], linewidth=1.8)
    ax3.plot(hours, hist_psm["unserved"], label="PSM'", color=COLORS["PSM'"], linewidth=1.5, linestyle="--")
    ax3.plot(hours, hist_res["unserved"], label="PPO-Residual", color=COLORS["PPO-Residual"], linewidth=1.6, alpha=0.9)

    ax3.set_ylabel("Load Shed (kW)")
    ax3.set_xlabel("Hour of Week (0 = Mon 00:00, 167 = Sun 23:00, Dec 2-8 2019)")
    ax3.set_title("C. Load Shedding Dispatch (The Safety vs. Economic Penalty Trade-off)", fontweight="bold")
    ax3.set_ylim(-2, 110)
    ax3.set_xlim(0, 167)
    ax3.legend(loc="upper right", ncol=3, fontsize=9, frameon=True, framealpha=0.92, facecolor="white", edgecolor="#DDD")
    ax3.grid(True)

    # X-axis ticks every 24 hours (days of the week)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_ticks = [i * 24 + 12 for i in range(7)]
    ax3.set_xticks([i * 24 for i in range(8)])
    ax3.set_xticklabels([f"0h\n(Mon)", "24h\n(Tue)", "48h\n(Wed)", "72h\n(Thu)", "96h\n(Fri)", "120h\n(Sat)", "144h\n(Sun)", "168h\n(End)"])

    fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.08, hspace=0.18)

    out_file = os.path.join(REPO_ROOT, "results", "figures", "comparison", "w49_dispatch_trajectory.png")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
