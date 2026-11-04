#!/usr/bin/env python3
"""Plot hour-by-hour dispatch trajectory on the cap-binding stress week (2019-W49).

Visualizes Grid-Only, PSM', PPO-Residual, and Shielded PPO over 168 hours:
1. Grid import power vs. the 368.15 kW contract cap (highlighting violations and cap shaving).
2. Battery active power dispatch commands (kW).
3. Battery state of charge (SoC) dynamics and reserve floors.
4. Load shedding profiles illustrating the safety vs. penalty trade-off.
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
from env.smartgrid_env import SmartGridEnv
from env.cap_shield import CapShield
from experiments.train_core import load_agent_for_method, load_config_with_base
from tools.report.plots import fig_w49_dispatch_trajectory

W49_START_IDX = 3360  # 2019-12-02 00:00:00 to 2019-12-08 23:00:00 (168 hours)
EP_LEN = 168
CAP_KW = 368.153918


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
    out_dir = os.path.join(REPO_ROOT, "results", "figures", "battery_dynamics")
    os.makedirs(out_dir, exist_ok=True)
    fig_w49_dispatch_trajectory(out_dir)


if __name__ == "__main__":
    main()
