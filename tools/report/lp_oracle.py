#!/usr/bin/env python3
"""Hindsight linear programming oracle (LP') for the microgrid benchmark.

Computes the optimal physical dispatch with perfect foresight under the import
cap using scipy.optimize.linprog (HiGHS solver).
"""

import argparse
import csv
import os
import sys
import numpy as np
from scipy.optimize import linprog

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from env.smartgrid_env import SmartGridEnv


def solve_week_lp(env: SmartGridEnv, start_idx: int) -> dict:
    """Solve perfect-foresight LP for a single episode."""
    ep_len = env.episode_length
    s0 = start_idx
    s1 = s0 + ep_len

    load = env._zone_loads[s0:s1].sum(axis=1)
    pv = env._pv_kw_series[s0:s1]
    price_kwh = np.maximum(0.0, env._price[s0:s1]) / 1000.0
    net = load - pv

    cap = env.grid_import_cap_kw
    c_bat = env.bat_capacity_kwh
    p_max = env.bat_max_power_kw
    eta_c = env.bat_eta_charge
    eta_d = env.bat_eta_discharge
    soc_min = env.bat_soc_min
    soc_max = env.bat_soc_max
    soc_init = env.bat_soc_init
    w_wear = env.cost_wear_eur_kwh
    w_uns = env.cost_unserved_eur_kwh

    # Variables per step: [p_ch, p_dis, unserved, grid_import]
    c = np.zeros(4 * ep_len)
    bounds = []
    for t in range(ep_len):
        c[4 * t : 4 * t + 4] = [w_wear, w_wear, w_uns, price_kwh[t]]
        bounds.extend([(0.0, p_max), (0.0, p_max), (0.0, float(load[t])), (0.0, cap)])

    A_ub = []
    b_ub = []

    # Power balance: import + p_dis + pv + unserved >= load + p_ch
    for t in range(ep_len):
        row = np.zeros(4 * ep_len)
        row[4 * t : 4 * t + 4] = [1.0, -1.0, -1.0, -1.0]
        A_ub.append(row)
        b_ub.append(-float(net[t]))

    # SoC limits and rate bounds matching environment dynamics
    for k in range(ep_len):
        row_hi = np.zeros(4 * ep_len)
        row_lo = np.zeros(4 * ep_len)
        for t in range(k):
            row_hi[4 * t + 0] = eta_c / c_bat
            row_hi[4 * t + 1] = -1.0 / (eta_d * c_bat)
            row_lo[4 * t + 0] = -eta_c / c_bat
            row_lo[4 * t + 1] = 1.0 / (eta_d * c_bat)

        row_hi[4 * k + 0] = eta_c / c_bat
        row_lo[4 * k + 1] = 1.0 / (eta_d * c_bat)
        A_ub.append(row_hi)
        b_ub.append(soc_max - soc_init)
        A_ub.append(row_lo)
        b_ub.append(soc_init - soc_min)

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed at start_idx={start_idx}: {res.message}")

    x = res.x
    p_ch = x[0::4]
    p_dis = x[1::4]
    unserved = x[2::4]
    grid_imp = x[3::4]

    energy_cost = float(np.sum(grid_imp * price_kwh))
    unserved_cost = float(np.sum(unserved * w_uns))
    wear_cost = float(np.sum((p_ch + p_dis) * w_wear))
    total_cost = energy_cost + unserved_cost + wear_cost

    return {
        "cost_eur": round(total_cost, 2),
        "energy_cost_eur": round(energy_cost, 2),
        "unserved_cost_eur": round(unserved_cost, 2),
        "wear_cost_eur": round(wear_cost, 2),
        "mean_exceedance_kw": 0.0,
        "max_exceedance_kw": 0.0,
        "violation_hours": 0,
    }


def compute_split_lp(split: str = "val") -> dict:
    """Solve LP oracle across all episodes of a split."""
    env = SmartGridEnv({"split": split, "charge_clamp": True})
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len

    tot_cost = 0.0
    tot_energy = 0.0
    tot_unserved = 0.0
    tot_wear = 0.0

    for w in range(n_weeks):
        res = solve_week_lp(env, w * ep_len)
        tot_cost += res["cost_eur"]
        tot_energy += res["energy_cost_eur"]
        tot_unserved += res["unserved_cost_eur"]
        tot_wear += res["wear_cost_eur"]

    return {
        "split": split,
        "n_weeks": n_weeks,
        "cost_eur": round(tot_cost, 2),
        "energy_cost_eur": round(tot_energy, 2),
        "unserved_cost_eur": round(tot_unserved, 2),
        "wear_cost_eur": round(tot_wear, 2),
        "mean_exceedance_kw": 0.0,
        "violation_rate_pct": 0.0,
    }


def update_constants_and_baselines(val_res: dict, test_res: dict):
    """Write exact LP oracle values into project tables."""
    tables_dir = os.path.join(REPO_ROOT, "results", "tables")

    # Update frozen_baselines.csv
    fb_path = os.path.join(tables_dir, "frozen_baselines.csv")
    rows = []
    if os.path.exists(fb_path):
        with open(fb_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("method") != "LP'":
                    rows.append(r)

    rows.append({
        "method": "LP'",
        "val_cost_eur": f"{val_res['cost_eur']:.2f}",
        "mean_exceedance_kw": "0.0000",
        "feasible": "True",
    })

    with open(fb_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "val_cost_eur", "mean_exceedance_kw", "feasible"])
        w.writeheader()
        w.writerows(rows)
    print(f"Updated {fb_path}")

    # Update project_constants.csv
    pc_path = os.path.join(tables_dir, "project_constants.csv")
    constants = {}
    if os.path.exists(pc_path):
        with open(pc_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                constants[r["name"]] = r

    psm_val = 55530.22
    psm_test = 37674.27
    gap_val = psm_val - val_res["cost_eur"]
    gap_test = psm_test - test_res["cost_eur"]

    constants["lp_prime_val_eur"] = {
        "name": "lp_prime_val_eur",
        "value": f"{val_res['cost_eur']:.2f}",
        "unit": "EUR",
        "source": "tools/report/lp_oracle.py perfect foresight HiGHS solver on Val",
    }
    constants["lp_prime_test_eur"] = {
        "name": "lp_prime_test_eur",
        "value": f"{test_res['cost_eur']:.2f}",
        "unit": "EUR",
        "source": "tools/report/lp_oracle.py perfect foresight HiGHS solver on Test",
    }
    constants["gap_psm_to_lp_eur"] = {
        "name": "gap_psm_to_lp_eur",
        "value": f"{gap_val:.2f}",
        "unit": "EUR",
        "source": "PSM' Val cost minus LP' Val cost",
    }
    constants["gap_psm_to_lp_test_eur"] = {
        "name": "gap_psm_to_lp_test_eur",
        "value": f"{gap_test:.2f}",
        "unit": "EUR",
        "source": "PSM' Test cost minus LP' Test cost",
    }
    # Ensure canonical G2 gate
    constants["g2_cost_eur"]["value"] = "58306.73"

    with open(pc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "value", "unit", "source"])
        w.writeheader()
        w.writerows(constants.values())
    print(f"Updated {pc_path}")


def main():
    parser = argparse.ArgumentParser(description="Hindsight LP oracle solver.")
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--update-tables", action="store_true", help="Update CSV tables.")
    args = parser.parse_args()

    val_res = compute_split_lp("val") if args.split in ("val", "all") else None
    test_res = compute_split_lp("test") if args.split in ("test", "all") else None

    if val_res:
        print(f"LP' Val (26 weeks):  Cost = {val_res['cost_eur']:,.2f} EUR (Energy: {val_res['energy_cost_eur']:,.2f}, Unserved: {val_res['unserved_cost_eur']:,.2f}, Wear: {val_res['wear_cost_eur']:,.2f})")
    if test_res:
        print(f"LP' Test (24 weeks): Cost = {test_res['cost_eur']:,.2f} EUR (Energy: {test_res['energy_cost_eur']:,.2f}, Unserved: {test_res['unserved_cost_eur']:,.2f}, Wear: {test_res['wear_cost_eur']:,.2f})")

    if args.update_tables and val_res and test_res:
        update_constants_and_baselines(val_res, test_res)

    return 0


if __name__ == "__main__":
    sys.exit(main())
