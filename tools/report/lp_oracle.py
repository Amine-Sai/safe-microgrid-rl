#!/usr/bin/env python3
"""Hindsight linear programming (LP') and mixed-integer (MILP') offline oracles.

LP' and MILP' are sums of per-week perfect-foresight oracles: each week is solved
independently with SoC reset to 0.50 at week start.
- LP': continuous relaxation using scipy.optimize.linprog (HiGHS solver).
- MILP': exact offline oracle via scipy.optimize.milp (HiGHS solver) enforcing the
  non-linear piecewise industrial motor operational constraint:
  s_ind in [0.0, 0.30] U {1.0} (i.e. partial shed <= 30% or full emergency shutdown/trip).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.smartgrid_env import SmartGridEnv
from tools.report.load_results import (
    DEFAULT_G2_COST_MULTIPLIER,
    compute_g2_cost_gate,
    load_baselines,
)


def _compute_active_soc_and_throughput(
    p_ch: np.ndarray,
    p_dis: np.ndarray,
    c_bat: float,
    eta_c: float,
    eta_d: float,
    soc_min: float,
    soc_max: float,
    soc_init: float,
    soc_pinned_tol: float = 0.005,
) -> Tuple[float, float]:
    """Compute physical battery throughput (MWh) and active SoC percentage."""
    throughput_mwh = float(np.sum(p_ch + p_dis) / 1000.0)

    ep_len = len(p_ch)
    soc = soc_init
    pinned_count = 0
    for t in range(ep_len):
        if soc <= (soc_min + soc_pinned_tol) or soc >= (soc_max - soc_pinned_tol):
            pinned_count += 1
        delta = (p_ch[t] * eta_c - p_dis[t] / eta_d) / c_bat
        soc = float(np.clip(soc + delta, soc_min, soc_max))

    active_soc_pct = float(100.0 * (1.0 - pinned_count / ep_len))
    return throughput_mwh, active_soc_pct


def solve_week_lp(env: SmartGridEnv, start_idx: int) -> Dict[str, Any]:
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
    soc_tol = getattr(env, "soc_pinned_tolerance", 0.005)

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
        for t in range(k + 1):
            row_hi[4 * t + 0] = eta_c / c_bat
            row_hi[4 * t + 1] = -1.0 / (eta_d * c_bat)
            row_lo[4 * t + 0] = -eta_c / c_bat
            row_lo[4 * t + 1] = 1.0 / (eta_d * c_bat)

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

    throughput_mwh, active_soc_pct = _compute_active_soc_and_throughput(
        p_ch=p_ch,
        p_dis=p_dis,
        c_bat=c_bat,
        eta_c=eta_c,
        eta_d=eta_d,
        soc_min=soc_min,
        soc_max=soc_max,
        soc_init=soc_init,
        soc_pinned_tol=soc_tol,
    )

    return {
        "cost_eur": round(total_cost, 2),
        "energy_cost_eur": round(energy_cost, 2),
        "unserved_cost_eur": round(unserved_cost, 2),
        "wear_cost_eur": round(wear_cost, 2),
        "mean_exceedance_kw": 0.0,
        "max_exceedance_kw": 0.0,
        "violation_hours": 0,
        "battery_throughput_mwh": round(throughput_mwh, 3),
        "active_soc_pct": round(active_soc_pct, 2),
    }


def solve_week_milp(
    env: SmartGridEnv,
    start_idx: int,
    mip_rel_gap: Optional[float] = None,
    time_limit: Optional[float] = None,
) -> Dict[str, Any]:
    r"""Solve exact mixed-integer offline oracle (MILP') for a single episode.

    Formulates the non-linear piecewise industrial motor operational constraint:
    $s_{\text{ind}} \in [0.0, 1 - f_{\text{motor-min}}] \cup \{1.0\}$.
    With default motor_min_pct = 0.70, $u_{\text{ind}} \in [0.0, 0.30 \cdot L_{\text{ind}}] \cup \{L_{\text{ind}}\}$.

    Decision variables per timestep $t \in [0, \dots, T-1]$ (6 variables per step):
    - 0: p_ch [0, P_max] (kW)
    - 1: p_dis [0, P_max] (kW)
    - 2: u_other [0, L_other] (kW, aggregate unserved over zones 0..3)
    - 3: u_ind [0, L_ind] (kW, industrial unserved)
    - 4: z_ind in {0, 1} (binary motor trip indicator: 0 -> run <= 30% shed, 1 -> 100% trip)
    - 5: grid_import [0, P_cap] (kW)
    """
    ep_len = env.episode_length
    s0 = start_idx
    s1 = s0 + ep_len

    zone_loads = env._zone_loads[s0:s1]
    load_other = zone_loads[:, :4].sum(axis=1)
    load_ind = zone_loads[:, 4]
    load_tot = zone_loads.sum(axis=1)
    pv = env._pv_kw_series[s0:s1]
    price_kwh = np.maximum(0.0, env._price[s0:s1]) / 1000.0
    net = load_tot - pv

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
    motor_min = env.motor_min_pct
    soc_tol = getattr(env, "soc_pinned_tolerance", 0.005)

    n_vars = 6 * ep_len
    c = np.zeros(n_vars)
    integrality = np.zeros(n_vars)
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)

    for t in range(ep_len):
        base = 6 * t
        c[base : base + 6] = [w_wear, w_wear, w_uns, w_uns, 0.0, price_kwh[t]]
        integrality[base + 4] = 1  # z_ind is binary
        lb[base : base + 6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ub[base : base + 6] = [
            p_max,
            p_max,
            float(load_other[t]),
            float(load_ind[t]),
            1.0,
            cap,
        ]

    bounds = Bounds(lb, ub)

    A_rows = []
    lhs = []
    rhs = []

    # 1. Power balance: import + p_dis + pv + u_other + u_ind >= load_tot + p_ch
    for t in range(ep_len):
        base = 6 * t
        row = np.zeros(n_vars)
        row[base : base + 6] = [1.0, -1.0, -1.0, -1.0, 0.0, -1.0]
        A_rows.append(row)
        lhs.append(-np.inf)
        rhs.append(-float(net[t]))

    # 2. SoC upper and lower limits matching battery dynamics
    for k in range(ep_len):
        row_hi = np.zeros(n_vars)
        row_lo = np.zeros(n_vars)
        for t in range(k + 1):
            base = 6 * t
            row_hi[base + 0] = eta_c / c_bat
            row_hi[base + 1] = -1.0 / (eta_d * c_bat)
            row_lo[base + 0] = -eta_c / c_bat
            row_lo[base + 1] = 1.0 / (eta_d * c_bat)

        A_rows.append(row_hi)
        lhs.append(-np.inf)
        rhs.append(soc_max - soc_init)
        A_rows.append(row_lo)
        lhs.append(-np.inf)
        rhs.append(soc_init - soc_min)

    # 3. Piecewise industrial motor operational constraints:
    # Upper bound: u_ind <= (1 - motor_min) * L_ind + motor_min * L_ind * z_ind
    #   <=> u_ind - (motor_min * L_ind) * z_ind <= (1 - motor_min) * L_ind
    # Lower bound: u_ind >= L_ind * z_ind
    #   <=> -u_ind + L_ind * z_ind <= 0
    for t in range(ep_len):
        base = 6 * t
        l_i = float(load_ind[t])

        row_ub = np.zeros(n_vars)
        row_ub[base + 3] = 1.0
        row_ub[base + 4] = -motor_min * l_i
        A_rows.append(row_ub)
        lhs.append(-np.inf)
        rhs.append((1.0 - motor_min) * l_i)

        row_lb = np.zeros(n_vars)
        row_lb[base + 3] = -1.0
        row_lb[base + 4] = l_i
        A_rows.append(row_lb)
        lhs.append(-np.inf)
        rhs.append(0.0)

    constraints = LinearConstraint(np.array(A_rows), lhs, rhs)

    options: Dict[str, Any] = {}
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = float(mip_rel_gap)
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options if options else None,
    )

    if not res.success and (res.x is None or res.status not in (0, 1)):
        raise RuntimeError(f"MILP failed at start_idx={start_idx}: {res.message} (status={res.status})")

    x = res.x
    p_ch = x[0::6]
    p_dis = x[1::6]
    u_other = x[2::6]
    u_ind = x[3::6]
    z_ind = x[4::6]
    grid_imp = x[5::6]

    energy_cost = float(np.sum(grid_imp * price_kwh))
    unserved_cost = float(np.sum((u_other + u_ind) * w_uns))
    wear_cost = float(np.sum((p_ch + p_dis) * w_wear))
    total_cost = energy_cost + unserved_cost + wear_cost

    throughput_mwh, active_soc_pct = _compute_active_soc_and_throughput(
        p_ch=p_ch,
        p_dis=p_dis,
        c_bat=c_bat,
        eta_c=eta_c,
        eta_d=eta_d,
        soc_min=soc_min,
        soc_max=soc_max,
        soc_init=soc_init,
        soc_pinned_tol=soc_tol,
    )

    tripped_hours = int(np.sum(z_ind > 0.5))

    return {
        "cost_eur": round(total_cost, 2),
        "energy_cost_eur": round(energy_cost, 2),
        "unserved_cost_eur": round(unserved_cost, 2),
        "wear_cost_eur": round(wear_cost, 2),
        "mean_exceedance_kw": 0.0,
        "max_exceedance_kw": 0.0,
        "violation_hours": 0,
        "battery_throughput_mwh": round(throughput_mwh, 3),
        "active_soc_pct": round(active_soc_pct, 2),
        "tripped_hours": tripped_hours,
    }


def compute_split_lp(split: str = "val") -> Dict[str, Any]:
    """Solve LP' across all weeks of a split."""
    env = SmartGridEnv({"split": split, "charge_clamp": True})
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len

    tot_cost = 0.0
    tot_energy = 0.0
    tot_unserved = 0.0
    tot_wear = 0.0
    tot_tp = 0.0
    soc_pct_list = []

    for w in range(n_weeks):
        res = solve_week_lp(env, w * ep_len)
        tot_cost += res["cost_eur"]
        tot_energy += res["energy_cost_eur"]
        tot_unserved += res["unserved_cost_eur"]
        tot_wear += res["wear_cost_eur"]
        tot_tp += res["battery_throughput_mwh"]
        soc_pct_list.append(res["active_soc_pct"])

    mean_active_soc = float(np.mean(soc_pct_list)) if soc_pct_list else 0.0

    return {
        "split": split,
        "n_weeks": n_weeks,
        "cost_eur": round(tot_cost, 2),
        "energy_cost_eur": round(tot_energy, 2),
        "unserved_cost_eur": round(tot_unserved, 2),
        "wear_cost_eur": round(tot_wear, 2),
        "mean_exceedance_kw": 0.0,
        "violation_rate_pct": 0.0,
        "battery_throughput_mwh": round(tot_tp, 3),
        "active_soc_pct": round(mean_active_soc, 2),
    }


def compute_split_milp(
    split: str = "val",
    mip_rel_gap: Optional[float] = None,
    time_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve MILP' across all weeks of a split."""
    env = SmartGridEnv({"split": split, "charge_clamp": True})
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len

    tot_cost = 0.0
    tot_energy = 0.0
    tot_unserved = 0.0
    tot_wear = 0.0
    tot_tp = 0.0
    tot_tripped = 0
    soc_pct_list = []

    for w in range(n_weeks):
        res = solve_week_milp(
            env,
            w * ep_len,
            mip_rel_gap=mip_rel_gap,
            time_limit=time_limit,
        )
        tot_cost += res["cost_eur"]
        tot_energy += res["energy_cost_eur"]
        tot_unserved += res["unserved_cost_eur"]
        tot_wear += res["wear_cost_eur"]
        tot_tp += res["battery_throughput_mwh"]
        tot_tripped += res.get("tripped_hours", 0)
        soc_pct_list.append(res["active_soc_pct"])

    mean_active_soc = float(np.mean(soc_pct_list)) if soc_pct_list else 0.0

    return {
        "split": split,
        "n_weeks": n_weeks,
        "cost_eur": round(tot_cost, 2),
        "energy_cost_eur": round(tot_energy, 2),
        "unserved_cost_eur": round(tot_unserved, 2),
        "wear_cost_eur": round(tot_wear, 2),
        "mean_exceedance_kw": 0.0,
        "violation_rate_pct": 0.0,
        "battery_throughput_mwh": round(tot_tp, 3),
        "active_soc_pct": round(mean_active_soc, 2),
        "tripped_hours": tot_tripped,
    }


def update_constants_and_baselines(val_res: Dict[str, Any], test_res: Dict[str, Any]):
    """Write exact LP oracle values into project tables."""
    tables_dir = os.path.join(REPO_ROOT, "results", "tables")

    # Update frozen_baselines.csv
    fb_path = os.path.join(tables_dir, "frozen_baselines.csv")
    rows = []
    fieldnames = None
    if os.path.exists(fb_path):
        with open(fb_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for r in reader:
                if r.get("method") != "LP'":
                    rows.append(r)

    if fieldnames is None:
        fieldnames = ["method", "val_cost_eur", "mean_exceedance_kw", "feasible"]

    lp_row = {fn: "" for fn in fieldnames}
    lp_row.update({
        "method": "LP'",
        "val_cost_eur": f"{val_res['cost_eur']:.2f}",
        "mean_exceedance_kw": "0.0000",
        "feasible": "True",
        "energy_purchase_eur": f"{val_res['energy_cost_eur']:.2f}",
        "battery_wear_eur": f"{val_res['wear_cost_eur']:.2f}",
        "unserved_penalty_eur": f"{val_res['unserved_cost_eur']:.2f}",
        "battery_throughput_mwh": f"{val_res.get('battery_throughput_mwh', 0.0):.3f}",
        "active_soc_pct": f"{val_res.get('active_soc_pct', 0.0):.2f}",
    })

    rows.append(lp_row)

    with open(fb_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
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

    baselines = load_baselines()
    psm_val = float(baselines.get("PSM'", {}).get("cost_eur", 55530.22))
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

    canonical_g2 = compute_g2_cost_gate(psm_val, multiplier=DEFAULT_G2_COST_MULTIPLIER)
    if "g2_cost_eur" in constants:
        constants["g2_cost_eur"]["value"] = f"{canonical_g2:.2f}"
        constants["g2_cost_eur"]["source"] = (
            f"{DEFAULT_G2_COST_MULTIPLIER:.2f} x PSM' Val cost (tools/report/load_results.py)"
        )
    if "g2_cost_multiplier" in constants:
        constants["g2_cost_multiplier"]["value"] = f"{DEFAULT_G2_COST_MULTIPLIER:.2f}"
        constants["g2_cost_multiplier"]["source"] = (
            f"G2 gate = {DEFAULT_G2_COST_MULTIPLIER:.2f} x PSM' Val cost"
        )

    with open(pc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "value", "unit", "source"])
        w.writeheader()
        w.writerows(constants.values())
    print(f"Updated {pc_path}")


def main():
    parser = argparse.ArgumentParser(description="Hindsight LP and MILP oracle solver.")
    parser.add_argument("--oracle", choices=["lp", "milp", "both"], default="lp")
    parser.add_argument("--split", choices=["val", "test", "all"], default="all")
    parser.add_argument("--mip-rel-gap", type=float, default=None, help="Relative MIP optimality gap.")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds per week.")
    parser.add_argument("--update-tables", action="store_true", help="Update CSV tables.")
    args = parser.parse_args()

    if args.oracle in ("lp", "both"):
        val_res = compute_split_lp("val") if args.split in ("val", "all") else None
        test_res = compute_split_lp("test") if args.split in ("test", "all") else None

        if val_res:
            print(
                f"LP' Val (26 weeks):  Cost = {val_res['cost_eur']:,.2f} EUR "
                f"(Energy: {val_res['energy_cost_eur']:,.2f}, Unserved: {val_res['unserved_cost_eur']:,.2f}, "
                f"Wear: {val_res['wear_cost_eur']:,.2f}, Throughput: {val_res['battery_throughput_mwh']:.3f} MWh, "
                f"Active SoC: {val_res['active_soc_pct']:.2f}%)"
            )
        if test_res:
            print(
                f"LP' Test (24 weeks): Cost = {test_res['cost_eur']:,.2f} EUR "
                f"(Energy: {test_res['energy_cost_eur']:,.2f}, Unserved: {test_res['unserved_cost_eur']:,.2f}, "
                f"Wear: {test_res['wear_cost_eur']:,.2f}, Throughput: {test_res['battery_throughput_mwh']:.3f} MWh, "
                f"Active SoC: {test_res['active_soc_pct']:.2f}%)"
            )

        if args.update_tables and val_res and test_res:
            update_constants_and_baselines(val_res, test_res)

    if args.oracle in ("milp", "both"):
        val_milp = (
            compute_split_milp("val", mip_rel_gap=args.mip_rel_gap, time_limit=args.time_limit)
            if args.split in ("val", "all")
            else None
        )
        test_milp = (
            compute_split_milp("test", mip_rel_gap=args.mip_rel_gap, time_limit=args.time_limit)
            if args.split in ("test", "all")
            else None
        )

        if val_milp:
            print(
                f"MILP' Val (26 weeks):  Cost = {val_milp['cost_eur']:,.2f} EUR "
                f"(Energy: {val_milp['energy_cost_eur']:,.2f}, Unserved: {val_milp['unserved_cost_eur']:,.2f}, "
                f"Wear: {val_milp['wear_cost_eur']:,.2f}, Throughput: {val_milp['battery_throughput_mwh']:.3f} MWh, "
                f"Active SoC: {val_milp['active_soc_pct']:.2f}%, Tripped: {val_milp['tripped_hours']}h)"
            )
        if test_milp:
            print(
                f"MILP' Test (24 weeks): Cost = {test_milp['cost_eur']:,.2f} EUR "
                f"(Energy: {test_milp['energy_cost_eur']:,.2f}, Unserved: {test_milp['unserved_cost_eur']:,.2f}, "
                f"Wear: {test_milp['wear_cost_eur']:,.2f}, Throughput: {test_milp['battery_throughput_mwh']:.3f} MWh, "
                f"Active SoC: {test_milp['active_soc_pct']:.2f}%, Tripped: {test_milp['tripped_hours']}h)"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
