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

from env.smartgrid_env import NUM_ZONES, SmartGridEnv, default_config
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
    dt_hours: float = 1.0,
) -> Tuple[float, float]:
    """Compute physical battery throughput (MWh) and active SoC percentage."""
    throughput_mwh = float(np.sum(p_ch + p_dis) * dt_hours / 1000.0)

    ep_len = len(p_ch)
    soc = soc_init
    pinned_count = 0
    for t in range(ep_len):
        if soc <= (soc_min + soc_pinned_tol) or soc >= (soc_max - soc_pinned_tol):
            pinned_count += 1
        delta = (p_ch[t] * eta_c - p_dis[t] / eta_d) * dt_hours / c_bat
        soc = float(np.clip(soc + delta, soc_min, soc_max))

    active_soc_pct = float(100.0 * (1.0 - pinned_count / ep_len))
    return throughput_mwh, active_soc_pct


def _resolve_zone_voll(env: SmartGridEnv) -> np.ndarray:
    """Resolve 5-element per-zone VoLL vector from SmartGridEnv."""
    if hasattr(env, "get_zone_voll"):
        return np.asarray(env.get_zone_voll(), dtype=np.float64)
    return np.full(NUM_ZONES, float(env.cost_unserved_eur_kwh), dtype=np.float64)


def _build_common_constraints(
    n_vars: int,
    ep_len: int,
    stride: int,
    idx_u0: int,
    idx_imp: int,
    idx_peak: Optional[int],
    net: np.ndarray,
    c_bat: float,
    eta_c: float,
    eta_d: float,
    dt: float,
    soc_min: float,
    soc_max: float,
    soc_init: float,
) -> Tuple[list, list]:
    """Build power balance, SoC dynamics, and optional peak epigraph constraints (A x <= b)."""
    A_rows = []
    rhs = []

    # 1. Power balance: import + p_dis + pv + sum_z u_{t,z} >= load + p_ch
    #    <=> p_ch - p_dis - sum_z u_{t,z} - import <= -net
    for t in range(ep_len):
        base = stride * t
        row = np.zeros(n_vars)
        row[base + 0] = 1.0
        row[base + 1] = -1.0
        row[base + idx_u0 : base + idx_u0 + NUM_ZONES] = -1.0
        row[base + idx_imp] = -1.0
        A_rows.append(row)
        rhs.append(-float(net[t]))

    # 2. SoC upper and lower limits matching environment dynamics
    for k in range(ep_len):
        row_hi = np.zeros(n_vars)
        row_lo = np.zeros(n_vars)
        for t in range(k + 1):
            base = stride * t
            row_hi[base + 0] = eta_c * dt / c_bat
            row_hi[base + 1] = -dt / (eta_d * c_bat)
            row_lo[base + 0] = -eta_c * dt / c_bat
            row_lo[base + 1] = dt / (eta_d * c_bat)
        A_rows.append(row_hi)
        rhs.append(soc_max - soc_init)
        A_rows.append(row_lo)
        rhs.append(soc_init - soc_min)

    # 3. Weekly peak demand epigraph: grid_import_t - P_peak <= 0
    if idx_peak is not None:
        for t in range(ep_len):
            row = np.zeros(n_vars)
            row[stride * t + idx_imp] = 1.0
            row[idx_peak] = -1.0
            A_rows.append(row)
            rhs.append(0.0)

    return A_rows, rhs


def solve_week_lp(env: SmartGridEnv, start_idx: int) -> Dict[str, Any]:
    """Solve perfect-foresight LP for a single episode."""
    ep_len = env.episode_length
    s0 = start_idx
    s1 = s0 + ep_len

    zone_loads = env._zone_loads[s0:s1].astype(np.float64)
    load = zone_loads.sum(axis=1)
    pv = env._pv_kw_series[s0:s1]
    price_kwh = np.maximum(0.0, env._price[s0:s1]) / 1000.0
    net = load - pv

    defaults = default_config()
    cap = env.grid_import_cap_kw
    c_bat = env.bat_capacity_kwh
    p_max = env.bat_max_power_kw
    eta_c = env.bat_eta_charge
    eta_d = env.bat_eta_discharge
    soc_min = env.bat_soc_min
    soc_max = env.bat_soc_max
    soc_init = env.bat_soc_init
    dt = float(getattr(env, "dt_hours", defaults["dt_hours"]))
    w_wear = env.cost_wear_eur_kwh
    zone_voll = _resolve_zone_voll(env)
    demand_rate = float(getattr(env, "demand_charge_eur_per_kw_week", defaults["demand_charge_eur_per_kw_week"]))
    soc_tol = float(getattr(env, "soc_pinned_tolerance", defaults["soc_pinned_tolerance"]))

    # Variables per step: [p_ch, p_dis, u_0..u_4, grid_import] (+ optional P_peak)
    idx_u0 = 2
    idx_imp = 2 + NUM_ZONES
    stride = 3 + NUM_ZONES
    has_demand = demand_rate > 0.0
    n_vars = stride * ep_len + (1 if has_demand else 0)
    idx_peak = stride * ep_len if has_demand else None

    c = np.zeros(n_vars)
    bounds = []
    for t in range(ep_len):
        base = stride * t
        c[base + 0] = w_wear * dt
        c[base + 1] = w_wear * dt
        c[base + idx_u0 : base + idx_u0 + NUM_ZONES] = zone_voll * dt
        c[base + idx_imp] = price_kwh[t] * dt
        bounds.append((0.0, p_max))
        bounds.append((0.0, p_max))
        for z in range(NUM_ZONES):
            bounds.append((0.0, float(zone_loads[t, z])))
        bounds.append((0.0, cap))

    if idx_peak is not None:
        c[idx_peak] = demand_rate
        bounds.append((0.0, cap))

    A_ub, b_ub = _build_common_constraints(
        n_vars=n_vars,
        ep_len=ep_len,
        stride=stride,
        idx_u0=idx_u0,
        idx_imp=idx_imp,
        idx_peak=idx_peak,
        net=net,
        c_bat=c_bat,
        eta_c=eta_c,
        eta_d=eta_d,
        dt=dt,
        soc_min=soc_min,
        soc_max=soc_max,
        soc_init=soc_init,
    )

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed at start_idx={start_idx}: {res.message}")

    x = res.x[: stride * ep_len].reshape(ep_len, stride)
    p_ch = x[:, 0]
    p_dis = x[:, 1]
    u_zones = x[:, idx_u0 : idx_u0 + NUM_ZONES]
    grid_imp = x[:, idx_imp]

    energy_cost = float(np.sum(grid_imp * price_kwh) * dt)
    unserved_cost = float(np.sum(u_zones * zone_voll[None, :]) * dt)
    wear_cost = float(np.sum((p_ch + p_dis) * w_wear) * dt)
    week_peak_kw = float(np.max(grid_imp)) if len(grid_imp) > 0 else 0.0
    demand_charge_cost = float(demand_rate * week_peak_kw)
    total_cost = energy_cost + unserved_cost + wear_cost + demand_charge_cost

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
        dt_hours=dt,
    )

    return {
        "cost_eur": round(total_cost, 2),
        "energy_cost_eur": round(energy_cost, 2),
        "unserved_cost_eur": round(unserved_cost, 2),
        "wear_cost_eur": round(wear_cost, 2),
        "demand_charge_cost_eur": round(demand_charge_cost, 2),
        "week_peak_grid_kw": round(week_peak_kw, 3),
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

    Decision variables per timestep $t \in [0, \dots, T-1]$:
    - 0: p_ch [0, P_max] (kW)
    - 1: p_dis [0, P_max] (kW)
    - 2..6: u_{t, z} [0, L_{t, z}] (kW, per-zone unserved matching ZONE_NAMES; zone 4 is industrial)
    - 7: z_ind in {0, 1} (binary motor trip indicator: 0 -> run <= 30% shed, 1 -> 100% trip)
    - 8: grid_import [0, P_cap] (kW)
    - 9 (optional, when industrial_trip_penalty_eur > 0): y_trip in [0, 1] (0->1 trip transition)
    Plus optional continuous epigraph peak P_peak when demand_charge_eur_per_kw_week > 0.
    """
    ep_len = env.episode_length
    s0 = start_idx
    s1 = s0 + ep_len

    zone_loads = env._zone_loads[s0:s1].astype(np.float64)
    load_ind = zone_loads[:, 4]
    load_tot = zone_loads.sum(axis=1)
    pv = env._pv_kw_series[s0:s1]
    price_kwh = np.maximum(0.0, env._price[s0:s1]) / 1000.0
    net = load_tot - pv

    defaults = default_config()
    cap = env.grid_import_cap_kw
    c_bat = env.bat_capacity_kwh
    p_max = env.bat_max_power_kw
    eta_c = env.bat_eta_charge
    eta_d = env.bat_eta_discharge
    soc_min = env.bat_soc_min
    soc_max = env.bat_soc_max
    soc_init = env.bat_soc_init
    dt = float(getattr(env, "dt_hours", defaults["dt_hours"]))
    w_wear = env.cost_wear_eur_kwh
    zone_voll = _resolve_zone_voll(env)
    motor_min = env.motor_min_pct
    demand_rate = float(getattr(env, "demand_charge_eur_per_kw_week", defaults["demand_charge_eur_per_kw_week"]))
    trip_penalty_eur = float(getattr(env, "industrial_trip_penalty_eur", defaults["industrial_trip_penalty_eur"]))
    soc_tol = float(getattr(env, "soc_pinned_tolerance", defaults["soc_pinned_tolerance"]))

    has_demand = demand_rate > 0.0
    has_trip_pen = trip_penalty_eur > 0.0

    idx_u0 = 2
    idx_u_ind = idx_u0 + 4
    idx_z = 2 + NUM_ZONES
    idx_imp = 3 + NUM_ZONES
    idx_trip = 4 + NUM_ZONES
    stride = 4 + NUM_ZONES + (1 if has_trip_pen else 0)
    n_vars = stride * ep_len + (1 if has_demand else 0)
    idx_peak = stride * ep_len if has_demand else None

    c = np.zeros(n_vars)
    integrality = np.zeros(n_vars)
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)

    for t in range(ep_len):
        base = stride * t
        c[base + 0] = w_wear * dt
        c[base + 1] = w_wear * dt
        c[base + idx_u0 : base + idx_u0 + NUM_ZONES] = zone_voll * dt
        c[base + idx_imp] = price_kwh[t] * dt
        integrality[base + idx_z] = 1  # z_ind is binary

        ub[base + 0] = p_max
        ub[base + 1] = p_max
        ub[base + idx_u0 : base + idx_u0 + NUM_ZONES] = zone_loads[t]
        ub[base + idx_z] = 1.0 if load_ind[t] > 0.0 else 0.0
        ub[base + idx_imp] = cap

        if has_trip_pen:
            c[base + idx_trip] = trip_penalty_eur
            ub[base + idx_trip] = 1.0

    if idx_peak is not None:
        c[idx_peak] = demand_rate
        ub[idx_peak] = cap

    bounds = Bounds(lb, ub)

    A_rows, rhs = _build_common_constraints(
        n_vars=n_vars,
        ep_len=ep_len,
        stride=stride,
        idx_u0=idx_u0,
        idx_imp=idx_imp,
        idx_peak=idx_peak,
        net=net,
        c_bat=c_bat,
        eta_c=eta_c,
        eta_d=eta_d,
        dt=dt,
        soc_min=soc_min,
        soc_max=soc_max,
        soc_init=soc_init,
    )

    # Piecewise industrial motor operational constraints:
    # u_ind - (motor_min * L_ind) * z_ind <= (1 - motor_min) * L_ind
    # -u_ind + L_ind * z_ind <= 0
    for t in range(ep_len):
        base = stride * t
        l_i = float(load_ind[t])

        row_ub = np.zeros(n_vars)
        row_ub[base + idx_u_ind] = 1.0
        row_ub[base + idx_z] = -motor_min * l_i
        A_rows.append(row_ub)
        rhs.append((1.0 - motor_min) * l_i)

        row_lb = np.zeros(n_vars)
        row_lb[base + idx_u_ind] = -1.0
        row_lb[base + idx_z] = l_i
        A_rows.append(row_lb)
        rhs.append(0.0)

    # Industrial trip transition constraints: y_trip_t >= z_ind_t - z_ind_{t-1}
    if has_trip_pen:
        for t in range(ep_len):
            base = stride * t
            row_tr = np.zeros(n_vars)
            row_tr[base + idx_z] = 1.0
            row_tr[base + idx_trip] = -1.0
            if t > 0:
                row_tr[stride * (t - 1) + idx_z] = -1.0
            A_rows.append(row_tr)
            rhs.append(0.0)

    lhs = [-np.inf] * len(rhs)
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

    x = res.x[: stride * ep_len].reshape(ep_len, stride)
    p_ch = x[:, 0]
    p_dis = x[:, 1]
    u_zones = x[:, idx_u0 : idx_u0 + NUM_ZONES]
    z_ind = x[:, idx_z]
    grid_imp = x[:, idx_imp]

    energy_cost = float(np.sum(grid_imp * price_kwh) * dt)
    unserved_cost = float(np.sum(u_zones * zone_voll[None, :]) * dt)
    wear_cost = float(np.sum((p_ch + p_dis) * w_wear) * dt)
    week_peak_kw = float(np.max(grid_imp)) if len(grid_imp) > 0 else 0.0
    demand_charge_cost = float(demand_rate * week_peak_kw)

    z_bin = (z_ind > 0.5).astype(int)
    trip_transitions = int(z_bin[0] + np.sum(np.maximum(0, z_bin[1:] - z_bin[:-1]))) if len(z_bin) > 0 else 0
    trip_penalty_cost = float(trip_penalty_eur * trip_transitions)
    total_cost = energy_cost + unserved_cost + wear_cost + demand_charge_cost + trip_penalty_cost

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
        dt_hours=dt,
    )

    tripped_hours = int(np.sum(z_bin))

    return {
        "cost_eur": round(total_cost, 2),
        "energy_cost_eur": round(energy_cost, 2),
        "unserved_cost_eur": round(unserved_cost, 2),
        "wear_cost_eur": round(wear_cost, 2),
        "demand_charge_cost_eur": round(demand_charge_cost, 2),
        "week_peak_grid_kw": round(week_peak_kw, 3),
        "trip_penalty_cost_eur": round(trip_penalty_cost, 2),
        "trip_transitions": trip_transitions,
        "mean_exceedance_kw": 0.0,
        "max_exceedance_kw": 0.0,
        "violation_hours": 0,
        "battery_throughput_mwh": round(throughput_mwh, 3),
        "active_soc_pct": round(active_soc_pct, 2),
        "tripped_hours": tripped_hours,
    }


def compute_split_lp(
    split: str = "val",
    env_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Solve LP' across all weeks of a split."""
    cfg: Dict[str, Any] = {"split": split, "charge_clamp": True}
    if env_config:
        cfg.update(env_config)
    env = SmartGridEnv(cfg)
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len

    tot_cost = 0.0
    tot_energy = 0.0
    tot_unserved = 0.0
    tot_wear = 0.0
    tot_demand = 0.0
    tot_tp = 0.0
    soc_pct_list = []

    for w in range(n_weeks):
        res = solve_week_lp(env, w * ep_len)
        tot_cost += res["cost_eur"]
        tot_energy += res["energy_cost_eur"]
        tot_unserved += res["unserved_cost_eur"]
        tot_wear += res["wear_cost_eur"]
        tot_demand += res["demand_charge_cost_eur"]
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
        "demand_charge_cost_eur": round(tot_demand, 2),
        "mean_exceedance_kw": 0.0,
        "violation_rate_pct": 0.0,
        "battery_throughput_mwh": round(tot_tp, 3),
        "active_soc_pct": round(mean_active_soc, 2),
    }


def compute_split_milp(
    split: str = "val",
    mip_rel_gap: Optional[float] = None,
    time_limit: Optional[float] = None,
    env_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Solve MILP' across all weeks of a split."""
    cfg: Dict[str, Any] = {"split": split, "charge_clamp": True}
    if env_config:
        cfg.update(env_config)
    env = SmartGridEnv(cfg)
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len

    tot_cost = 0.0
    tot_energy = 0.0
    tot_unserved = 0.0
    tot_wear = 0.0
    tot_demand = 0.0
    tot_trip_pen = 0.0
    tot_tp = 0.0
    tot_tripped = 0
    tot_trip_transitions = 0
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
        tot_demand += res["demand_charge_cost_eur"]
        tot_trip_pen += res["trip_penalty_cost_eur"]
        tot_tp += res["battery_throughput_mwh"]
        tot_tripped += res.get("tripped_hours", 0)
        tot_trip_transitions += res.get("trip_transitions", 0)
        soc_pct_list.append(res["active_soc_pct"])

    mean_active_soc = float(np.mean(soc_pct_list)) if soc_pct_list else 0.0

    return {
        "split": split,
        "n_weeks": n_weeks,
        "cost_eur": round(tot_cost, 2),
        "energy_cost_eur": round(tot_energy, 2),
        "unserved_cost_eur": round(tot_unserved, 2),
        "wear_cost_eur": round(tot_wear, 2),
        "demand_charge_cost_eur": round(tot_demand, 2),
        "trip_penalty_cost_eur": round(tot_trip_pen, 2),
        "mean_exceedance_kw": 0.0,
        "violation_rate_pct": 0.0,
        "battery_throughput_mwh": round(tot_tp, 3),
        "active_soc_pct": round(mean_active_soc, 2),
        "tripped_hours": tot_tripped,
        "trip_transitions": tot_trip_transitions,
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
