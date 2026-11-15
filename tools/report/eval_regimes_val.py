#!/usr/bin/env python3
"""Deterministic Val (2019 H1, 26 weeks) benchmark across Regime 1, Regime 2 (C&I), and OOD suite.

Evaluates:
  - Regime 1 Val (26 weeks): Grid-Only, PSM', LP', MILP', MPCPolicy (perfect/causal_noisy/persistence, H=24),
    5-seed clamped_ppo_diff_shield (final & best), and 5-seed causal23_ppo_diff_shield (final & best).
  - Regime 2 Val (26 weeks, ci_* tariffs from configs/config.yaml): Grid-Only, PSM', LP', MILP',
    MPCPolicy (perfect/causal_noisy, H=24), and 5-seed ci_ppo_shielded, ci_ppo_diff_shield, ci_ppo_diff_6d.
  - OOD Stress Suite on Val representative stress week (2019-W04, start_idx=336) for
    PSM', MPCPolicy(causal_noisy, H=24), and clamped_ppo_diff_shield (seed 101).

Writes structured results to results/tables/regime_benchmarks_val.csv.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
from agents.mpc import MPCPolicy
from env.cap_shield import CapShield
from env.smartgrid_env import SmartGridEnv
from experiments.train_core import (
    agent_name,
    build_env,
    is_shielded_run,
    load_agent_for_method,
    load_config_with_base,
)
from tools.report.evaluate_test_split import (
    _step_policy_and_measure,
    compute_tail_and_safety_metrics,
    evaluate_ood_stress_suite,
)
from tools.report.lp_oracle import compute_split_lp, compute_split_milp

SEEDS: Tuple[int, ...] = (101, 102, 103, 104, 105)
REGIME2_REF_CONFIG = "ci_ppo_diff_shield"
REGIME2_ENV_KEYS = (
    "demand_charge_eur_per_kw_week",
    "zone_voll_eur_kwh",
    "industrial_trip_penalty_eur",
)
OOD_CONTINGENCIES_ORDER = (
    "pv_inverter_trip",
    "bess_derate",
    "grid_cap_reduction",
    "forecast_noise",
)

CSV_FIELDNAMES = [
    "suite",
    "regime",
    "method",
    "row_type",
    "seed",
    "checkpoint_type",
    "contingency",
    "n_weeks",
    "cost_eur",
    "energy_purchase_eur",
    "battery_wear_eur",
    "unserved_penalty_eur",
    "demand_charge_eur",
    "industrial_trip_eur",
    "mean_exceedance_kw",
    "max_hourly_exceedance_kw",
    "cvar95_cost_eur",
    "cvar95_exceedance_kw",
    "violation_hours",
    "violation_rate_pct",
    "max_consecutive_violation_hours",
    "unserved_pct",
    "battery_throughput_mwh",
    "active_soc_pct",
    "shield_intervention_rate",
    "latency_p50_ms",
    "latency_p99_ms",
]

AGGREGATE_NUMERIC_FIELDS = [
    "cost_eur",
    "energy_purchase_eur",
    "battery_wear_eur",
    "unserved_penalty_eur",
    "demand_charge_eur",
    "industrial_trip_eur",
    "mean_exceedance_kw",
    "max_hourly_exceedance_kw",
    "cvar95_cost_eur",
    "cvar95_exceedance_kw",
    "violation_hours",
    "violation_rate_pct",
    "max_consecutive_violation_hours",
    "unserved_pct",
    "battery_throughput_mwh",
    "active_soc_pct",
    "shield_intervention_rate",
    "latency_p50_ms",
    "latency_p99_ms",
]


def _resolve_val_stress_week_idx(target_week: Optional[str] = None) -> tuple[str, int]:
    """Resolve representative Val stress week (defaults to highest_exceedance from split_report.json)."""
    report_path = os.path.join(REPO_ROOT, "data", "processed", "split_report.json")
    if target_week is None and os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            sr = json.load(f)
        target_week = (
            sr.get("stress_weeks", {}).get("val", {}).get("highest_exceedance", {}).get("week")
        )
    target_week = target_week or "2019-W04"

    df_val = pd.read_parquet(os.path.join(REPO_ROOT, "data", "processed", "val.parquet"))
    iso_labels = (
        df_val.index.isocalendar().year.astype(str)
        + "-W"
        + df_val.index.isocalendar().week.astype(str).str.zfill(2)
    ).values
    matches = np.where(iso_labels == target_week)[0]
    if len(matches) < 168:
        return target_week, 0
    return target_week, int(matches[0])


def _evaluate_rollout_full(
    env: Any,
    start_indices: Sequence[int],
    policy: Any = None,
    agent_wrapper: Any = None,
    action_tol: float = 1e-4,
) -> Dict[str, Any]:
    """Roll out deterministic policy/agent over start_indices with full Regime 1+2 cost & safety telemetry."""
    base_env = getattr(env, "unwrapped", env)
    ep_len = int(getattr(base_env, "episode_length", 168))
    dt_hours = float(getattr(base_env, "dt_hours", 1.0))
    violation_tol = float(getattr(base_env, "violation_tolerance_kw", 1e-4))
    soc_pin_tol = float(getattr(base_env, "soc_pinned_tolerance", 0.005))
    soc_lo = float(base_env.bat_soc_min) + soc_pin_tol
    soc_hi = float(base_env.bat_soc_max) - soc_pin_tol

    total_energy = 0.0
    total_unserved = 0.0
    total_wear = 0.0
    total_demand_charge = 0.0
    total_trip = 0.0
    total_throughput_kwh = 0.0
    unserved_kwh = 0.0
    demand_kwh = 0.0
    violation_hours = 0

    hourly_costs: List[float] = []
    exceedances: List[float] = []
    violation_flags: List[bool] = []
    latencies_ms: List[float] = []
    shield_deltas: List[float] = []
    soc_pins: List[float] = []

    if policy is not None and hasattr(policy, "reset") and callable(policy.reset):
        try:
            policy.reset(env)
        except TypeError:
            policy.reset()

    for start_idx in start_indices:
        obs, _ = env.reset(options={"start_idx": int(start_idx)})
        for _ in range(ep_len):
            soc_pins.append(float(obs[0] <= soc_lo or obs[0] >= soc_hi))
            next_obs, _, lat_ms, s_delta, done, info = _step_policy_and_measure(
                env=env,
                obs=obs,
                policy=policy,
                agent_wrapper=agent_wrapper,
                action_tol=action_tol,
            )
            obs = next_obs

            e_cost = float(info["energy_cost"])
            u_cost = float(info["unserved_cost"])
            w_cost = float(info["battery_wear_cost"])
            dc_cost = float(info.get("cost_demand_charge_eur", 0.0))
            tr_cost = float(info.get("cost_industrial_trip_eur", 0.0))
            step_cost = e_cost + u_cost + w_cost + dc_cost + tr_cost

            total_energy += e_cost
            total_unserved += u_cost
            total_wear += w_cost
            total_demand_charge += dc_cost
            total_trip += tr_cost
            hourly_costs.append(step_cost)

            total_throughput_kwh += abs(float(info.get("battery_power_kw", 0.0))) * dt_hours
            uns_kw = float(info.get("total_unserved_kw", 0.0))
            serv_kw = float(info.get("total_served_kw", 0.0))
            unserved_kwh += uns_kw * dt_hours
            demand_kwh += (serv_kw + uns_kw) * dt_hours

            exc = float(info["grid_exceedance_kw"])
            exceedances.append(exc)
            is_viol = bool(info.get("is_violation", exc > violation_tol))
            violation_flags.append(is_viol)
            if is_viol:
                violation_hours += 1

            latencies_ms.append(lat_ms)
            shield_deltas.append(s_delta)
            if done:
                break

    total_hours = max(1, len(exceedances))
    total_cost = total_energy + total_unserved + total_wear + total_demand_charge + total_trip
    tail = compute_tail_and_safety_metrics(
        hourly_costs=hourly_costs,
        hourly_exceedances=exceedances,
        violation_flags=violation_flags,
        violation_tolerance_kw=violation_tol,
    )
    shield_rate = float(np.mean([1.0 if d > action_tol else 0.0 for d in shield_deltas])) if shield_deltas else 0.0
    lat_p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    lat_p99 = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    return {
        "n_weeks": len(start_indices),
        "cost_eur": round(total_cost, 2),
        "energy_purchase_eur": round(total_energy, 2),
        "battery_wear_eur": round(total_wear, 2),
        "unserved_penalty_eur": round(total_unserved, 2),
        "demand_charge_eur": round(total_demand_charge, 2),
        "industrial_trip_eur": round(total_trip, 2),
        "mean_exceedance_kw": round(float(np.mean(exceedances)) if exceedances else 0.0, 4),
        "max_hourly_exceedance_kw": tail["max_hourly_exceedance_kw"],
        "cvar95_cost_eur": tail["cvar95_cost_eur"],
        "cvar95_exceedance_kw": tail["cvar95_exceedance_kw"],
        "violation_hours": int(violation_hours),
        "violation_rate_pct": round(violation_hours / total_hours * 100.0, 2),
        "max_consecutive_violation_hours": tail["max_consecutive_violation_hours"],
        "unserved_pct": round(unserved_kwh / max(1e-6, demand_kwh) * 100.0, 4),
        "battery_throughput_mwh": round(total_throughput_kwh / 1000.0, 3),
        "active_soc_pct": round(100.0 - (float(np.mean(soc_pins)) * 100.0 if soc_pins else 100.0), 2),
        "shield_intervention_rate": round(shield_rate, 4),
        "latency_p50_ms": round(lat_p50, 4),
        "latency_p99_ms": round(lat_p99, 4),
    }


def _oracle_to_row(suite: str, regime: str, method: str, res: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LP'/MILP' split summary dictionary into canonical CSV row."""
    return {
        "suite": suite,
        "regime": regime,
        "method": method,
        "row_type": "single",
        "seed": "N/A",
        "checkpoint_type": "N/A",
        "contingency": "nominal",
        "n_weeks": res["n_weeks"],
        "cost_eur": res["cost_eur"],
        "energy_purchase_eur": res["energy_cost_eur"],
        "battery_wear_eur": res["wear_cost_eur"],
        "unserved_penalty_eur": res["unserved_cost_eur"],
        "demand_charge_eur": res.get("demand_charge_cost_eur", 0.0),
        "industrial_trip_eur": res.get("trip_penalty_cost_eur", 0.0),
        "mean_exceedance_kw": 0.0,
        "max_hourly_exceedance_kw": 0.0,
        "cvar95_cost_eur": "",
        "cvar95_exceedance_kw": 0.0,
        "violation_hours": 0,
        "violation_rate_pct": 0.0,
        "max_consecutive_violation_hours": 0,
        "unserved_pct": "",
        "battery_throughput_mwh": res["battery_throughput_mwh"],
        "active_soc_pct": res["active_soc_pct"],
        "shield_intervention_rate": "",
        "latency_p50_ms": "",
        "latency_p99_ms": "",
    }


def _summarize_seed_rows(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append mean and std summary rows across a list of per-seed rows."""
    if not seed_rows:
        return []
    base = seed_rows[0]
    mean_row: Dict[str, Any] = {
        "suite": base["suite"],
        "regime": base["regime"],
        "method": base["method"],
        "row_type": "mean",
        "seed": "mean",
        "checkpoint_type": base["checkpoint_type"],
        "contingency": base["contingency"],
        "n_weeks": base["n_weeks"],
    }
    std_row: Dict[str, Any] = {
        "suite": base["suite"],
        "regime": base["regime"],
        "method": base["method"],
        "row_type": "std",
        "seed": "std",
        "checkpoint_type": base["checkpoint_type"],
        "contingency": base["contingency"],
        "n_weeks": base["n_weeks"],
    }
    for k in AGGREGATE_NUMERIC_FIELDS:
        vals = [float(r[k]) for r in seed_rows if r.get(k) not in ("", None)]
        if vals:
            mean_row[k] = round(statistics.fmean(vals), 4)
            std_row[k] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        else:
            mean_row[k] = ""
            std_row[k] = ""
    return [mean_row, std_row]


def _evaluate_rl_seeds(
    suite: str,
    regime: str,
    exp_id: str,
    checkpoint_type: str = "final",
    seeds: Sequence[int] = SEEDS,
) -> List[Dict[str, Any]]:
    """Evaluate 5 seeds of a trained RL experiment on Val (26 weeks)."""
    cfg_base = load_config_with_base(exp_id)
    folder = agent_name(cfg_base["method"], cfg_base)
    seed_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        cfg = dict(cfg_base, seed=seed)
        run_json_path = os.path.join(REPO_ROOT, "results", "runs", folder, f"{exp_id}_seed{seed}.json")
        ckpt_key = "final_checkpoint" if checkpoint_type == "final" else "best_checkpoint"
        if os.path.exists(run_json_path):
            with open(run_json_path, encoding="utf-8") as f:
                run_data = json.load(f)
            ckpt_path = os.path.join(REPO_ROOT, run_data[ckpt_key]["weights"])
        else:
            ckpt_path = os.path.join(REPO_ROOT, "checkpoints", folder, f"{exp_id}_seed{seed}_final.pt")

        env_val = build_env(cfg, split="val", seed=seed + 100)
        if is_shielded_run(cfg["method"], cfg):
            env_val = CapShield(env_val)
        base_env = getattr(env_val, "unwrapped", env_val)
        n_weeks = base_env._n_timesteps // base_env.episode_length
        start_indices = [w * base_env.episode_length for w in range(n_weeks)]

        agent = load_agent_for_method(cfg["method"], cfg)
        agent.load_models(ckpt_path)

        res = _evaluate_rollout_full(env_val, start_indices=start_indices, agent_wrapper=agent)
        seed_rows.append({
            "suite": suite,
            "regime": regime,
            "method": exp_id,
            "row_type": "per_seed",
            "seed": seed,
            "checkpoint_type": checkpoint_type,
            "contingency": "nominal",
            **res,
        })

    return seed_rows + _summarize_seed_rows(seed_rows)


def _evaluate_regime_baselines(
    suite: str,
    regime: str,
    env_cfg: Dict[str, Any],
    mpc_modes: Sequence[str],
    milp_time_limit: float = 2.0,
) -> List[Dict[str, Any]]:
    """Evaluate Grid-Only, PSM', LP', MILP', and MPCPolicy modes on Val (26 weeks)."""
    rows: List[Dict[str, Any]] = []
    base_cfg = {"split": "val", "charge_clamp": True, "seed": 42, **env_cfg}
    env = SmartGridEnv(base_cfg)
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len
    start_indices = [w * ep_len for w in range(n_weeks)]

    for name, pol in (("Grid-Only", GridOnlyPolicy()), ("PSM'", PeakShaveMinPolicy())):
        res = _evaluate_rollout_full(env, start_indices=start_indices, policy=pol)
        rows.append({
            "suite": suite,
            "regime": regime,
            "method": name,
            "row_type": "single",
            "seed": "N/A",
            "checkpoint_type": "N/A",
            "contingency": "nominal",
            **res,
        })

    lp_res = compute_split_lp("val", env_config=env_cfg or None)
    rows.append(_oracle_to_row(suite, regime, "LP'", lp_res))

    milp_res = compute_split_milp("val", time_limit=milp_time_limit, env_config=env_cfg or None)
    rows.append(_oracle_to_row(suite, regime, "MILP'", milp_res))

    for mode in mpc_modes:
        env_mpc = SmartGridEnv(base_cfg)
        pol = MPCPolicy(env=env_mpc, horizon=24, forecast_mode=mode, seed=42)
        res = _evaluate_rollout_full(env_mpc, start_indices=start_indices, policy=pol)
        rows.append({
            "suite": suite,
            "regime": regime,
            "method": f"MPCPolicy({mode}, H=24)",
            "row_type": "single",
            "seed": "42",
            "checkpoint_type": "N/A",
            "contingency": "nominal",
            **res,
        })

    return rows


def _ood_suite_to_rows(
    suite: str,
    method: str,
    seed: Any,
    checkpoint_type: str,
    ood_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert evaluate_ood_stress_suite output dictionary into canonical CSV rows."""
    rows: List[Dict[str, Any]] = []
    for cont, res in ood_results.items():
        rows.append({
            "suite": suite,
            "regime": "regime1",
            "method": method,
            "row_type": "single",
            "seed": seed,
            "checkpoint_type": checkpoint_type,
            "contingency": cont,
            "n_weeks": 1,
            "cost_eur": res.get("cost_eur", ""),
            "energy_purchase_eur": "",
            "battery_wear_eur": "",
            "unserved_penalty_eur": "",
            "demand_charge_eur": 0.0,
            "industrial_trip_eur": 0.0,
            "mean_exceedance_kw": res.get("mean_exceedance_kw", ""),
            "max_hourly_exceedance_kw": res.get("max_hourly_exceedance_kw", ""),
            "cvar95_cost_eur": res.get("cvar95_cost_eur", ""),
            "cvar95_exceedance_kw": res.get("cvar95_exceedance_kw", ""),
            "violation_hours": res.get("violation_hours", ""),
            "violation_rate_pct": res.get("violation_rate_pct", ""),
            "max_consecutive_violation_hours": res.get("max_consecutive_violation_hours", ""),
            "unserved_pct": res.get("unserved_pct", ""),
            "battery_throughput_mwh": "",
            "active_soc_pct": "",
            "shield_intervention_rate": res.get("shield_intervention_rate", ""),
            "latency_p50_ms": res.get("latency_p50_ms", ""),
            "latency_p99_ms": res.get("latency_p99_ms", ""),
        })
    return rows


def _evaluate_val_ood(week_label: str, start_idx: int) -> List[Dict[str, Any]]:
    """Run evaluate_ood_stress_suite on representative Val week for PSM', MPCPolicy(causal_noisy), and clamped_ppo_diff_shield."""
    rows: List[Dict[str, Any]] = []
    suite = f"ood_val_{week_label}"
    contingencies = list(OOD_CONTINGENCIES_ORDER)

    env_psm = SmartGridEnv({"split": "val", "charge_clamp": True, "seed": 42})
    psm_ood = evaluate_ood_stress_suite(
        env_psm,
        policy=PeakShaveMinPolicy(),
        start_idx=start_idx,
        contingencies=contingencies,
    )
    rows.extend(_ood_suite_to_rows(suite, "PSM'", "N/A", "N/A", psm_ood))

    env_mpc = SmartGridEnv({"split": "val", "charge_clamp": True, "seed": 42})
    mpc_pol = MPCPolicy(env=env_mpc, horizon=24, forecast_mode="causal_noisy", seed=42)
    mpc_ood = evaluate_ood_stress_suite(
        env_mpc,
        policy=mpc_pol,
        start_idx=start_idx,
        contingencies=contingencies,
    )
    rows.extend(_ood_suite_to_rows(suite, "MPCPolicy(causal_noisy, H=24)", "42", "N/A", mpc_ood))

    cfg_rl = load_config_with_base("clamped_ppo_diff_shield")
    cfg_rl["seed"] = 101
    env_rl = CapShield(build_env(cfg_rl, split="val", seed=201))
    agent_rl = load_agent_for_method(cfg_rl["method"], cfg_rl)
    ckpt_rl = os.path.join(
        REPO_ROOT, "checkpoints", "ppo_diff_shield", "clamped_ppo_diff_shield_seed101_final.pt"
    )
    agent_rl.load_models(ckpt_rl)
    rl_ood = evaluate_ood_stress_suite(
        env_rl,
        policy=None,
        agent_wrapper=agent_rl,
        start_idx=start_idx,
        contingencies=contingencies,
    )
    rows.extend(_ood_suite_to_rows(suite, "clamped_ppo_diff_shield", 101, "final", rl_ood))

    return rows


def run_all_val_benchmarks(out_csv: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run Regime 1, Regime 2, and Val OOD benchmarks and save results/tables/regime_benchmarks_val.csv."""
    t0 = time.time()
    all_rows: List[Dict[str, Any]] = []

    print("[1/3] Running Regime 1 Val (26 weeks)...")
    all_rows.extend(
        _evaluate_regime_baselines(
            suite="regime1_val",
            regime="regime1",
            env_cfg={},
            mpc_modes=("perfect", "causal_noisy", "persistence"),
            milp_time_limit=2.0,
        )
    )
    all_rows.extend(_evaluate_rl_seeds("regime1_val", "regime1", "clamped_ppo_diff_shield", "final"))
    all_rows.extend(_evaluate_rl_seeds("regime1_val", "regime1", "clamped_ppo_diff_shield", "best"))
    all_rows.extend(_evaluate_rl_seeds("regime1_val", "regime1", "causal23_ppo_diff_shield", "final"))
    all_rows.extend(_evaluate_rl_seeds("regime1_val", "regime1", "causal23_ppo_diff_shield", "best"))

    print("[2/3] Running Regime 2 Val (26 weeks, ci_* tariffs)...")
    ci_cfg = load_config_with_base(REGIME2_REF_CONFIG)
    regime2_env_cfg = {k: ci_cfg[k] for k in REGIME2_ENV_KEYS}
    all_rows.extend(
        _evaluate_regime_baselines(
            suite="regime2_val",
            regime="regime2",
            env_cfg=regime2_env_cfg,
            mpc_modes=("perfect", "causal_noisy"),
            milp_time_limit=2.0,
        )
    )
    for exp_id in ("ci_ppo_shielded", "ci_ppo_diff_shield", "ci_ppo_diff_6d"):
        all_rows.extend(_evaluate_rl_seeds("regime2_val", "regime2", exp_id, "final"))

    print("[3/3] Running OOD stress suite on Val representative week...")
    week_label, start_idx = _resolve_val_stress_week_idx()
    all_rows.extend(_evaluate_val_ood(week_label, start_idx))

    dest = out_csv or os.path.join(REPO_ROOT, "results", "tables", "regime_benchmarks_val.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {dest} ({len(all_rows)} rows) in {time.time() - t0:.1f}s")
    return all_rows


def _print_summary(rows: List[Dict[str, Any]]) -> None:
    """Print compact summary tables for Regime 1, Regime 2, and OOD Val."""
    for suite_prefix in ("regime1_val", "regime2_val"):
        print(f"\n=== {suite_prefix.upper()} ===")
        for r in rows:
            if r["suite"] == suite_prefix and r["row_type"] in ("single", "mean"):
                print(
                    f"  {r['method']:30s} [{r['checkpoint_type']:4s}|{r['row_type']:6s}] "
                    f"cost={r['cost_eur']:9.2f} EUR | exc={r['mean_exceedance_kw']:6.4f} kW | "
                    f"max_exc={r['max_hourly_exceedance_kw']} | cvar95={r['cvar95_cost_eur']} | "
                    f"dc={r['demand_charge_eur']} | trip={r['industrial_trip_eur']} | "
                    f"tp={r['battery_throughput_mwh']} MWh | soc={r['active_soc_pct']}% | "
                    f"p50={r['latency_p50_ms']}ms p99={r['latency_p99_ms']}ms"
                )
    print("\n=== VAL OOD STRESS SUITE ===")
    for r in rows:
        if r["suite"].startswith("ood_val_"):
            print(
                f"  {r['method']:28s} | {r['contingency']:18s} | "
                f"cost={r['cost_eur']:8.2f} EUR | exc={r['mean_exceedance_kw']:6.4f} kW | "
                f"max_exc={r['max_hourly_exceedance_kw']} | viol_h={r['violation_hours']}"
            )


def main() -> int:
    rows = run_all_val_benchmarks()
    _print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
