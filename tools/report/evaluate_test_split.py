#!/usr/bin/env python3
"""Locked one-shot evaluation on the closed Test split (2019 H2).

Requires `--unlock-test-split` to prevent premature execution.
Evaluates baselines (Grid-Only, PSM') and trained checkpoints on the full
24 Test weeks (4,032 hours) and on the 4 pre-identified stress weeks.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from agents.heuristics import GridOnlyPolicy, PeakShaveMinPolicy
from env.smartgrid_env import SmartGridEnv
from experiments.train_core import load_agent_for_method, load_config_with_base


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def evaluate_policy_on_split(env: SmartGridEnv, policy, agent_wrapper=None):
    """Deterministic rollout over all weeks of an environment split."""
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len
    total_hours = n_weeks * ep_len

    total_cost = 0.0
    unserved_kwh = 0.0
    demand_kwh = 0.0
    violation_hours = 0
    exceedances = []
    cap_clipped_kw = []

    for week in range(n_weeks):
        obs, _ = env.reset(options={"start_idx": week * ep_len})
        for _ in range(ep_len):
            if agent_wrapper is not None:
                action = agent_wrapper.select_action(obs, evaluate=True)[0]
            else:
                action = policy.get_action(obs)

            obs, _, _, _, info = env.step(action)
            total_cost += info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
            unserved_kwh += info["total_unserved_kw"] * env.dt_hours
            demand_kwh += (info["total_served_kw"] + info["total_unserved_kw"]) * env.dt_hours

            exc = info["grid_exceedance_kw"]
            exceedances.append(exc)
            if info.get("is_violation", exc > env.violation_tolerance_kw):
                violation_hours += 1
            cap_clipped_kw.append(info.get("charge_clipped_by_cap_kw", 0.0))

    return {
        "cost_eur": round(float(total_cost), 2),
        "mean_exceedance_kw": round(float(np.mean(exceedances)), 4),
        "violation_rate_pct": round(float(violation_hours / total_hours * 100.0), 2),
        "unserved_pct": round(float(unserved_kwh / max(1e-6, demand_kwh) * 100.0), 4),
        "mean_cap_clipped_kw": round(float(np.mean(cap_clipped_kw)), 4),
    }


def evaluate_single_week(env: SmartGridEnv, start_idx: int, policy, agent_wrapper=None):
    """Evaluate one 168-hour week starting at start_idx."""
    ep_len = env.episode_length
    obs, _ = env.reset(options={"start_idx": start_idx})
    total_cost = 0.0
    exceedances = []
    violation_hours = 0

    for _ in range(ep_len):
        if agent_wrapper is not None:
            action = agent_wrapper.select_action(obs, evaluate=True)[0]
        else:
            action = policy.get_action(obs)
        obs, _, _, _, info = env.step(action)
        total_cost += info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"]
        exc = info["grid_exceedance_kw"]
        exceedances.append(exc)
        if info.get("is_violation", exc > env.violation_tolerance_kw):
            violation_hours += 1

    return {
        "cost_eur": round(float(total_cost), 2),
        "mean_exceedance_kw": round(float(np.mean(exceedances)), 4),
        "violation_hours": int(violation_hours),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on closed Test split.")
    parser.add_argument("--unlock-test-split", action="store_true",
                        help="Safety gate: explicitly unlock evaluation on the Test split.")
    parser.add_argument("--family", default="clamped",
                        choices=["clamped", "clamped_forecast", "unclamped", "all"],
                        help="Which checkpoint family to evaluate.")
    args = parser.parse_args()

    if not args.unlock_test_split:
        print("[LOCKED] The Test split (2019 H2) is closed.")
        print("Pass `--unlock-test-split` to confirm this is the final one-shot evaluation.")
        return 1

    split_name = "test"
    report_path = os.path.join(REPO_ROOT, "data", "processed", "split_report.json")
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found.")
        return 1

    with open(report_path, encoding="utf-8") as f:
        split_report = json.load(f)

    test_stress = split_report.get("stress_weeks", {}).get("test", {})

    # Build reference environment
    env = SmartGridEnv({"split": split_name, "charge_clamp": True})
    ep_len = env.episode_length
    n_weeks = env._n_timesteps // ep_len

    # Map stress week labels to start_idx in the test environment
    import pandas as pd
    test_parquet = os.path.join(REPO_ROOT, "data", "processed", f"{split_name}.parquet")
    df_test = pd.read_parquet(test_parquet)
    iso_labels = (
        df_test.index.isocalendar().year.astype(str)
        + "-W"
        + df_test.index.isocalendar().week.astype(str).str.zfill(2)
    ).values

    stress_indices = {}
    for crit, info in test_stress.items():
        week_target = info["week"]
        matches = np.where(iso_labels == week_target)[0]
        if len(matches) >= ep_len:
            stress_indices[crit] = int(matches[0])

    print("=" * 80)
    print(f"LOCKED TEST SPLIT EVALUATION (Split: {split_name}, {n_weeks} weeks, {env._n_timesteps} hours)")
    print("=" * 80)

    # 1. Evaluate reference heuristics
    eval_records = []
    stress_records = []

    heuristics = [("Grid-Only", GridOnlyPolicy()), ("PSM'", PeakShaveMinPolicy())]
    for name, pol in heuristics:
        print(f"Evaluating {name}...", end=" ", flush=True)
        res = evaluate_policy_on_split(env, pol)
        print(f"Cost={res['cost_eur']:,.2f} EUR | Exc={res['mean_exceedance_kw']:.4f} kW | Viol={res['violation_rate_pct']:.2f}%")
        eval_records.append({
            "model": name, "family": "baseline", "seed": "N/A", **res
        })
        for crit, s_idx in stress_indices.items():
            s_res = evaluate_single_week(env, s_idx, pol)
            stress_records.append({
                "model": name, "family": "baseline", "seed": "N/A",
                "stress_criterion": crit, "week": test_stress[crit]["week"], **s_res
            })

    # 2. Find and evaluate matching checkpoints
    ckpt_pattern = os.path.join(REPO_ROOT, "checkpoints", "*", "*_final.pt")
    all_ckpts = sorted(glob.glob(ckpt_pattern))

    for ckpt_path in all_ckpts:
        fname = os.path.basename(ckpt_path)
        # Determine family
        if fname.startswith("clamped_forecast_"):
            ckpt_fam = "clamped_forecast"
        elif fname.startswith("clamped_") or fname.startswith("s3v2_"):
            ckpt_fam = "clamped"
        elif fname.startswith("unclamped_") or fname.startswith("s3_"):
            ckpt_fam = "unclamped"
        else:
            continue

        if args.family != "all" and ckpt_fam != args.family:
            continue

        # Extract config stem and seed
        stem = fname.replace("_final.pt", "")
        parts = stem.split("_seed")
        config_name = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 42

        try:
            cfg = load_config_with_base(config_name)
            cfg["seed"] = seed
            method = cfg["method"]
            agent = load_agent_for_method(method, cfg)
            agent.load_models(ckpt_path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        env_cfg = {"split": split_name, "charge_clamp": cfg.get("charge_clamp", True)}
        if cfg.get("forecast_horizon_hours", 0) > 0:
            env_cfg["forecast_horizon_hours"] = cfg["forecast_horizon_hours"]
        env_eval = SmartGridEnv(env_cfg)

        if cfg.get("shielded"):
            from env.cap_shield import CapShield
            env_eval = CapShield(env_eval)

        print(f"Evaluating {stem}...", end=" ", flush=True)
        res = evaluate_policy_on_split(env_eval, None, agent_wrapper=agent)
        print(f"Cost={res['cost_eur']:,.2f} EUR | Exc={res['mean_exceedance_kw']:.4f} kW | Viol={res['violation_rate_pct']:.2f}%")
        eval_records.append({
            "model": stem, "family": ckpt_fam, "seed": seed, **res
        })
        for crit, s_idx in stress_indices.items():
            s_res = evaluate_single_week(env_eval, s_idx, None, agent_wrapper=agent)
            stress_records.append({
                "model": stem, "family": ckpt_fam, "seed": seed,
                "stress_criterion": crit, "week": test_stress[crit]["week"], **s_res
            })

    # Save outputs
    tables_dir = os.path.join(REPO_ROOT, "results", "tables")
    runs_dir = os.path.join(REPO_ROOT, "results", "runs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    out_csv = os.path.join(tables_dir, "test_evaluation.csv")
    if eval_records:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_records[0]))
            w.writeheader()
            w.writerows(eval_records)
        print(f"Wrote {out_csv} ({len(eval_records)} rows)")

    stress_csv = os.path.join(tables_dir, "test_stress_weeks.csv")
    if stress_records:
        with open(stress_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stress_records[0]))
            w.writeheader()
            w.writerows(stress_records)
        print(f"Wrote {stress_csv} ({len(stress_records)} rows)")

    lock_record = {
        "timestamp_utc": str(pd.Timestamp.utcnow()),
        "git_commit": _get_git_commit(),
        "split": split_name,
        "n_hours": env._n_timesteps,
        "n_weeks": n_weeks,
        "records_count": len(eval_records),
    }
    lock_file = os.path.join(runs_dir, "test_lock.json")
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(lock_record, f, indent=2)
    print(f"Wrote lock provenance to {lock_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
