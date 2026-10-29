#!/usr/bin/env python3
"""Audit: measure the effect of the charge clamp before and after training.

Rolls out Grid-Only, PSM', uniform-random, and checkpoints found
under checkpoints/ on the Val split, reporting:
- Cost (EUR)
- Mean exceedance (kW)
- Violation rate (%)
- charge_clipped_by_cap_kw mean, max, % of steps, total kWh
- charge_clipped_by_soc_kw mean (SoC ceiling saturation)

Also provides --backfill-runs to deterministically re-evaluate and backfill
saved clamped checkpoints on the Val split.
"""

import glob
import json
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from env.smartgrid_env import SmartGridEnv
from env.cap_shield import CapShield
from agents.heuristics import GridOnlyPolicy, psm_prime_policy


def evaluate_policy_clipping(policy_fn, env_cfg=None):
    cfg = {"split": "val", "seed": 42, "charge_clamp": True}
    if env_cfg:
        cfg.update(env_cfg)
    env = SmartGridEnv(cfg)
    ep_len = env.episode_length
    split_weeks = env._n_timesteps // ep_len
    total_hours = split_weeks * ep_len

    cap_clipped_kw = []
    soc_clipped_kw = []
    clipped_by_cap = []
    requested_kw = []
    applied_kw = []
    exceedances = []
    violation_hours = 0
    total_cost = 0.0

    for week in range(split_weeks):
        obs, _ = env.reset(options={"start_idx": week * ep_len})
        for _ in range(ep_len):
            action = policy_fn(obs)
            obs, r, _, _, info = env.step(action)
            clip_cap = info.get("charge_clipped_by_cap_kw", 0.0)
            clip_soc = info.get("charge_clipped_by_soc_kw", 0.0)
            exc = info["grid_exceedance_kw"]

            cap_clipped_kw.append(clip_cap)
            soc_clipped_kw.append(clip_soc)
            clipped_by_cap.append(bool(info.get("is_charge_clipped_by_cap", False)))
            requested_kw.append(info.get("charge_requested_kw", 0.0))
            applied_kw.append(info.get("charge_applied_kw", 0.0))
            exceedances.append(exc)
            if info.get("is_violation", exc > env.violation_tolerance_kw):
                violation_hours += 1
            total_cost += (info["energy_cost"] + info["unserved_cost"] + info["battery_wear_cost"])

    cap_clip_arr = np.array(cap_clipped_kw)
    soc_clip_arr = np.array(soc_clipped_kw)
    cap_pct_arr = np.array(clipped_by_cap)
    exc_arr = np.array(exceedances)

    return {
        "cost_eur": float(total_cost),
        "mean_exceedance_kw": float(np.mean(exc_arr)),
        "violation_rate_pct": float(violation_hours / total_hours * 100.0),
        "cap_clip_mean_kw": float(np.mean(cap_clip_arr)),
        "cap_clip_max_kw": float(np.max(cap_clip_arr)),
        "cap_clip_pct": float(np.mean(cap_pct_arr) * 100.0),
        "cap_clip_total_kwh": float(np.sum(cap_clip_arr) * env.dt_hours),
        "soc_clip_mean_kw": float(np.mean(soc_clip_arr)),
        "requested_mean_kw": float(np.mean(requested_kw)),
        "applied_mean_kw": float(np.mean(applied_kw)),
    }


def backfill_runs():
    """Deterministically re-evaluate all saved clamped checkpoints on Val and update run JSONs."""
    from experiments.train_core import (
        load_config_with_base,
        load_agent_for_method,
        build_env,
        evaluate_deterministic_val,
    )

    run_files = sorted(glob.glob(os.path.join(REPO_ROOT, "results", "runs", "*", "*.json")))
    target_runs = [
        f for f in run_files
        if not f.endswith("_grid.json") and any(os.path.basename(f).startswith(p) for p in ("s3v2_", "clamped_"))
    ]
    print(f"Backfilling {len(target_runs)} clamped run JSONs from checkpoints...")

    updated_count = 0
    for rpath in target_runs:
        with open(rpath, "r", encoding="utf-8") as f:
            d = json.load(f)

        fc = d.get("final_checkpoint") or {}
        ckpt_path = fc.get("weights")
        if not ckpt_path or not os.path.exists(ckpt_path):
            # Try finding final.pt in checkpoints
            fname = os.path.basename(rpath).replace(".json", "_final.pt")
            agent_dir = os.path.basename(os.path.dirname(rpath))
            candidate = os.path.join(REPO_ROOT, "checkpoints", agent_dir, fname)
            if os.path.exists(candidate):
                ckpt_path = candidate
            else:
                print(f"Skipping {rpath}: checkpoint {ckpt_path} not found")
                continue

        # Load config
        exp_id = os.path.basename(rpath).split("_seed")[0]
        seed = d.get("seed", 101)
        cfg = load_config_with_base(exp_id)
        cfg["seed"] = seed
        agent = load_agent_for_method(cfg["method"], cfg)
        agent.load_models(ckpt_path)

        # Build validation env
        env_val = build_env(dict(cfg, shed_mapping=cfg.get("shed_mapping", "linear")), split="val", seed=seed + 100)
        if cfg.get("shielded"):
            env_val = CapShield(env_val)

        new_metrics = evaluate_deterministic_val(env_val, agent)
        d["final_checkpoint"]["metrics"] = new_metrics
        d["final_metrics"] = new_metrics

        # Also re-evaluate eval_checkpoints if intermediate weights exist
        ckpt_dir = os.path.dirname(ckpt_path)
        stem = os.path.basename(ckpt_path).replace("_final.pt", "")
        for tag in list(d.get("eval_checkpoints", {}).keys()):
            step_num = tag.split("_")[0]
            step_ckpt = os.path.join(ckpt_dir, f"{stem}_step{step_num}.pt")
            if os.path.exists(step_ckpt):
                agent.load_models(step_ckpt)
                d["eval_checkpoints"][tag] = evaluate_deterministic_val(env_val, agent)

        # Update best checkpoint
        cost_limit = cfg.get("cost_limit", 0.05)
        tol = getattr(env_val, "violation_tolerance_kw", 1e-4)
        feasible = [(k, v) for k, v in d["eval_checkpoints"].items() if v["mean_exceedance_kw"] <= cost_limit + tol]
        if feasible:
            best_k, best_v = min(feasible, key=lambda x: x[1]["total_cost_eur"])
        else:
            best_k, best_v = min(d["eval_checkpoints"].items(), key=lambda x: (x[1]["mean_exceedance_kw"], x[1]["total_cost_eur"]))
        d["best_checkpoint"]["checkpoint"] = best_k
        d["best_checkpoint"]["metrics"] = best_v

        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        print(f"  [OK] {os.path.basename(rpath)}: cost={new_metrics['total_cost_eur']:.2f}, "
              f"exc={new_metrics['mean_exceedance_kw']:.4f}, viol={new_metrics['violation_rate_pct']:.2f}%, "
              f"cap_clip={new_metrics['mean_charge_clipped_by_cap_kw']:.4f} kW ({new_metrics['cap_clipped_steps_pct']:.2f}%)")
        updated_count += 1

    print(f"Successfully backfilled {updated_count} run files.")


def main():
    if "--backfill-runs" in sys.argv:
        backfill_runs()
        return

    print("=" * 115)
    print("CHARGE CLAMP AUDIT (Val Split, 26 Weeks, 4,368 Hours)")
    print("=" * 115)
    print(f"{'Policy / Method':<26} {'Cost (EUR)':>10} {'Exceed (kW)':>12} {'Viol %':>8} "
          f"{'Cap Clip kW':>12} {'Cap Clip %':>11} {'Cap kWh':>9} {'SoC Clip kW':>12}")
    print("-" * 115)

    policies = {}
    policies["Grid-Only"] = lambda o: GridOnlyPolicy().get_action(o)
    policies["PSM'"] = lambda o: psm_prime_policy().get_action(o)

    rng = np.random.RandomState(42)
    policies["uniform-random"] = lambda o: rng.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)

    # Check for final checkpoints in checkpoints/
    ckpt_files = sorted(glob.glob(os.path.join(REPO_ROOT, "checkpoints", "**", "*_final.pt"), recursive=True))
    for ckpt_path in ckpt_files:
        fname = os.path.basename(ckpt_path)
        parts = fname.replace("_final.pt", "").split("_seed")
        cfg_id = parts[0]
        try:
            from experiments.train_core import load_config_with_base, load_agent_for_method
            cfg = load_config_with_base(cfg_id)
            agent = load_agent_for_method(cfg["method"], cfg)
            agent.load_models(ckpt_path)
            if cfg.get("shielded"):
                shield = CapShield(SmartGridEnv({"split": "val", "seed": 42, "charge_clamp": True}))
                policies[f"ckpt:{fname}"] = lambda o, ag=agent, sh=shield: sh.compose_action(ag.select_action(o, evaluate=True)[0], o)
            else:
                policies[f"ckpt:{fname}"] = lambda o, ag=agent: ag.select_action(o, evaluate=True)[0]
        except Exception:
            pass

    for name, pol in policies.items():
        res = evaluate_policy_clipping(pol)
        print(f"{name:<26} {res['cost_eur']:10.2f} {res['mean_exceedance_kw']:12.4f} "
              f"{res['violation_rate_pct']:7.2f}% {res['cap_clip_mean_kw']:12.4f} "
              f"{res['cap_clip_pct']:10.2f}% {res['cap_clip_total_kwh']:9.1f} {res['soc_clip_mean_kw']:12.2f}")

    print("=" * 115)


if __name__ == "__main__":
    main()
