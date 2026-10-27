#!/usr/bin/env python3
"""Run the clamped experiment family (300k steps, charge_clamp: true, seeds 101-105).

Methods:
- clamped_ppo
- clamped_ppo_lagrangian
- clamped_ppo_residual
- clamped_ppo_shielded
"""

import concurrent.futures
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


def _run_single_job(args):
    config_id, seed = args
    # Pin threads to 1 per process for clean CPU scaling
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch as th
    th.set_num_threads(1)

    from experiments.train_core import load_config_with_base, run_paths, train_single_method

    cfg = load_config_with_base(config_id)
    cfg["seed"] = seed
    method = cfg["method"]
    result_path, ckpt_dir = run_paths(method, cfg, config_id)

    print(f"STARTING: {config_id} seed={seed} -> {result_path}", flush=True)
    t0 = time.time()
    res = train_single_method(
        method, cfg, ckpt_dir=ckpt_dir, run_name=config_id, allow_overwrite=True
    )
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)

    val_cost = res["final_checkpoint"]["metrics"]["total_cost_eur"]
    val_exc = res["final_checkpoint"]["metrics"]["mean_exceedance_kw"]
    print(
        f"FINISHED: {config_id} seed={seed} in {elapsed:.1f}s | "
        f"cost={val_cost:.2f} | exc={val_exc:.4f} -> {result_path}",
        flush=True,
    )
    return config_id, seed, val_cost, val_exc, elapsed


def main():
    configs = ["clamped_ppo", "clamped_ppo_lagrangian", "clamped_ppo_residual", "clamped_ppo_shielded"]
    seeds = [101, 102, 103, 104, 105]

    jobs = [(c, s) for c in configs for s in seeds]
    max_workers = min(8, os.cpu_count() or 4)

    print("=" * 80)
    print(f"RUNNING CLAMPED EXPERIMENT GRID ({len(jobs)} jobs, {max_workers} parallel workers)")
    print("=" * 80)
    t_start = time.time()

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"FAILED: {job}: {e}", file=sys.stderr)
                raise e

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"ALL {len(results)} RUNS FINISHED in {total_time/60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
