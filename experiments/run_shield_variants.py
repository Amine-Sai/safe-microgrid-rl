import os
import sys
import json
import time
from multiprocessing import Pool

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

def run_job(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    import torch as th
    th.set_num_threads(1)
    
    from experiments.train_core import load_config_with_base, run_paths, train_single_method
    
    cfg_name, method, seed, run_id = job
    print(f"Starting {run_id} seed {seed}...", flush=True)
    t0 = time.time()
    
    cfg = load_config_with_base(cfg_name)
    cfg["method"] = method
    cfg["seed"] = seed
    
    result_path, ckpt_dir = run_paths(method, cfg, run_id)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    res = train_single_method(
        method,
        cfg,
        ckpt_dir=ckpt_dir,
        run_name=run_id,
        allow_overwrite=True,
    )
    
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)
        
    elapsed = time.time() - t0
    final_cost = res["final_checkpoint"]["metrics"]["total_cost_eur"]
    final_exc = res["final_checkpoint"]["metrics"]["mean_exceedance_kw"]
    print(f"FINISHED {run_id} seed {seed} in {elapsed:.1f}s | Val Cost: {final_cost:.2f} EUR | Exc: {final_exc:.4f} kW", flush=True)
    return {
        "job": job,
        "elapsed": elapsed,
        "final_cost": final_cost,
        "final_exc": final_exc,
        "result_path": result_path,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO-Shield-Aux and PPO-Diff-Shield variants.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ppo_shield_aux", "ppo_diff_shield"],
        help="Methods to run: diff_shield, ppo_diff_shield, shield_aux, ppo_shield_aux, or all.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[101, 102, 103, 104, 105],
        help="Random seeds to train (default: 101 102 103 104 105).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max parallel worker processes.",
    )
    args = parser.parse_args()

    # Normalise methods
    selected_methods = set()
    for m in args.methods:
        m_lower = m.lower().replace("-", "_")
        if m_lower in ("all",):
            selected_methods.add("ppo_shield_aux")
            selected_methods.add("ppo_diff_shield")
        elif m_lower in ("diff_shield", "ppo_diff_shield", "diff"):
            selected_methods.add("ppo_diff_shield")
        elif m_lower in ("shield_aux", "ppo_shield_aux", "aux"):
            selected_methods.add("ppo_shield_aux")
        else:
            selected_methods.add(m_lower)

    jobs = []
    if "ppo_shield_aux" in selected_methods:
        for s in args.seeds:
            jobs.append(("clamped_ppo_shield_aux", "ppo_shield_aux", s, "clamped_ppo_shield_aux"))
    if "ppo_diff_shield" in selected_methods:
        for s in args.seeds:
            jobs.append(("clamped_ppo_diff_shield", "ppo_diff_shield", s, "clamped_ppo_diff_shield"))

    if not jobs:
        print(f"No matching jobs for methods={args.methods} and seeds={args.seeds}.")
        return

    n_workers = min(len(jobs), max(1, args.workers))
    print(f"Submitting {len(jobs)} jobs to Pool({n_workers})...", flush=True)
    t_start = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_job, jobs)
        
    total_time = time.time() - t_start
    print(f"All {len(results)} jobs completed in {total_time:.1f}s!", flush=True)
    for r in results:
        print(f"  {r['job'][3]} seed {r['job'][2]}: {r['final_cost']:.2f} EUR, exc={r['final_exc']:.4f} kW ({r['elapsed']:.1f}s)")

if __name__ == "__main__":
    main()
