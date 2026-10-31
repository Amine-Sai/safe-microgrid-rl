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
    jobs = []
    seeds = [101, 102, 103, 104, 105]
    for s in seeds:
        jobs.append(("clamped_ppo_shield_aux", "ppo_shield_aux", s, "clamped_ppo_shield_aux"))
    for s in seeds:
        jobs.append(("clamped_ppo_diff_shield", "ppo_diff_shield", s, "clamped_ppo_diff_shield"))
        
    print(f"Submitting {len(jobs)} jobs to Pool(10)...", flush=True)
    t_start = time.time()
    with Pool(processes=10) as pool:
        results = pool.map(run_job, jobs)
        
    total_time = time.time() - t_start
    print(f"All {len(results)} jobs completed in {total_time:.1f}s!", flush=True)
    for r in results:
        print(f"  {r['job'][3]} seed {r['job'][2]}: {r['final_cost']:.2f} EUR, exc={r['final_exc']:.4f} kW ({r['elapsed']:.1f}s)")

if __name__ == "__main__":
    main()
