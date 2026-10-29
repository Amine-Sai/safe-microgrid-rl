import sys
import os
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th

from experiments.train_core import PPO_METHODS, train_single_method


def run_worker(job):
    """Run one (method, seed) job in a fresh process with a single torch thread."""
    # Pin thread pools to 1
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    method, seed, steps, run_id, eval_interval, config_path, overwrite = job
    th.set_num_threads(1)
    from experiments.train_core import load_config_with_base, run_paths
    cfg = load_config_with_base(config_path)
    cfg["method"] = method
    cfg["seed"] = seed
    if steps:
        cfg["total_steps"] = steps
    if eval_interval:
        cfg["eval_interval_steps"] = eval_interval

    result_path, ckpt_dir = run_paths(method, cfg, run_id)
    if os.path.exists(result_path) and not overwrite:
        raise FileExistsError(
            f"{result_path} already exists - refusing to overwrite it "
            f"(pass --overwrite if the old run is really superseded)")

    t0 = time.time()
    res = train_single_method(method, cfg, ckpt_dir=ckpt_dir, run_name=run_id,
                              allow_overwrite=overwrite)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)
    return {
        "method": method,
        "agent": res.get("agent"),
        "run_id": run_id,
        "seed": seed,
        "steps": cfg["total_steps"],
        "wall_clock_sec": time.time() - t0,
        "final_val_cost": res["final_checkpoint"]["metrics"]["total_cost_eur"],
        "final_val_exc": res["final_checkpoint"]["metrics"]["mean_exceedance_kw"],
        "final_lambda": res["lambda_summary"]["lambda_end"],
        "result_path": result_path,
        "ckpt_path": os.path.join(ckpt_dir, f"{run_id}_seed{seed}_final.pt"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="all", help="comma list or 'all'")
    ap.add_argument("--seeds", default="42", help="comma list")
    ap.add_argument("--steps", type=int, default=0, help="override total_steps (0 = config)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run-id", default=None,
                    help="run id for the whole grid (default: a timestamp, so two "
                         "launches never write to the same file)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing result / checkpoint")
    ap.add_argument("--eval-interval", type=int, default=0, help="override eval_interval_steps")
    ap.add_argument("--config", default="configs/config.yaml",
                    help="config YAML (or experiment id) to load per job")
    ap.add_argument("--verify", action="store_true", help="run one job twice, compare")
    ap.add_argument("--concurrent-verify", action="store_true",
                    help="run two IDENTICAL jobs concurrently, compare to 1e-6")
    args = ap.parse_args()

    methods = PPO_METHODS if args.methods == "all" else args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    if args.concurrent_verify:
        from concurrent.futures import ProcessPoolExecutor
        method, seed = methods[0], seeds[0]
        print(f"CONCURRENT VERIFY: two identical {method} seed={seed} jobs launched at once...")
        with ProcessPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(run_worker, (method, seed, args.steps, run_id, args.eval_interval, args.config, True))
            f2 = ex.submit(run_worker, (method, seed, args.steps, run_id, args.eval_interval, args.config, True))
            r1, r2 = f1.result(), f2.result()
        keys = ["final_val_cost", "final_val_exc", "final_lambda"]
        match = all(abs(r1[k] - r2[k]) < 1e-6 for k in keys if r1[k] is not None and r2[k] is not None)
        print(f"  run1 cost={r1['final_val_cost']:.6f}  run2 cost={r2['final_val_cost']:.6f}")
        print(f"  concurrent determinism (match to 1e-6): {match}")
        if not match:
            print("  NON-DETERMINISM UNDER CONCURRENCY — investigate before any parallel run.")
        return

    if args.verify:
        # Run the same (method, seed) in two fresh processes; metrics must match.
        method, seed = methods[0], seeds[0]
        print(f"VERIFY: running {method} seed={seed} twice in separate processes...")
        r1 = run_worker((method, seed, args.steps, run_id, args.eval_interval, args.config, True))
        r2 = run_worker((method, seed, args.steps, run_id, args.eval_interval, args.config, True))
        keys = ["final_val_cost", "final_val_exc", "final_lambda"]
        match = all(abs(r1[k] - r2[k]) < 1e-6 for k in keys if r1[k] is not None and r2[k] is not None)
        print(f"  run1 cost={r1['final_val_cost']:.6f}  run2 cost={r2['final_val_cost']:.6f}")
        print(f"  deterministic (match to 1e-6): {match}")
        if not match:
            print("  NON-DETERMINISM DETECTED — investigate before any parallel run.")
        return

    jobs = [(m, s, args.steps, run_id, args.eval_interval, args.config, args.overwrite)
            for m in methods for s in seeds]
    print(f"PARALLEL GRID: {len(methods)} methods x {len(seeds)} seeds = {len(jobs)} runs")
    print(f"  run id={run_id}  workers={args.workers}  torch.set_num_threads(1) per process")

    t0 = time.time()
    results = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_worker, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{i}/{len(jobs)}] {r['method']} seed={r['seed']} done in "
                      f"{r['wall_clock_sec']/60:.1f} min | Val cost={r['final_val_cost']:.2f} "
                      f"exc={r['final_val_exc']:.4f}", flush=True)
            except Exception as e:
                print(f"  [{i}/{len(jobs)}] {j[0]} seed={j[1]} FAILED: {e}", flush=True)
                results.append({"method": j[0], "seed": j[1], "error": str(e)})

    wall = time.time() - t0
    n_ok = sum(1 for r in results if "error" not in r)
    serial_est = sum(r.get("wall_clock_sec", 0) for r in results)
    print(f"\nCOMPLETE: {n_ok}/{len(jobs)} succeeded | wall={wall/60:.1f} min | "
          f"sum-of-runs={serial_est/60:.1f} min | speedup={serial_est/wall:.2f}x")

    agents = sorted({r.get("agent") for r in results if "error" not in r and r.get("agent")})
    grid_dir = os.path.join("results", "runs", agents[0]) if len(agents) == 1 \
        else os.path.join("results", "runs")
    out = os.path.join(grid_dir, run_id + "_grid.json")
    os.makedirs(grid_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"wall_clock_sec": wall, "speedup": serial_est / wall if wall else 0,
                   "n_ok": n_ok, "n_total": len(jobs), "run_id": run_id,
                   "results": results}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
