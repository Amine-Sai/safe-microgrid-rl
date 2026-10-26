import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

GRID_ONLY_VAL = 45337.92170812376
D_MAIN = 0.05


def _job(job):
    import torch as th
    th.set_num_threads(1)
    from experiments.train_core import load_config_with_base, run_paths, train_single_method

    cfg_name, method, seed, steps, eval_interval, run_name = job
    cfg = load_config_with_base(cfg_name)
    cfg["method"] = method
    cfg["seed"] = seed
    if steps:
        cfg["total_steps"] = steps
    if eval_interval:
        cfg["eval_interval_steps"] = eval_interval

    result_path, ckpt_dir = run_paths(method, cfg, run_name)
    res = train_single_method(method, cfg, ckpt_dir=ckpt_dir, run_name=run_name,
                              allow_overwrite=True)

    # per-eval curve (deterministic) + the extra diagnostics the guide asks for
    ec = res.get("eval_checkpoints", {})
    curve = {int(k.split("_")[0]): v["total_cost_eur"] for k, v in ec.items()}
    curve_exc = {int(k.split("_")[0]): v["mean_exceedance_kw"] for k, v in ec.items()}
    lam = res["lambda_summary"]

    return {
        "config": cfg_name, "method": method, "seed": seed,
        "final_val_cost": res["final_checkpoint"]["metrics"]["total_cost_eur"],
        "final_val_exc": res["final_checkpoint"]["metrics"]["mean_exceedance_kw"],
        "final_unserved_pct": res["final_checkpoint"]["metrics"]["unserved_energy_pct"],
        "final_violation_rate_pct": res["final_checkpoint"]["metrics"]["violation_rate_pct"],
        "curve_val_cost": curve, "curve_val_exc": curve_exc,
        "lambda_end": lam["lambda_end"],
        "alpha_final": res["degeneracy_check"].get("mean_shed_cmd"),  # placeholder, replaced below
        "result_path": result_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma list of config stems")
    ap.add_argument("--method", default="ppo",
                    help="agent method (PPO family; SAC agents are not shipped)")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-interval", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-id", default="s1b", help="run id prefix for the batch")
    args = ap.parse_args()

    cfgs = args.configs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(c, args.method, s, args.steps, args.eval_interval,
             f"{args.run_id}_{c}") for c in cfgs for s in seeds]

    print(f"BATCH: {len(cfgs)} configs x {len(seeds)} seeds = {len(jobs)} runs, "
          f"workers={args.workers}")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = __import__("time").time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                r = f.result()
                results.append(r)
                print(f"  [{i}/{len(jobs)}] {r['config']:18s} seed={r['seed']} "
                      f"J(20k)={r['final_val_cost']:,.2f} exc={r['final_val_exc']:.4f} "
                      f"lam={r['lambda_end']}", flush=True)
            except Exception as e:
                print(f"  [{i}/{len(jobs)}] {j[0]} seed={j[1]} FAILED: {e}", flush=True)
    wall = __import__("time").time() - t0

    # pass proxy
    print("\nPASS PROXY: J(20k) <= 1.05*GridOnly AND J(20k) <= 1.02*J(15k)")
    passing = []
    for r in sorted(results, key=lambda x: x["config"]):
        c = r["curve_val_cost"]
        j20, j15 = c.get(args.steps), c.get(15000)
        g1 = j20 <= 1.05 * GRID_ONLY_VAL
        g2 = (j15 is None) or (j20 <= 1.02 * j15)
        vals = sorted(c.values())
        dd = (vals[-1] - vals[0]) / GRID_ONLY_VAL if vals else 0
        ok = g1 and g2
        if ok:
            passing.append(r["config"])
        print(f"  {r['config']:18s} J(20k)={j20:,.0f}  <=1.05*GO:{str(g1):5s}  "
              f"<=1.02*J(15k):{str(g2):5s}  drawdown/GO={dd:.3f}  -> {'PASS' if ok else 'fail'}")

    out = {
        "wall_clock_sec": wall, "n_runs": len(jobs),
        "results": results, "passing_configs": passing,
        "grid_only_val": GRID_ONLY_VAL, "d_main": D_MAIN,
    }
    batch_path = os.path.join("results", "runs", f"{args.run_id}_batch.json")
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)
    with open(batch_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {batch_path}  ({wall/60:.1f} min)")
    print("PASSING:", passing if passing else "none")


if __name__ == "__main__":
    main()
