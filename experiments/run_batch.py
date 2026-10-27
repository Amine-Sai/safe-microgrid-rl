import argparse
import csv
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


def _load_reference_constants() -> tuple[float, float]:
    """Load Grid-Only validation cost and d_main exceedance gate from results/tables/."""
    grid_only_val_eur = 45337.92170812376
    exceedance_gate_kw = 0.05

    baselines_csv = os.path.join(REPO_ROOT, "results", "tables", "frozen_baselines.csv")
    if os.path.exists(baselines_csv):
        with open(baselines_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("method") == "Grid-Only" and row.get("val_cost_eur"):
                    grid_only_val_eur = float(row["val_cost_eur"])

    constants_csv = os.path.join(REPO_ROOT, "results", "tables", "project_constants.csv")
    if os.path.exists(constants_csv):
        with open(constants_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("name") == "d_main_kw_per_step" and row.get("value"):
                    exceedance_gate_kw = float(row["value"])

    return grid_only_val_eur, exceedance_gate_kw


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
    res = train_single_method(
        method,
        cfg,
        ckpt_dir=ckpt_dir,
        run_name=run_name,
        allow_overwrite=True,
    )

    eval_checkpoints = res.get("eval_checkpoints", {})
    curve_val_cost = {
        int(k.split("_")[0]): v["total_cost_eur"] for k, v in eval_checkpoints.items()
    }
    curve_val_exc = {
        int(k.split("_")[0]): v["mean_exceedance_kw"] for k, v in eval_checkpoints.items()
    }
    lambda_summary = res["lambda_summary"]

    return {
        "config": cfg_name,
        "method": method,
        "seed": seed,
        "final_val_cost": res["final_checkpoint"]["metrics"]["total_cost_eur"],
        "final_val_exc": res["final_checkpoint"]["metrics"]["mean_exceedance_kw"],
        "final_unserved_pct": res["final_checkpoint"]["metrics"]["unserved_energy_pct"],
        "final_violation_rate_pct": res["final_checkpoint"]["metrics"]["violation_rate_pct"],
        "curve_val_cost": curve_val_cost,
        "curve_val_exc": curve_val_exc,
        "lambda_end": lambda_summary["lambda_end"],
        "mean_shed_cmd": res["degeneracy_check"].get("mean_shed_cmd"),
        "result_path": result_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True, help="comma list of config stems")
    parser.add_argument("--method", default="ppo", help="agent method (PPO family)")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", default="batch", help="run id prefix for the batch")
    args = parser.parse_args()

    grid_only_val_eur, exceedance_gate_kw = _load_reference_constants()

    config_names = args.configs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [
        (cfg_name, args.method, seed, args.steps, args.eval_interval, f"{args.run_id}_{cfg_name}")
        for cfg_name in config_names
        for seed in seeds
    ]

    print(
        f"BATCH: {len(config_names)} configs x {len(seeds)} seeds = {len(jobs)} runs, "
        f"workers={args.workers}"
    )
    from concurrent.futures import ProcessPoolExecutor, as_completed

    start_time = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_job, job): job for job in jobs}
        for idx, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                run_result = future.result()
                results.append(run_result)
                print(
                    f"  [{idx}/{len(jobs)}] {run_result['config']:18s} seed={run_result['seed']} "
                    f"J({args.steps})={run_result['final_val_cost']:,.2f} "
                    f"exc={run_result['final_val_exc']:.4f} "
                    f"lam={run_result['lambda_end']}",
                    flush=True,
                )
            except Exception as err:
                print(f"  [{idx}/{len(jobs)}] {job[0]} seed={job[1]} FAILED: {err}", flush=True)
    wall_clock_sec = time.time() - start_time

    print("\nPASS PROXY: J(end) <= 1.05*GridOnly AND J(end) <= 1.02*J(15k)")
    passing_configs = []
    for run_result in sorted(results, key=lambda x: x["config"]):
        cost_curve = run_result["curve_val_cost"]
        cost_at_end = cost_curve.get(args.steps)
        cost_at_15k = cost_curve.get(15000)
        within_grid_only_bound = (
            cost_at_end is not None and cost_at_end <= 1.05 * grid_only_val_eur
        )
        within_stability_bound = (cost_at_15k is None) or (
            cost_at_end is not None and cost_at_end <= 1.02 * cost_at_15k
        )
        sorted_costs = sorted(cost_curve.values())
        drawdown_ratio = (
            (sorted_costs[-1] - sorted_costs[0]) / grid_only_val_eur if sorted_costs else 0.0
        )
        passed = within_grid_only_bound and within_stability_bound
        if passed:
            passing_configs.append(run_result["config"])
        print(
            f"  {run_result['config']:18s} J(end)={cost_at_end or 0.0:,.0f}  "
            f"<=1.05*GO:{str(within_grid_only_bound):5s}  "
            f"<=1.02*J(15k):{str(within_stability_bound):5s}  "
            f"drawdown/GO={drawdown_ratio:.3f}  -> {'PASS' if passed else 'fail'}"
        )

    output_payload = {
        "wall_clock_sec": wall_clock_sec,
        "n_runs": len(jobs),
        "results": results,
        "passing_configs": passing_configs,
        "grid_only_val": grid_only_val_eur,
        "d_main": exceedance_gate_kw,
    }
    batch_path = os.path.join("results", "runs", f"{args.run_id}_batch.json")
    os.makedirs(os.path.dirname(batch_path), exist_ok=True)
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nwrote {batch_path}  ({wall_clock_sec / 60:.1f} min)")
    print("PASSING:", passing_configs if passing_configs else "none")


if __name__ == "__main__":
    main()
