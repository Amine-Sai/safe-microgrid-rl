#!/usr/bin/env python3
"""Run multi-seed experiment grids (clamped, clamped_forecast) with CPU scaling.

Usage:
    python experiments/run_clamped_grid.py --families clamped
    python experiments/run_clamped_grid.py --families clamped,clamped_forecast --workers 4
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

FAMILY_CONFIGS = {
    "clamped": [
        "clamped_ppo",
        "clamped_ppo_lagrangian",
        "clamped_ppo_residual",
        "clamped_ppo_shielded",
    ],
    "clamped_forecast": [
        "clamped_forecast_ppo",
        "clamped_forecast_ppo_lagrangian",
        "clamped_forecast_ppo_residual",
        "clamped_forecast_ppo_shielded",
    ],
}


def _run_single_job(args):
    config_id, seed, total_steps, eval_interval, overwrite = args

    # Single thread per process for clean CPU scaling
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch as th
    th.set_num_threads(1)
    from experiments.train_core import load_config_with_base, run_paths, train_single_method

    cfg = load_config_with_base(config_id)
    cfg["seed"] = seed
    if total_steps > 0:
        cfg["total_steps"] = total_steps
    if eval_interval > 0:
        cfg["eval_interval_steps"] = eval_interval

    method = cfg["method"]
    result_path, ckpt_dir = run_paths(method, cfg, config_id)

    if os.path.exists(result_path) and not overwrite:
        print(f"[SKIP] {config_id} seed={seed} already exists -> {result_path}", flush=True)
        return config_id, seed, 0.0, 0.0, 0.0

    print(f"[START] {config_id} seed={seed} ({cfg['total_steps']} steps) -> {result_path}", flush=True)
    t0 = time.time()
    res = train_single_method(
        method, cfg, ckpt_dir=ckpt_dir, run_name=config_id, allow_overwrite=True
    )
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    val_cost = res["final_checkpoint"]["metrics"]["total_cost_eur"]
    val_exc = res["final_checkpoint"]["metrics"]["mean_exceedance_kw"]
    print(
        f"[DONE] {config_id} seed={seed} in {elapsed:.1f}s | "
        f"cost={val_cost:,.2f} EUR | exc={val_exc:.4f} kW",
        flush=True,
    )
    return config_id, seed, val_cost, val_exc, elapsed


def main():
    parser = argparse.ArgumentParser(description="Multi-seed grid runner for clamped families.")
    parser.add_argument(
        "--families",
        default="clamped",
        help="Comma-separated families: clamped, clamped_forecast, or all.",
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Optional explicit comma-separated config list.",
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Override total_steps (0 = use config setting).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Override eval_interval_steps (0 = use config setting).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-running completed runs.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip export_csv.py and plots.py after training completes.",
    )
    args = parser.parse_args()

    if args.configs:
        config_list = [c.strip() for c in args.configs.split(",") if c.strip()]
        selected_families = ["custom"]
    else:
        raw_fams = [f.strip() for f in args.families.split(",") if f.strip()]
        if "all" in raw_fams:
            selected_families = list(FAMILY_CONFIGS.keys())
        else:
            selected_families = raw_fams

        config_list = []
        for fam in selected_families:
            if fam not in FAMILY_CONFIGS:
                raise ValueError(f"Unknown family {fam!r}. Available: {list(FAMILY_CONFIGS)}")
            config_list.extend(FAMILY_CONFIGS[fam])

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    jobs = [(c, s, args.steps, args.eval_interval, args.overwrite) for c in config_list for s in seeds]

    print("=" * 80)
    print(f"EXPERIMENT GRID: {len(jobs)} jobs across configs: {config_list}")
    print(f"Workers: {args.workers} | Seeds: {seeds}")
    print("=" * 80)

    t_start = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_single_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"FINISHED {len(results)} jobs in {total_time / 60:.2f} minutes")
    print("=" * 80)

    if not args.no_postprocess:
        from tools.report.export_csv import main as export_csv_main
        from tools.report.plots import main as plots_main

        print("\n[POST-PROCESS] Updating summary tables...")
        export_csv_main()

        for fam in selected_families:
            if fam in FAMILY_CONFIGS:
                print(f"[POST-PROCESS] Generating figures for {fam}...")
                sys.argv = ["plots.py", "--family", fam]
                plots_main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
