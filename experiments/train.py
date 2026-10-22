import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_config(path):
    from experiments.train_core import load_config_with_base
    return load_config_with_base(path)


def main():
    ap = argparse.ArgumentParser(description="train one (method, seed) run")
    ap.add_argument("--config", required=True, help="YAML with all settings, or an experiment id")
    ap.add_argument("--method", default=None, help="override config `method`")
    ap.add_argument("--seed", type=int, default=None, help="override config `seed`")
    ap.add_argument("--run-name", default="", help="extra run id (default: the experiment id)")
    ap.add_argument("--out", default=None, help="write result JSON here")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing result / checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.method:
        cfg["method"] = args.method
    if args.seed is not None:
        cfg["seed"] = args.seed

    method = cfg.get("method")
    if not method:
        ap.error("config must set `method` (or pass --method)")

    from experiments.train_core import train_single_method, resolve_run_name, run_paths

    run_id = resolve_run_name(args.config, args.run_name)
    result_path, ckpt_dir = run_paths(method, cfg, run_id)
    if os.path.exists(result_path) and not args.overwrite:
        ap.error(f"{result_path} already exists - pass --overwrite to replace it")

    res = train_single_method(method, cfg, ckpt_dir=ckpt_dir, run_name=run_id,
                              allow_overwrite=args.overwrite)

    out = args.out or result_path
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
