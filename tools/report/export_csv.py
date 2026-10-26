import csv
import glob
import json
import os
import sys

TABLES = os.path.join("results", "tables")
RUNS = os.path.join("results", "runs")


def run_files():
    """Every per-run result JSON, i.e. results/runs/<agent>/<run id>_seed<S>.json.

    The grid roll-ups (`*_grid.json`) summarise a batch, they are not runs.
    """
    files = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*", "*.json"))):
        if os.path.basename(f).endswith("_grid.json"):
            continue
        files.append(f)
    return files


def run_rows():
    """One row per run found under results/runs/<agent>/."""
    rows, seen = [], set()
    for f in run_files():
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict) or "method" not in d:
            continue
        agent = os.path.basename(os.path.dirname(f))
        cfg = os.path.basename(f)[:-5]
        run_id = "%s/%s" % (agent, cfg)
        if run_id in seen:
            continue
        seen.add(run_id)
        fc = d.get("final_checkpoint") or {}
        m = fc.get("metrics") if isinstance(fc, dict) else None
        m = m or {}
        rows.append({
            "run_id": run_id, "agent": agent, "method": d["method"],
            "steps": d.get("total_steps"),
            "seed": d.get("seed"),
            "val_cost_eur": m.get("total_cost_eur"),
            "exceedance_kw": m.get("mean_exceedance_kw"),
            "violation_rate_pct": m.get("violation_rate_pct"),
            "lambda": (d.get("lambda_summary") or {}).get("lambda_end"),
            "wall_clock_seconds": d.get("wall_clock_sec"),
            "result_path": f.replace("\\", "/"),
        })
    return rows


def per_seed_table():
    """Per-seed S2 + rung table with G1/G2 flags. No means.

    Columns: run, agent, method, seed, cost, unserved, exc, viol, lam start/end/max,
    |a|>=0.99 fraction, SoC-pin fraction, G1 flag, G2 flag. G1 = exc <= 0.05 (d_main),
    G2 = cost <= 58,309 (1.05*PSM'). Written to results/tables/s2_per_seed.csv.
    """
    G1_EXC, G2_COST = 0.05, 1.05 * 55531.95
    rows = []
    for f in run_files():
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict) or "final_checkpoint" not in d:
            continue
        m = (d["final_checkpoint"] or {}).get("metrics") or {}
        if not m or m.get("total_cost_eur") is None:
            continue
        lam = d.get("lambda_summary") or {}
        traj = lam.get("trajectory") or lam.get("trajectory_sample") or []
        lams = [x.get("lambda") for x in traj if isinstance(x, dict)]
        cost = m["total_cost_eur"]
        exc = m.get("mean_exceedance_kw")
        agent = os.path.basename(os.path.dirname(f))
        rows.append({
            "run": os.path.basename(f)[:-5], "agent": agent, "method": d.get("method"),
            "seed": d.get("seed"), "steps": d.get("total_steps"),
            "cost_eur": round(cost, 2),
            "unserved_pct": m.get("unserved_energy_pct"),
            "exceedance_kw": exc,
            "violation_rate_pct": m.get("violation_rate_pct"),
            "lambda_start": lam.get("lambda_start"), "lambda_end": lam.get("lambda_end"),
            "lambda_max": (round(max(lams), 4) if lams else None),
            "saturated_frac": m.get("frac_saturated"),
            "soc_pinned_pct": m.get("soc_pinned_pct"),
            "G1_pass": (exc is not None and exc <= G1_EXC),
            "G2_pass": (cost <= G2_COST),
        })
    p = os.path.join(TABLES, "s2_per_seed.csv")
    if not rows:
        print("no per-seed run JSONs matched - %s left unchanged" % p)
        return None
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p}  ({len(rows)} per-seed rows, no means)")
    return p


def main():
    os.makedirs(TABLES, exist_ok=True)
    written = []

    rows = run_rows()
    p = os.path.join(TABLES, "runs.csv")
    with open(p, "w", newline="") as f:
        flds = ["run_id", "agent", "method", "steps", "seed", "val_cost_eur",
                "exceedance_kw", "violation_rate_pct", "lambda", "wall_clock_seconds",
                "result_path"]
        w = csv.DictWriter(f, fieldnames=flds, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    written.append(p)
    print(f"wrote {p}  ({len(rows)} runs)")
    written.append(per_seed_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
