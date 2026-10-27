import csv
import glob
import json
import os
import sys

TABLES = os.path.join("results", "tables")
RUNS = os.path.join("results", "runs")


def load_project_constants():
    """Load evaluation gates and physical constants from results/tables/project_constants.csv."""
    constants_path = os.path.join(TABLES, "project_constants.csv")
    constants = {}
    if os.path.exists(constants_path):
        with open(constants_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                constants[row["name"]] = float(row["value"])
    exceedance_gate_kw = constants.get("d_main_kw_per_step", 0.05)
    cost_gate_eur = constants.get("g2_cost_eur", 58308.55)
    return exceedance_gate_kw, cost_gate_eur


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
    for run_file_path in run_files():
        try:
            with open(run_file_path, encoding="utf-8") as run_file:
                run_data = json.load(run_file)
        except Exception:
            continue
        if not isinstance(run_data, dict) or "method" not in run_data:
            continue
        agent = os.path.basename(os.path.dirname(run_file_path))
        cfg = os.path.basename(run_file_path)[:-5]
        run_id = "%s/%s" % (agent, cfg)
        if run_id in seen:
            continue
        seen.add(run_id)
        final_checkpoint = run_data.get("final_checkpoint") or {}
        checkpoint_metrics = (
            final_checkpoint.get("metrics") if isinstance(final_checkpoint, dict) else None
        ) or {}

        cap_clip = checkpoint_metrics.get(
            "mean_charge_clipped_by_cap_kw", checkpoint_metrics.get("charge_clipped_kw_mean")
        )
        soc_clip = checkpoint_metrics.get("mean_charge_clipped_by_soc_kw")
        cap_pct = checkpoint_metrics.get(
            "cap_clipped_steps_pct", checkpoint_metrics.get("clipped_pct")
        )

        rows.append({
            "run_id": run_id,
            "agent": agent,
            "method": run_data["method"],
            "steps": run_data.get("total_steps"),
            "seed": run_data.get("seed"),
            "val_cost_eur": checkpoint_metrics.get("total_cost_eur"),
            "exceedance_kw": checkpoint_metrics.get("mean_exceedance_kw"),
            "violation_rate_pct": checkpoint_metrics.get("violation_rate_pct"),
            "mean_charge_clipped_by_cap_kw": cap_clip,
            "mean_charge_clipped_by_soc_kw": soc_clip,
            "cap_clipped_steps_pct": cap_pct,
            # Legacy metric aliases
            "charge_clipped_kw_mean": cap_clip,
            "clipped_pct": cap_pct,
            "lambda": (run_data.get("lambda_summary") or {}).get("lambda_end"),
            "wall_clock_seconds": run_data.get("wall_clock_sec"),
            "result_path": run_file_path.replace("\\", "/"),
        })
    return rows


def per_seed_table():
    """Per-seed metrics table with gate pass/fail flags.

    Written to results/tables/per_seed_metrics.csv (and mirrored to s2_per_seed.csv).
    """
    exceedance_gate_kw, cost_gate_eur = load_project_constants()
    rows = []
    for run_file_path in run_files():
        try:
            with open(run_file_path, encoding="utf-8") as run_file:
                run_data = json.load(run_file)
        except Exception:
            continue
        if not isinstance(run_data, dict) or "final_checkpoint" not in run_data:
            continue
        checkpoint_metrics = (run_data["final_checkpoint"] or {}).get("metrics") or {}
        if not checkpoint_metrics or checkpoint_metrics.get("total_cost_eur") is None:
            continue
        lambda_summary = run_data.get("lambda_summary") or {}
        traj = (
            lambda_summary.get("trajectory")
            or lambda_summary.get("trajectory_sample")
            or []
        )
        lams = [x.get("lambda") for x in traj if isinstance(x, dict)]
        cost = checkpoint_metrics["total_cost_eur"]
        exc = checkpoint_metrics.get("mean_exceedance_kw")
        agent = os.path.basename(os.path.dirname(run_file_path))

        cap_clip = checkpoint_metrics.get(
            "mean_charge_clipped_by_cap_kw", checkpoint_metrics.get("charge_clipped_kw_mean")
        )
        soc_clip = checkpoint_metrics.get("mean_charge_clipped_by_soc_kw")
        cap_pct = checkpoint_metrics.get(
            "cap_clipped_steps_pct", checkpoint_metrics.get("clipped_pct")
        )

        passes_exceedance_gate = bool(exc is not None and exc <= exceedance_gate_kw)
        passes_cost_gate = bool(cost <= cost_gate_eur)

        rows.append({
            "run": os.path.basename(run_file_path)[:-5],
            "agent": agent,
            "method": run_data.get("method"),
            "seed": run_data.get("seed"),
            "steps": run_data.get("total_steps"),
            "cost_eur": round(cost, 2),
            "unserved_pct": checkpoint_metrics.get("unserved_energy_pct"),
            "exceedance_kw": exc,
            "violation_rate_pct": checkpoint_metrics.get("violation_rate_pct"),
            "mean_charge_clipped_by_cap_kw": cap_clip,
            "mean_charge_clipped_by_soc_kw": soc_clip,
            "cap_clipped_steps_pct": cap_pct,
            "lambda_start": lambda_summary.get("lambda_start"),
            "lambda_end": lambda_summary.get("lambda_end"),
            "lambda_max": (round(max(lams), 4) if lams else None),
            "saturated_frac": checkpoint_metrics.get("frac_saturated"),
            "soc_pinned_pct": checkpoint_metrics.get("soc_pinned_pct"),
            "passes_exceedance_gate": passes_exceedance_gate,
            "passes_cost_gate": passes_cost_gate,
            # Legacy column aliases
            "G1_pass": passes_exceedance_gate,
            "G2_pass": passes_cost_gate,
            "charge_clipped_kw_mean": cap_clip,
            "clipped_pct": cap_pct,
        })
    if not rows:
        print("no per-seed run JSONs matched")
        return None

    paths = [
        os.path.join(TABLES, "per_seed_metrics.csv"),
        os.path.join(TABLES, "s2_per_seed.csv"),
    ]
    for table_path in paths:
        with open(table_path, "w", newline="") as table_file:
            writer = csv.DictWriter(table_file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {table_path}  ({len(rows)} per-seed rows, no means)")
    return paths[0]


def main():
    os.makedirs(TABLES, exist_ok=True)
    written = []

    rows = run_rows()
    p = os.path.join(TABLES, "runs.csv")
    with open(p, "w", newline="") as f:
        flds = [
            "run_id", "agent", "method", "steps", "seed", "val_cost_eur",
            "exceedance_kw", "violation_rate_pct", "lambda", "wall_clock_seconds",
            "result_path", "mean_charge_clipped_by_cap_kw", "mean_charge_clipped_by_soc_kw",
            "cap_clipped_steps_pct", "charge_clipped_kw_mean", "clipped_pct",
        ]
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
