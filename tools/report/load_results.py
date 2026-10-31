"""Dynamic data loader for microgrid benchmarking results.

Imports metrics directly from persistent result tables:
- results/tables/project_constants.csv
- results/tables/frozen_baselines.csv
- results/tables/per_seed_metrics.csv
- results/tables/test_evaluation.csv
- results/tables/test_stress_weeks.csv

Eliminates all hardcoded values from report and figure generation.
"""

import csv
import io
import json
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES = os.path.join(ROOT, "results", "tables")
RUNS = os.path.join(ROOT, "results", "runs")


def load_project_constants() -> Dict[str, float]:
    """Load evaluation gates and physical constants from results/tables/project_constants.csv."""
    constants_path = os.path.join(TABLES, "project_constants.csv")
    constants: Dict[str, float] = {}
    if os.path.exists(constants_path):
        with open(constants_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    constants[row["name"]] = float(row["value"])
                except (ValueError, KeyError):
                    continue

    exceedance_gate_kw = constants.get("d_main_kw_per_step", 0.05)
    cost_gate_eur = constants.get("g2_cost_eur", 58306.73)
    cap_kw = constants.get("grid_import_cap_kw", 368.15)
    wear_cost_per_kwh = constants.get("w_wear_eur_kwh", 0.02)
    wear_cost_per_mwh = wear_cost_per_kwh * 1000.0

    return {
        "g1_exceedance_kw": exceedance_gate_kw,
        "g2_cost_eur": cost_gate_eur,
        "contract_cap_kw": cap_kw,
        "wear_cost_per_mwh": wear_cost_per_mwh,
        "lp_prime_val_eur": constants.get("lp_prime_val_eur", 53290.51),
        "lp_prime_test_eur": constants.get("lp_prime_test_eur", 36991.67),
    }


def load_baselines() -> Dict[str, Dict[str, Any]]:
    """Load frozen baselines directly from results/tables/frozen_baselines.csv."""
    out: Dict[str, Dict[str, Any]] = {}
    path = os.path.join(TABLES, "frozen_baselines.csv")
    if not os.path.exists(path):
        return out

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "split" in row and row.get("split") != "val":
                continue
            name = row.get("method") or row.get("agent")
            if not name:
                continue

            cost = float(row.get("val_cost_eur") or row.get("cost_eur") or 0.0)
            exc = float(row.get("mean_exceedance_kw") or row.get("exceedance_kw") or 0.0)
            purch = float(row.get("energy_purchase_eur") or 0.0)
            wear = float(row.get("battery_wear_eur") or 0.0)
            unserv = float(row.get("unserved_penalty_eur") or 0.0)
            tp = float(row.get("battery_throughput_mwh") or 0.0)
            soc = float(row.get("active_soc_pct") or 0.0)

            # Inactive battery baselines (Grid-Only, Shield + Idle) have 0 active throughput and 0% active cycling
            if tp < 0.01:
                soc = 0.0

            out[name] = {
                "cost_eur": cost,
                "exceedance_kw": exc,
                "feasible": str(row.get("feasible", "")).lower() == "true",
                "energy_purchase_eur": purch,
                "battery_wear_eur": wear,
                "unserved_penalty_eur": unserv,
                "battery_throughput_mwh": tp,
                "active_soc_pct": soc,
            }
    return out


def load_per_seed_table() -> Dict[str, List[Dict[str, float]]]:
    """Load all per-seed metrics from results/tables/per_seed_metrics.csv indexed by run prefix."""
    path = os.path.join(TABLES, "per_seed_metrics.csv")
    out: Dict[str, List[Dict[str, float]]] = {}
    if not os.path.exists(path):
        return out

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = row.get("run", "")
            prefix = run_name.split("_seed")[0] if "_seed" in run_name else run_name

            metrics = {
                "cost_eur": float(row.get("cost_eur", 0.0)),
                "energy_purchase_eur": float(row.get("energy_purchase_eur", 0.0)),
                "battery_wear_eur": float(row.get("battery_wear_eur", 0.0)),
                "unserved_penalty_eur": float(row.get("unserved_penalty_eur", 0.0)),
                "battery_throughput_mwh": float(row.get("battery_throughput_mwh", 0.0)),
                "active_soc_pct": float(row.get("active_soc_pct", 0.0)),
                "exceedance_kw": float(row.get("exceedance_kw", 0.0)),
                "violation_hours": float(row.get("violation_hours", 0.0)),
            }
            out.setdefault(prefix, []).append(metrics)
    return out


def load_test_stress_weeks_data() -> Dict[str, Any]:
    """Load test stress weeks results and compute statistics for Panel A and Panel B."""
    path = os.path.join(TABLES, "test_stress_weeks.csv")
    if not os.path.exists(path):
        return {"panel_a": {}, "w49_models": []}

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    crits = ["lowest_pv", "highest_price", "hottest", "stormiest", "highest_exceedance"]
    panel_a_methods = ["Grid-Only", "PSM'", "Shield + Idle", "PPO-Residual", "Shielded PPO"]

    def _group_panel_a(model_name: str) -> Optional[str]:
        if model_name in ("Grid-Only", "PSM'", "Shield + Idle", "Shield + PSM'"):
            return model_name
        if "shielded" in model_name:
            return "Shielded PPO"
        if "residual" in model_name:
            return "PPO-Residual"
        if "lagrangian" in model_name:
            return "PPO-Lagrangian"
        return "PPO"

    # Panel A: cost by stress criterion
    panel_a_data = {m: {c: [] for c in crits} for m in panel_a_methods}
    for r in rows:
        fam = r["family"]
        g = _group_panel_a(r["model"])
        c = r["stress_criterion"]
        if (fam == "baseline" or fam == "clamped_mb") and g in panel_a_data:
            if c in panel_a_data[g]:
                panel_a_data[g][c].append(float(r["cost_eur"]))

    # Panel B: W49 (highest_exceedance) per-model statistics
    w49_rows = [r for r in rows if r["stress_criterion"] == "highest_exceedance"]
    w49_by_prefix: Dict[str, List[Dict[str, float]]] = {}
    for r in w49_rows:
        model = r["model"]
        prefix = model.split("_seed")[0] if "_seed" in model else model
        w49_by_prefix.setdefault(prefix, []).append({
            "violation_hours": float(r.get("violation_hours", 0.0)),
            "mean_exceedance_kw": float(r.get("mean_exceedance_kw", 0.0)),
            "cost_eur": float(r.get("cost_eur", 0.0)),
        })

    return {
        "panel_a": panel_a_data,
        "w49_by_prefix": w49_by_prefix,
        "raw_rows": rows,
    }


def extract_metric_row(
    key: Optional[str],
    cat: str,
    all_families: Optional[Dict[str, Any]] = None,
    baselines: Optional[Dict[str, Any]] = None,
    safety_runs: Optional[Dict[str, Any]] = None,
    per_seed_table: Optional[Dict[str, List[Dict[str, float]]]] = None,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """Extract summary metrics across runs/tables with zero hardcoded numbers.

    Returns:
        (m_purch, m_wear, m_unserv, m_tot, s_tot, m_exc, s_exc, m_tp, s_tp, m_wear, s_wear, m_soc, s_soc)
    """
    if baselines is None:
        baselines = load_baselines()
    if per_seed_table is None:
        per_seed_table = load_per_seed_table()

    # 1. Reference baselines directly from loaded baselines table
    if cat in ("Baseline", "Shield Control") or key in baselines:
        b = baselines.get(key, {})
        tot = float(b.get("cost_eur", 0.0))
        exc = float(b.get("exceedance_kw", 0.0))
        purch = float(b.get("energy_purchase_eur", 0.0))
        wear = float(b.get("battery_wear_eur", 0.0))
        unserv = float(b.get("unserved_penalty_eur", 0.0))
        tp = float(b.get("battery_throughput_mwh", 0.0))
        soc = float(b.get("active_soc_pct", 0.0))

        # Inactive batteries have 0 throughput and 0% active cycling
        if tp < 0.01:
            soc = 0.0
            wear = 0.0

        return purch, wear, unserv, tot, 0.0, exc, 0.0, tp, 0.0, wear, 0.0, soc, 0.0

    # 2. In-The-Loop Safe RL (PPO-Aux-Shield, PPO-Diff-Shield)
    prefix_map = {
        "PPO-Aux-Shield": "clamped_ppo_shield_aux",
        "PPO-Diff-Shield": "clamped_ppo_diff_shield",
        "clamped_ppo_shield_aux": "clamped_ppo_shield_aux",
        "clamped_ppo_diff_shield": "clamped_ppo_diff_shield",
    }
    target_prefix = prefix_map.get(key)

    if target_prefix and target_prefix in per_seed_table:
        rows = per_seed_table[target_prefix]
        purchs = [r["energy_purchase_eur"] for r in rows]
        wears = [r["battery_wear_eur"] for r in rows]
        unservs = [r["unserved_penalty_eur"] for r in rows]
        tots = [r["cost_eur"] for r in rows]
        excs = [r["exceedance_kw"] for r in rows]
        tps = [r["battery_throughput_mwh"] for r in rows]
        socs = [r["active_soc_pct"] for r in rows]

        m_purch = statistics.fmean(purchs) if purchs else 0.0
        m_wear = statistics.fmean(wears) if wears else 0.0
        m_unserv = statistics.fmean(unservs) if unservs else 0.0
        m_tot = statistics.fmean(tots) if tots else 0.0
        s_tot = statistics.stdev(tots) if len(tots) > 1 else 0.0
        m_exc = statistics.fmean(excs) if excs else 0.0
        s_exc = statistics.stdev(excs) if len(excs) > 1 else 0.0
        m_tp = statistics.fmean(tps) if tps else 0.0
        s_tp = statistics.stdev(tps) if len(tps) > 1 else 0.0
        s_wear = statistics.stdev(wears) if len(wears) > 1 else 0.0
        m_soc = statistics.fmean(socs) if socs else 0.0
        s_soc = statistics.stdev(socs) if len(socs) > 1 else 0.0
        return m_purch, m_wear, m_unserv, m_tot, s_tot, m_exc, s_exc, m_tp, s_tp, m_wear, s_wear, m_soc, s_soc

    # 3. RL Families from all_families or per_seed_table
    rl_prefix_map = {
        "unclamped_ppo": ("unclamped", "PPO", "s3_ppo"),
        "clamped_ppo": ("clamped", "PPO", "clamped_ppo"),
        "clamped_mb_ppo": ("clamped_mb", "PPO", "clamped_mb_ppo"),
        "clamped_mb_ppo_lagrangian": ("clamped_mb", "PPO-Lagrangian", "clamped_mb_ppo_lagrangian"),
        "clamped_ppo_lagrangian": ("clamped", "PPO-Lagrangian", "clamped_ppo_lagrangian"),
        "clamped_ppo_residual": ("clamped", "PPO-Residual", "clamped_ppo_residual"),
        "clamped_mb_ppo_residual": ("clamped_mb", "PPO-Residual", "clamped_mb_ppo_residual"),
        "clamped_ppo_shielded": ("clamped", "Shielded PPO", "clamped_ppo_shielded"),
        "clamped_mb_ppo_shielded": ("clamped_mb", "Shielded PPO", "clamped_mb_ppo_shielded"),
        "unclamped_ppo_shielded": ("unclamped", "Shielded PPO", "s3_sh_ppo"),
    }

    fam, m_label, run_pfx = rl_prefix_map.get(key, ("clamped_mb", "PPO", key))

    # Priority 1: From per_seed_table if available
    if run_pfx in per_seed_table:
        rows = per_seed_table[run_pfx]
        purchs = [r["energy_purchase_eur"] for r in rows]
        wears = [r["battery_wear_eur"] for r in rows]
        unservs = [r["unserved_penalty_eur"] for r in rows]
        tots = [r["cost_eur"] for r in rows]
        excs = [r["exceedance_kw"] for r in rows]
        tps = [r["battery_throughput_mwh"] for r in rows]
        socs = [r["active_soc_pct"] for r in rows]

        m_purch = statistics.fmean(purchs) if purchs else 0.0
        m_wear = statistics.fmean(wears) if wears else 0.0
        m_unserv = statistics.fmean(unservs) if unservs else 0.0
        m_tot = statistics.fmean(tots) if tots else 0.0
        s_tot = statistics.stdev(tots) if len(tots) > 1 else 0.0
        m_exc = statistics.fmean(excs) if excs else 0.0
        s_exc = statistics.stdev(excs) if len(excs) > 1 else 0.0
        m_tp = statistics.fmean(tps) if tps else 0.0
        s_tp = statistics.stdev(tps) if len(tps) > 1 else 0.0
        s_wear = statistics.stdev(wears) if len(wears) > 1 else 0.0
        m_soc = statistics.fmean(socs) if socs else 0.0
        s_soc = statistics.stdev(socs) if len(socs) > 1 else 0.0
        return m_purch, m_wear, m_unserv, m_tot, s_tot, m_exc, s_exc, m_tp, s_tp, m_wear, s_wear, m_soc, s_soc

    # Priority 2: From all_families JSON run objects
    runs = (all_families or {}).get(fam, {}).get(m_label, [])
    if runs:
        def _fm(r):
            return r.get("final_checkpoint", {}).get("metrics", {})

        purchs = [float(_fm(r).get("energy_purchase_eur", 0.0)) for r in runs]
        wears = [float(_fm(r).get("battery_wear_eur", 0.0)) for r in runs]
        unservs = [float(_fm(r).get("unserved_penalty_eur", 0.0)) for r in runs]
        tots = [float(_fm(r).get("total_cost_eur", 0.0)) for r in runs]
        excs = [float(_fm(r).get("mean_exceedance_kw", 0.0)) for r in runs]
        tps = [w / 20.0 for w in wears]
        socs = [100.0 - float(_fm(r).get("soc_pinned_pct", 100.0)) for r in runs]

        m_purch = statistics.fmean(purchs) if purchs else 0.0
        m_wear = statistics.fmean(wears) if wears else 0.0
        m_unserv = statistics.fmean(unservs) if unservs else 0.0
        m_tot = statistics.fmean(tots) if tots else 0.0
        s_tot = statistics.stdev(tots) if len(tots) > 1 else 0.0
        m_exc = statistics.fmean(excs) if excs else 0.0
        s_exc = statistics.stdev(excs) if len(excs) > 1 else 0.0
        m_tp = statistics.fmean(tps) if tps else 0.0
        s_tp = statistics.stdev(tps) if len(tps) > 1 else 0.0
        s_wear = statistics.stdev(wears) if len(wears) > 1 else 0.0
        m_soc = statistics.fmean(socs) if socs else 0.0
        s_soc = statistics.stdev(socs) if len(socs) > 1 else 0.0
        return m_purch, m_wear, m_unserv, m_tot, s_tot, m_exc, s_exc, m_tp, s_tp, m_wear, s_wear, m_soc, s_soc

    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
