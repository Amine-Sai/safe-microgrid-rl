# `results/` — what is tracked and why

This directory is committed on purpose. It is small (~12 MB) and contains the
evidence behind the README table and figures, so a reader does not have to rerun
80 training jobs to check the claim.

## Layout

```
figures/
  overview_and_pareto/       Unified Pareto frontier & macro performance dashboards:
    unified_pareto.png       Unified Pareto frontier (Cost vs. Exceedance across environments & shields)
    <family>_family_summary.png 2x2 Macro performance dashboards per experimental family
    <family>_per_seed_results.png Per-seed cost and exceedance distributions vs. G1/G2 gates
  battery_dynamics/          Battery utilization and time-series dispatch:
    battery_usage_comparison.png Battery throughput (MWh) & active unpinned SoC operating time (%)
    w49_dispatch_trajectory.png 168-Hour dispatch trajectory on cap-binding week W49
  safety_and_exceedance/     Constraint enforcement and safety architecture spectrum:
    safety_architectures_comparison.png 4-Panel safety & shielding spectrum (Baselines, Post-hoc, In-the-loop)
    stress_weeks_comparison.png Test stress-week evaluation (Weather weeks vs. Cap-Binding W49)
    <family>_cost_vs_exceedance.png Cost vs. exceedance frontier per family
  economic_breakdown/        Economic cost composition and tariff impacts:
    grouped_cost_breakdown.png Grouped stacked cost breakdown (Energy, Wear, Unserved vs G1 & Throughput)
    <family>_cost_breakdown.png Stacked cost composition per family
  training_dynamics/         Convergence and constraint optimization dynamics:
    safety_learning_curves.png Safety learning curves (Cost, unserved penalty, wear progression)
    learning_and_constraint_dynamics.png Multi-panel dynamics (Cost, log exceedance, dual multiplier λ)
    targeted_learning_curves.png Targeted learning curves for competitive methods (Residual & Shielded)
    <family>_learning_curves.png Validation cost and exceedance progression over 300k steps
    <family>_lambda_trajectory.png Dual multiplier λ trajectories for constrained variants
runs/
  test_lock.json             lock token for one-shot evaluation on the Test split
  <agent>/                   one JSON per (experiment, seed)
    *_seed101.json ...       5 seeds per method (101–105)
    *_grid.json              batch roll-up, not a run
tables/
  runs.csv                   one row per seed run (cost, exceedance, wall clock)
  per_seed_metrics.csv       per-seed diagnostics, gate pass/fail flags, strict_feasible
  s2_per_seed.csv            legacy alias of per_seed_metrics.csv
  frozen_baselines.csv       Grid-Only, PSM', and exact LP' hindsight oracle values
  project_constants.csv      cap, d_main, weights, LP' bounds, gate constants
  test_evaluation.csv        locked one-shot evaluation on the 2019 H2 Test split
  test_stress_weeks.csv      evaluation on the 5 Test stress weeks (including cap-binding W49)
```

Agents are `ppo`, `ppo_lagrangian`, `ppo_residual`, `ppo_shielded`. Each contains
seed runs (101–105) at 300k steps, evaluated deterministically on the 26 Val weeks.

### Legacy Aliases and Exploratory Agents

- **Legacy Config Aliases:** In `configs/config.yaml` and `results/runs/`, legacy naming conventions
  (`s3_*` = `unclamped_*`, `s3v2_*` = `clamped_*`, as well as early exploratory stage-1/2 configs `r1`, `r2`, `r3`, `s2_*`, `v2_*`)
  are preserved for backward-compatible reproducibility with committed run JSON records and historical tables.
- **`PPOPotentialShapingAgent`:** (`agents/shaping.py`, configs `s2_ppo_pbrs` and `s2_ppo_lagrangian_pbrs`) implements
  potential-based reward shaping ($\Phi(s) = \kappa \cdot \text{SoC}$) from early stage-2 exploration. Because SoC shaping alone
  did not prevent grid-cap violations, it was superseded by the 4-method × 4-family main grid and is retained for unit-test coverage
  and ablation completeness.

### Experiment families

Three experiment families are fully evaluated (60 runs total):
- **Unclamped (`unclamped_*` / legacy `s3_*`):** `charge_clamp: false`. The original environment
  formulation where battery charging was not physically interlocked against the
  import cap.
- **Clamped (`clamped_*` / `s3v2_*`):** `charge_clamp: true`. Applies the physical
  clamp rate limit → SoC limit → cap clamp in the environment. Battery charging
  cannot induce grid import exceedance beyond available headroom. Additive
  columns `mean_charge_clipped_by_cap_kw` and `cap_clipped_steps_pct` track the clamp's activity.
- **Clamped Minibatch (`clamped_mb_*`):** `charge_clamp: true` with minibatch updates, 5 value epochs,
  and raw action storage, on the base 17-D state space (no lookahead). Serves as the un-confounded ablation control to
  disentangle PPO optimization improvements from 6h lookahead features.

## Regenerating the tables and figures

From the repository root, in an environment with the requirements installed:

```bash
python tools/report/lp_oracle.py --split both   # solve perfect-foresight LP' theoretical bounds
python tools/report/export_csv.py               # rebuild results/tables/*.csv from runs/
python tools/report/plots.py                     # rebuild all figures into thematic directories
python tools/report/plots.py --family comparison # rebuild only cross-family comparison figures
python tools/report/plot_stress_trajectory.py   # rebuild 168-hour dispatch trajectory for W49
python tools/status.py                          # compact health/status summary
```

The scripts read only the committed artefacts; they do not retrain and do not
touch the closed Test split.

## What is *not* here

- Raw or processed datasets (`data/raw/`, `data/processed/`) — downloaded at run
  time; see the README.
- Model checkpoints (`checkpoints/`) — large and not needed to check the reported
  metrics.
- Full test-split timeseries trajectories — only aggregate metrics and stress-week evaluations are stored under results/tables/.
