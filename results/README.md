# `results/` — what is tracked and why

This directory is committed on purpose. It is small (~2.4 MB) and contains the
evidence behind the README table and figures, so a reader does not have to rerun
20 training jobs to check the claim.

## Layout

```
figures/
  unclamped/                 5 PNGs for unclamped family (17-D)
  clamped/                   5 PNGs for clamped family (17-D)
  clamped_mb/                5 PNGs for clamped minibatch control family (17-D)
  clamped_forecast/          5 PNGs for clamped + 6h forecast family (23-D)
  comparison/                4 cross-family comparison PNGs:
    unified_pareto.png       Unified Pareto frontier (Cost vs. Exceedance across environments)
    grouped_cost_breakdown.png Grouped stacked cost breakdown & exceedance across environments
    targeted_learning_curves.png Targeted learning curves for competitive methods (Residual & Shielded)
    stress_weeks_comparison.png Test split stress-week evaluation (Weather weeks vs. Cap-Binding 2019-W49)
runs/<agent>/                one JSON per (experiment, seed)
  *_seed101.json ...         5 seeds per method (101–105)
  *_grid.json                batch roll-up, not a run
tables/
  runs.csv                   one row per seed run (cost, exceedance, wall clock)
  per_seed_metrics.csv       per-seed diagnostics, gate pass/fail flags, strict_feasible
  s2_per_seed.csv            legacy alias of per_seed_metrics.csv
  frozen_baselines.csv       Grid-Only, PSM', and exact LP' hindsight oracle values
  project_constants.csv      cap, d_main, weights, LP' bounds, gate constants
  test_evaluation.csv        locked one-shot evaluation on the 2019 H2 Test split
  test_stress_weeks.csv      evaluation on the 5 Test stress weeks (including cap-binding 2019-W49)
```

Agents are `ppo`, `ppo_lagrangian`, `ppo_residual`, `ppo_shielded`. Each contains
seed runs (101–105) at 300k steps, evaluated deterministically on the 26 Val weeks.

### Experiment families

Four experiment families are fully evaluated (80 runs total):
- **Unclamped (`unclamped_*` / legacy `s3_*`):** `charge_clamp: false`. The original environment
  formulation where battery charging was not physically interlocked against the
  import cap.
- **Clamped (`clamped_*` / `s3v2_*`):** `charge_clamp: true`. Applies the physical
  clamp rate limit → SoC limit → cap clamp in the environment. Battery charging
  cannot induce grid import exceedance beyond available headroom. Additive
  columns `mean_charge_clipped_by_cap_kw` and `cap_clipped_steps_pct` track the clamp's activity.
- **Clamped Minibatch (`clamped_mb_*`):** `charge_clamp: true` with minibatch updates, 5 value epochs,
  and raw action storage, on the base 17-D state space (no lookahead). Serves as the ablation control to
  disentangle PPO optimization improvements from 6h lookahead features.
- **Clamped + Forecast (`clamped_forecast_*`):** `charge_clamp: true` with `forecast_horizon_hours: 6`.
  Extends the observation space from 17-D to 23-D with rolling 6-hour PV and load lookaheads.

## Regenerating the tables and figures

From the repository root, in an environment with the requirements installed:

```bash
python tools/report/lp_oracle.py --split both   # solve perfect-foresight LP' theoretical bounds
python tools/report/export_csv.py               # rebuild results/tables/*.csv from runs/
python tools/report/plots.py --family all        # rebuild all per-family and comparison figures
python tools/report/plots.py --family comparison # rebuild only cross-family comparison figures
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
