# `results/` — what is tracked and why

This directory is committed on purpose. It is small (~2.4 MB) and contains the
evidence behind the README table and figures, so a reader does not have to rerun
20 training jobs to check the claim.

## Layout

```
figures/                     PNGs used in the README (unclamped family)
  learning_curves.png        Val cost during training, mean ± spread over 5 seeds
  cost_vs_exceedance.png     cost against mean cap exceedance (the trade-off)
  per_seed_results.png       every seed, not just the mean
  cost_breakdown.png         energy + unserved + wear for each method
  lambda_trajectory.png      dual variable over training (diagnostic, not in README)
  clamped/                   corresponding 5 PNGs for the clamped family
runs/<agent>/                one JSON per (experiment, seed)
  *_seed101.json ...         5 seeds per method (101–105)
  *_grid.json                batch roll-up, not a run
tables/
  runs.csv                   one row per seed run (cost, exceedance, wall clock)
  per_seed_metrics.csv       per-seed diagnostics + gate pass/fail flags
  s2_per_seed.csv            legacy alias of per_seed_metrics.csv
  frozen_baselines.csv       Grid-Only / PSM' reference values
  project_constants.csv      cap, d_main, weights, gate constants
```

Agents are `ppo`, `ppo_lagrangian`, `ppo_residual`, `ppo_shielded`. Each contains
seed runs (101–105) at 300k steps, evaluated deterministically on the 26 Val weeks.

### Experiment families

Two experiment families are tracked:
- **Unclamped (`unclamped_*` / legacy `s3_*`):** `charge_clamp: false`. The original environment
  formulation where battery charging was not physically interlocked against the
  import cap.
- **Clamped (`clamped_*` / `s3v2_*`):** `charge_clamp: true`. Applies the physical
  clamp rate limit → SoC limit → cap clamp in the environment. Battery charging
  cannot induce grid import exceedance beyond available headroom. Additive
  columns `mean_charge_clipped_by_cap_kw` and `cap_clipped_steps_pct` track the clamp's activity.

## Regenerating the tables and figures

From the repository root, in an environment with the requirements installed:

```bash
python tools/report/export_csv.py              # rebuild results/tables/*.csv from runs/
python tools/report/plots.py --family unclamped # rebuild results/figures/*.png
python tools/report/plots.py --family clamped   # rebuild results/figures/clamped/*.png
python tools/status.py                         # compact health/status summary
```

The scripts read only the committed artefacts; they do not retrain and do not
touch the closed Test split.

## What is *not* here

- Raw or processed datasets (`data/raw/`, `data/processed/`) — downloaded at run
  time; see the README.
- Model checkpoints (`checkpoints/`) — large and not needed to check the reported
  metrics.
- Any Test-split result. The reported table is Val only.
