# `results/` — what is tracked and why

This directory is committed on purpose. It is small (~2.4 MB) and contains the
evidence behind the README table and figures, so a reader does not have to rerun
20 training jobs to check the claim.

## Layout

```
figures/                     PNGs used in the README
  learning_curves.png        Val cost during training, mean ± spread over 5 seeds
  cost_vs_exceedance.png     cost against mean cap exceedance (the trade-off)
  per_seed_results.png       every seed, not just the mean
  cost_breakdown.png         energy + unserved + wear for each method
  lambda_trajectory.png      dual variable over training (diagnostic, not in README)
runs/<agent>/                one JSON per (experiment, seed)
  *_seed101.json ...         5 seeds per method (101–105)
  *_grid.json                batch roll-up, not a run
tables/
  runs.csv                   one row per seed run (cost, exceedance, wall clock)
  s2_per_seed.csv            per-seed diagnostics + G1/G2 gate flags
  frozen_baselines.csv       Grid-Only / PSM' reference values
  project_constants.csv      cap, d_main, weights, gate constants
```

Agents are `ppo`, `ppo_lagrangian`, `ppo_residual`, `ppo_shielded`. Each contains
5 seed runs (101–105) at 300k steps, evaluated deterministically on the 26 Val
weeks.

## Regenerating the tables and figures

From the repository root, in an environment with the requirements installed:

```bash
python tools/report/export_csv.py    # rebuild results/tables/*.csv from runs/
python tools/report/plots.py         # rebuild results/figures/*.png from tables + runs
python tools/status.py               # compact health/status summary
```

The scripts read only the committed artefacts; they do not retrain and do not
touch the closed Test split.

## What is *not* here

- Raw or processed datasets (`data/raw/`, `data/processed/`) — downloaded at run
  time; see the README.
- Model checkpoints (`checkpoints/`) — large and not needed to check the reported
  metrics.
- Any Test-split result. The reported table is Val only.
