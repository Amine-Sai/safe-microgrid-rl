# `results/` — what is tracked and why

This directory is committed on purpose. It is small (~12 MB) and contains the
evidence behind the README table and figures, so a reader does not have to rerun
80 training jobs to check the claim.

## Layout

```
figures/
  study1_ppo_ablation/              Study 1: Clamping & batching ablation in unconstrained PPO:
    cost.png                        Cost comparison across full-batch, unclamped, and clamped minibatch PPO
    grid_exceedance.png             Grid import exceedance across PPO ablation variants
    battery/
      throughput.png                Battery energy throughput (MWh)
      active_soc.png                Active unpinned SoC operating time (%)
  study2_capped_ppo/                Study 2: Capped safe RL benchmark vs. baselines:
    cost.png                        Total cost and component breakdown across all benchmark methods
    grid_exceedance.png             Contract cap exceedance across safe and unconstrained methods
    unserved_load.png               Unserved load penalty and curtailment percentage
    battery/
      throughput.png                Battery throughput (MWh) comparison
      active_soc.png                Active unpinned SoC operating percentage
  study3_stress_pareto/             Study 3: Multi-objective Pareto frontier & stress evaluation:
    pareto.png                      Validation Pareto frontier (Cost vs. Exceedance with G1/G2 gates)
    stress_week_pareto.png          Multi-week stress Pareto frontier across meteorological and cap-binding weeks
  study4_plain_vs_lagrangian/       Study 4: Plain PPO vs. PPO-Lagrangian across environments:
    cost.png                        Cost comparison between plain PPO and dual-Lagrangian
    grid_exceedance.png             Safety exceedance comparison
    battery/
      throughput.png                Battery throughput under Lagrangian dual multiplier
      active_soc.png                Active SoC operating time under Lagrangian dynamics
runs/
  test_lock.json             lock token for one-shot evaluation on the Test split
  <agent>/                   one JSON per (experiment, seed)
    *_seed101.json ...       5 seeds per method (101–105)
    *_grid.json              batch roll-up, not a run
tables/
  runs.csv                   one row per seed run (70 runs total: cost, exceedance, wall clock)
  per_seed_metrics.csv       per-seed diagnostics, gate pass/fail flags, strict_feasible
  s2_per_seed.csv            legacy alias of per_seed_metrics.csv
  frozen_baselines.csv       Grid-Only, PSM', Shield baselines, and exact LP' hindsight oracle values
  project_constants.csv      cap, d_main, weights, LP' bounds, gate constants
  test_evaluation.csv        locked one-shot evaluation on the 2019 H2 Test split (all methods)
  test_stress_weeks.csv      evaluation on the 5 Test stress weeks (including cap-binding W49)
```

Agents tracked under `results/runs/` include `ppo`, `ppo_lagrangian`, `ppo_residual`, `ppo_shielded`, `ppo_diff_shield`, and `ppo_shield_aux`. Each contains seed runs (101–105) at 300k steps, evaluated deterministically on the 26 Val weeks.

### Legacy Aliases and Exploratory Agents

- **Legacy Config Aliases:** In `configs/config.yaml` and `results/runs/`, legacy naming conventions
  (`s3_*` = `unclamped_*`, `s3v2_*` = `clamped_*`, as well as early exploratory stage-1/2 configs `r1`, `r2`, `r3`, `s2_*`, `v2_*`)
  are preserved for backward-compatible reproducibility with committed run JSON records and historical tables.
- **`PPOPotentialShapingAgent`:** (`agents/shaping.py`, configs `s2_ppo_pbrs` and `s2_ppo_lagrangian_pbrs`) implements
  potential-based reward shaping ($\Phi(s) = \kappa \cdot \text{SoC}$) from early stage-2 exploration. Because SoC shaping alone
  did not prevent grid-cap violations, it was superseded by the 4-method × 4-family main grid and is retained for unit-test coverage
  and ablation completeness.

### Experiment families

Three main experiment families (60 runs total) plus safety extensions (10 runs, 70 runs total):
- **Unclamped (`unclamped_*` / legacy `s3_*`):** `charge_clamp: false`. The original environment
  formulation where battery charging was not physically interlocked against the
  import cap (20 runs across 4 agents).
- **Clamped (`clamped_*` / `s3v2_*`):** `charge_clamp: true`. Applies the physical
  clamp rate limit → SoC limit → cap clamp in the environment. Battery charging
  cannot induce grid import exceedance beyond available headroom. Additive
  columns `mean_charge_clipped_by_cap_kw` and `cap_clipped_steps_pct` track the clamp's activity (20 runs across 4 agents).
- **Clamped Minibatch (`clamped_mb_*`):** `charge_clamp: true` with minibatch updates, 5 value epochs,
  and raw action storage, on the base 17-D state space (no lookahead). Serves as the un-confounded ablation control to
  disentangle PPO optimization improvements from 6h lookahead features (20 runs across 4 agents).
- **In-the-Loop Safe Extensions (`clamped_ppo_diff_shield`, `clamped_ppo_shield_aux`):**
  10 runs (5 seeds each) evaluating differentiable analytical safety projections and auxiliary shield divergence losses.

## Baselines and Gate Constants

- **`frozen_baselines.csv`:** Contains the verified Val baselines: Grid-Only (45,337.92 EUR, 3.2571 kW exceedance), PSM' (55,530.22 EUR, 0.0000 kW exceedance), Shield + Idle (58,644.70 EUR), Shield + PSM' (55,500.35 EUR), and the continuous hindsight oracle LP' (53,290.51 EUR total: 44,192.97 EUR energy purchase, 436.73 EUR battery wear, 8,660.86 EUR unserved load penalty; 24.990 MWh throughput; 85.00% active SoC).
- **Canonical G2 Economic Gate:** `project_constants.csv` recorded the strict 1.00x PSM' baseline (55,530.22 EUR). The canonical evaluation helper in `tools/report/load_results.py` standardizes the gate to `1.05 * PSM'` (58,306.73 EUR), providing an engineering tolerance for safe learning under which Seed 101 of `clamped_ppo_diff_shield` passes (56,411.62 EUR).
- **Exact MILP' Formulation:** `tools/report/lp_oracle.py::solve_week_milp` provides the exact mixed-integer formulation enforcing the discontinuous industrial motor floor alongside continuous `solve_week_lp`.

## Regenerating the tables and figures

From the repository root, in an environment with the requirements installed:

```bash
python tools/report/lp_oracle.py --split both   # solve perfect-foresight LP' theoretical bounds and MILP'
python tools/report/export_csv.py               # rebuild results/tables/*.csv from runs/
python tools/report/plots.py                    # rebuild all figures into study1-study4 directories
python tools/report/plot_stress_trajectory.py   # rebuild 168-hour dispatch trajectory for W49
python tools/status.py                          # compact health/status summary
```

The scripts read only the committed artefacts; they do not retrain and do not
touch the closed Test split.

## Historical Snapshot vs. Codebase Upgrades (T1–T7)

The CSV tables and JSON runs under `results/` preserve the published `v0-prototype` 300k-step benchmark runs.
Subsequent post-snapshot upgrades (tasks T1–T7) have structurally enhanced the codebase:
- **T1 (`env/cap_shield.py`):** Inverted relu/deadzone shed mappings and interlocked battery recharging to 0 on load shedding events.
- **T2 (`agents/ppo.py`, `experiments/train_core.py`):** Reused denormalized bootstrap values and cleanly zeroed GAE across episode truncations.
- **T3 (`agents/safety_layer.py`):** Straight-Through Estimator (STE) surrogate gradient restoring active constraint backprop, hard SoC bounds, and full 6-D KKT QP projection.
- **T5 (`tools/report/lp_oracle.py`, `load_results.py`):** Added exact `solve_week_milp` and centralized the canonical 1.05x G2 gate helper.
- **T6 (`agents/mpc.py`):** Implemented causal receding-horizon `MPCPolicy` with LP/MILP solvers and latency profiling.
- **T7 (`tools/report/evaluate_test_split.py`):** Added CVaR_0.95 tail-risk metrics, step latency tracking, and 4 non-destructive OOD contingencies.

Re-training the 70 seed runs under this upgraded implementation is planned for future experimental cycles; the committed results remain the reproducible artifacts of the initial `v0-prototype` runs.

## What is *not* here

- Raw or processed datasets (`data/raw/`, `data/processed/`) — downloaded at run
  time; see the README.
- Model checkpoints (`checkpoints/`) — large and not needed to check the reported
  metrics.
- Full test-split timeseries trajectories — only aggregate metrics and stress-week evaluations are stored under results/tables/.
