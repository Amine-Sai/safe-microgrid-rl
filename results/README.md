# `results/` — what is tracked and why

This directory is committed on purpose. It is small (~12 MB) and contains the
evidence behind the README tables and figures, so a reader does not have to rerun
100 training jobs to check the claim.

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
  test_lock.json             lock token for one-shot evaluation on the Test split (v0-prototype snapshot)
  <agent>/                   one JSON per (experiment, seed)
    *_seed101.json ...       5 seeds per method (101–105)
    *_grid.json              batch roll-up, not a run
tables/
  runs.csv                   one row per seed run (100 runs total: cost, exceedance, wall clock)
  per_seed_metrics.csv       per-seed diagnostics, gate pass/fail flags, strict_feasible (100 rows)
  s2_per_seed.csv            legacy alias of per_seed_metrics.csv (100 rows)
  regime_benchmarks_val.csv  77-row Val benchmark across Regime 1 (baselines, LP'/MILP', MPC, final/best diff-shield), Regime 2 (C&I tariffs), and 2019-W04 OOD suite (CVaR95, latency P50/P99)
  frozen_baselines.csv       Grid-Only, PSM', Shield baselines, and exact LP' hindsight oracle values
  project_constants.csv      cap, d_main, weights, LP' bounds, gate constants
  test_evaluation.csv        locked one-shot evaluation on the 2019 H2 Test split (v0-prototype snapshot)
  test_stress_weeks.csv      evaluation on the 5 Test stress weeks (v0-prototype snapshot, including cap-binding W49)
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

Six experiment configurations totaling 100 runs (80 post-fix T12 runs + 20 legacy `s3_*` runs):
- **Unclamped (`unclamped_*` / legacy `s3_*`, `v0-prototype`):** `charge_clamp: false`. The original environment
  formulation where battery charging was not physically interlocked against the
  import cap (20 runs across 4 agents).
- **Clamped (`clamped_*` / `s3v2_*`, post-fix T12):** `charge_clamp: true`. Applies the physical
  clamp rate limit → SoC limit → cap clamp in the environment. Battery charging
  cannot induce grid import exceedance beyond available headroom (20 runs across 4 agents).
- **Clamped Minibatch (`clamped_mb_*`, post-fix T12):** `charge_clamp: true` with minibatch updates, 5 value epochs,
  and raw action storage, on the base 17-D state space (no lookahead) (20 runs across 4 agents).
- **In-the-Loop Safe Extensions (`clamped_ppo_diff_shield`, `clamped_ppo_shield_aux`, post-fix T12):**
  10 runs (5 seeds each) evaluating T3 STE-enabled differentiable safety projections and auxiliary shield divergence losses.
- **Regime 1 Causal 23-D Forecasts (`causal23_ppo_shielded`, `causal23_ppo_diff_shield`, `causal23_ppo_residual`, T12):**
  15 runs (5 seeds each) using `state_dim: 23` with 6h `causal_noisy` forecasts (exact D-1 spot prices, $\sqrt{h/H}$-scaled PV/wind/load noise).
- **Regime 2 C&I Demand-Charge & Zone-VoLL (`ci_ppo_shielded`, `ci_ppo_diff_shield`, `ci_ppo_diff_6d`, T12):**
  15 runs (5 seeds each) with `2.333333 EUR/kW-week` weekly peak demand charge, zone VoLL (`industrial: 3.0`, `residential: 1.5`, `commercial: 1.2`, `stadium: 0.4`, `street: 0.2`), and `50.0 EUR` industrial trip penalty.

## `regime_benchmarks_val.csv` Schema & Suites

Generated by `tools/report/eval_regimes_val.py` (77 rows across 27 columns, including `cvar95_cost_eur`, `max_hourly_exceedance_kw`, `max_consecutive_violation_hours`, `shield_intervention_rate`, `latency_p50_ms`, and `latency_p99_ms`):
- **`regime1_val` (35 rows):** 26-week Val comparison of `Grid-Only`, `PSM'`, `LP'`, `MILP'`, `MPCPolicy` (`perfect`, `causal_noisy`, `persistence`, $H=24$), and 5-seed `final` + `best` evaluations (with `mean`/`std` rows) of `clamped_ppo_diff_shield` and `causal23_ppo_diff_shield`.
- **`regime2_val` (27 rows):** 26-week Val C&I tariff comparison of `Grid-Only`, `PSM'`, `LP'`, `MILP'`, `MPCPolicy` (`perfect`, `causal_noisy`, $H=24$), and 5-seed `final` evaluations of `ci_ppo_shielded`, `ci_ppo_diff_shield`, and `ci_ppo_diff_6d`.
- **`ood_val_2019-W04` (15 rows — lines 64–78):** 1-week OOD contingency suite (`nominal`, `pv_inverter_trip`, `bess_derate`, `grid_cap_reduction`, `forecast_noise`) comparing `PSM'`, `MPCPolicy(causal_noisy, H=24)`, and `clamped_ppo_diff_shield` (Seed 101, `final`).

## Baselines and Gate Constants

- **`frozen_baselines.csv`:** Contains the verified Val baselines: Grid-Only (45,337.92 EUR, 3.2571 kW exceedance), PSM' (55,530.22 EUR, 0.0000 kW exceedance), Shield + Idle (58,644.70 EUR), Shield + PSM' (55,500.35 EUR), and the continuous hindsight oracle LP' (53,290.51 EUR total: 44,192.97 EUR energy purchase, 436.73 EUR battery wear, 8,660.86 EUR unserved load penalty; 24.990 MWh throughput; 85.00% active SoC).
- **Canonical G2 Economic Gate:** `project_constants.csv` recorded the strict 1.00x PSM' baseline (55,530.22 EUR). The canonical evaluation helper in `tools/report/load_results.py` standardizes the gate to `1.05 * PSM'` (58,306.73 EUR). Post-fix `clamped_ppo_diff_shield` now passes `1.05 * PSM'` on mean (`final`: 56,273.79 EUR; `best`: 55,665.01 EUR), 4/5 `final` seeds, and 5/5 `best` seeds, with Seed 101 (54,766.30 EUR) also beating `1.00 * PSM'`.
- **Exact MILP' Formulation:** `tools/report/lp_oracle.py::solve_week_milp` provides the exact mixed-integer formulation enforcing the discontinuous industrial motor floor alongside continuous `solve_week_lp`.

## Regenerating the tables and figures

From the repository root, in an environment with the requirements installed:

```bash
python tools/report/lp_oracle.py --split both   # solve perfect-foresight LP' theoretical bounds and MILP'
python tools/report/export_csv.py               # rebuild results/tables/{runs,per_seed_metrics,s2_per_seed}.csv
python tools/report/eval_regimes_val.py         # rebuild results/tables/regime_benchmarks_val.csv
python tools/report/plots.py                    # rebuild all figures into study1-study4 directories
python tools/report/plot_stress_trajectory.py   # rebuild 168-hour dispatch trajectory for W49
python tools/status.py                          # compact health/status summary
```

The scripts read only the committed artefacts and Val data; they do not retrain and do not
touch the locked Test split.

## Post-Fix Validation Runs (T1–T13) vs. Locked Test Snapshot (`v0-prototype`)

- **Updated Validation Artifacts (T12–T13):** `runs.csv`, `per_seed_metrics.csv`, `s2_per_seed.csv`, `regime_benchmarks_val.csv`, and `results/figures/study1..4/` reflect the 80 post-fix 300k-step runs (`clamped_*`, `clamped_mb_*`, `clamped_ppo_diff_shield`, `clamped_ppo_shield_aux`, `causal23_*`, `ci_*`) trained with T1 (`cap_shield.py`), T2 (bootstrap/truncation GAE fixes), T3 (STE surrogate gradient + SoC bounds + `forward_raw`), and T9–T11 (Regime 1 causal forecasts + Regime 2 C&I tariffs), alongside the 20 legacy `s3_*` runs.
- **Preserved Locked Test Snapshot (`v0-prototype`):** `results/tables/test_evaluation.csv`, `results/tables/test_stress_weeks.csv`, and `results/runs/test_lock.json` remain the **locked one-shot `v0-prototype` Test snapshot** and were deliberately NOT unlocked or re-evaluated.

## What is *not* here

- Raw or processed datasets (`data/raw/`, `data/processed/`) — downloaded at run
  time; see the README.
- Model checkpoints (`checkpoints/`) — large and not needed to check the reported
  metrics.
- Full test-split timeseries trajectories — only aggregate metrics and stress-week evaluations are stored under results/tables/.
