# `results/` — what is tracked and why

This directory is committed on purpose. It is small (~12 MB) and contains the
evidence behind the README tables and figures, so a reader does not have to rerun
100 training jobs to check the claim.

## Layout

```
figures/
  presentation/                     Consolidated 12-figure PNG benchmark suite (250 DPI):
    01_ablation_pathologies.png         Cost decomposition, cap violations, and battery activity ablation
    02_pareto_cost_safety.png           Broken-X Pareto: 0.00 kW safe lane | log exceedance | P50 latency
    03_learning_dynamics.png            50k–300k step Val learning curves (cost, violations, active SoC)
    04_regime1_causal_mpc.png           Regime 1 Cleveland dot+SD benchmark vs. LP'/MILP' and 24h MPC
    05_regime2_ci_ablation.png          Regime 2 C&I demand-charge savings & unshielded 6-D motor leakage
    06_ood_stress_2019w04.png           2019-W04 OOD contingency matrix (violation rate & CVaR95 cost)
    07_dispatch_case_2019w04.png        3-row full-168h dispatch case study on 2019-W04 (cycling vs. parking)
    08_seasonal_test_context.png        Val (H1) vs. locked Test v0 (H2) headroom & W49 stress dumbbell
    09_microgrid_architecture.png       CMDP architecture, 5-zone shares, BESS/PV & safety gates
    10_dataset_split_regimes.png        154-week chronological split timeline
    11_safety_mechanisms.png            Gradient flow comparison: post-hoc shield vs. v0 clip vs. T3 STE + CapShield
    12_seed_stability_tradeoff.png      5-seed spread (Regimes 1 & 2) & 24h MPC forecast sensitivity
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

## Benchmark Figures Guide

The 12 PNG figures in [`figures/presentation/`](figures/presentation) summarize the empirical study results directly from the committed CSVs/JSONs:

| Figure (`PNG`) | Panels & Visual Encoding | Data Sources | Core Takeaway |
|---|---|---|---|
| [`09_microgrid_architecture.png`](figures/presentation/09_microgrid_architecture.png) | 3-column CMDP block diagram: `17-D/23-D` state, PCC/PV/BESS & 5 zones, `6-D` projection & `G1`/`G2` | `env/smartgrid_env.py`, `env/cap_shield.py`, `configs/config.yaml`, `split_report.json` | Maps physical microgrid couplings (`368.15 kW` P90 cap, `250 kWp` PV, `500 kWh` BESS, discontinuous industrial motor floor) to closed-loop safety gates. |
| [`10_dataset_split_regimes.png`](figures/presentation/10_dataset_split_regimes.png) | 154-week chronological split timeline across `Train`, `Val`, and `Test` | `data/processed/*.parquet`, `split_report.json`, `configs/config.yaml` | Shows strict chronological partitioning (`104w` Train / `26w` Val / `24w` Test) and winter vs. autumn exceedance regimes. |
| [`11_safety_mechanisms.png`](figures/presentation/11_safety_mechanisms.png) | 3-track forward/backward gradient schematic: Post-Hoc Shield vs. `v0` 1-D clip vs. T3 STE + `CapShield` | `agents/safety_layer.py`, `agents/ppo.py`, `per_seed_metrics.csv`, `regime_benchmarks_val.csv` | Explains why external shields park storage (`4.0%` active SoC) and `v0` clipping vanishes active Jacobians, while T3 STE (`α=0.1`, `forward_raw`) restores convergence. |
| [`01_ablation_pathologies.png`](figures/presentation/01_ablation_pathologies.png) | 3-panel horizontal breakdown: (A) stacked cost decomposition vs. `PSM'`/`G2`, (B1) cap violation rate, (B2) active SoC | `per_seed_metrics.csv`, `frozen_baselines.csv`, `project_constants.csv` | Shows `PPO-Diff-Shield` eliminating cap leakage (`PPO`), Lagrangian churn (`€64.4k`, `5.45%` viol), and naive `Shielded PPO` battery parking (`4.0%` active SoC). |
| [`02_pareto_cost_safety.png`](figures/presentation/02_pareto_cost_safety.png) | Broken-X 3-column Pareto: `0.00 kW` safe lane \| log-scale leaky/unsafe exceedance \| P50 step latency | `per_seed_metrics.csv`, `frozen_baselines.csv`, `regime_benchmarks_val.csv` | Only `PPO-Diff-Shield` combines exact `0.00 kW` exceedance, G2-passing cost (`€56.3k` final / `€55.7k` best), and sub-ms speed (`0.78 ms` P50). |
| [`03_learning_dynamics.png`](figures/presentation/03_learning_dynamics.png) | 3-panel training curves (`50k–300k` steps, mean ± 1 SD, 5 seeds): Val cost, cap violation rate, active SoC | `runs/*/*_seed*.json` (`eval_checkpoints`) | `PPO-Diff-Shield` converges steadily below `G2` (`0.00%` viol, `31.2%` active SoC), whereas naive `Shielded PPO` parks storage and `PPO-Lagrangian` diverges. |
| [`04_regime1_causal_mpc.png`](figures/presentation/04_regime1_causal_mpc.png) | Cleveland dot + seed scatter (`total cost`) alongside `CVaR₉₅` and P50 latency panels | `regime_benchmarks_val.csv` (`regime1_val`) | Beats `MPCPolicy(persistence, H=24)` (`€58.13k`) and matches causal MPC efficiency while running `~4.5×` faster (`0.78 ms` vs. `3.52 ms` P50). |
| [`05_regime2_ci_ablation.png`](figures/presentation/05_regime2_ci_ablation.png) | 2-panel C&I stacked cost breakdown + non-energy (`demand charge` vs. `unserved + wear`) & leakage callouts | `regime_benchmarks_val.csv` (`regime2_val`) | `ci_ppo_diff_shield` (`€69.74k`) cuts demand charges by `€1,397` vs. `PSM'` and beats `ci_ppo_shielded`; unshielded `ci_ppo_diff_6d` leaks `8.80%` (`384.4h`). |
| [`12_seed_stability_tradeoff.png`](figures/presentation/12_seed_stability_tradeoff.png) | 5-seed spread (Seeds 101–105 across Regimes 1–2) & 24h MPC forecast degradation (`perfect` → `persistence`) | `regime_benchmarks_val.csv`, `per_seed_metrics.csv` | Confirms consistent 5-seed zero-violation convergence (`€55.67k ± €1.22k` R1 best; `€69.74k ± €0.64k` R2) while naive persistence adds `+€4.62k` to MPC. |
| [`06_ood_stress_2019w04.png`](figures/presentation/06_ood_stress_2019w04.png) | 2-panel 2019-W04 contingency matrix (`5` faults): (A) violation rate & max exceedance, (B) weekly cost & `CVaR₉₅` | `regime_benchmarks_val.csv` (`ood_val_2019-W04`) | `clamped_ppo_diff_shield` holds `0.00%` violations across all faults and beats both `MPCPolicy(causal_noisy)` and `PSM'` under `forecast_noise`. |
| [`07_dispatch_case_2019w04.png`](figures/presentation/07_dispatch_case_2019w04.png) | 3-row synchronized 168h dispatch on `2019-W04`: net grid import vs. cap, battery SoC, cumulative unserved load | Deterministic `168h` Val rollout (`2019-W04`) | Exposes `Shielded PPO` parking near the SoC ceiling while `PPO-Diff-Shield` cycles `[10%, 90%]` to cut load curtailment. |
| [`08_seasonal_test_context.png`](figures/presentation/08_seasonal_test_context.png) | Paired Val (H1) vs. locked Test `v0` (H2) bars + stress-week dumbbell (`W30/W45/W48/W51` mean → `W49`) | `per_seed_metrics.csv`, `frozen_baselines.csv`, `test_evaluation.csv`, `test_stress_weeks.csv` | Explains H1→H2 arbitrage compression (`3.3×` narrower `LP'→PSM'` headroom) while `2019-W49` drives safe curtailment (`0h` viol vs. `66h` Grid-Only). |

## Regenerating the tables and figures

From the repository root, in an environment with the requirements installed:

```bash
python tools/report/lp_oracle.py --split both       # solve perfect-foresight LP' theoretical bounds and MILP'
python tools/report/export_csv.py                   # rebuild results/tables/{runs,per_seed_metrics,s2_per_seed}.csv
python tools/report/eval_regimes_val.py             # rebuild results/tables/regime_benchmarks_val.csv
python tools/report/plot_stress_trajectory.py       # rebuild 168-hour dispatch trajectory for W49
python tools/status.py                              # compact health/status summary
```

The scripts read only the committed artefacts and Val data; they do not retrain and do not
touch the locked Test split.

## Post-Fix Validation Runs (T1–T13) vs. Locked Test Snapshot (`v0-prototype`)

- **Updated Validation Artifacts (T12–T13):** `runs.csv`, `per_seed_metrics.csv`, `s2_per_seed.csv`, `regime_benchmarks_val.csv`, and `results/figures/presentation/` (12 PNG benchmark figures) reflect the 80 post-fix 300k-step runs (`clamped_*`, `clamped_mb_*`, `clamped_ppo_diff_shield`, `clamped_ppo_shield_aux`, `causal23_*`, `ci_*`) trained with T1 (`cap_shield.py`), T2 (bootstrap/truncation GAE fixes), T3 (STE surrogate gradient + SoC bounds + `forward_raw`), and T9–T11 (Regime 1 causal forecasts + Regime 2 C&I tariffs), alongside the 20 legacy `s3_*` runs.
- **Preserved Locked Test Snapshot (`v0-prototype`):** `results/tables/test_evaluation.csv`, `results/tables/test_stress_weeks.csv`, and `results/runs/test_lock.json` remain the **locked one-shot `v0-prototype` Test snapshot** and were deliberately NOT unlocked or re-evaluated.

## What is *not* here

- Raw or processed datasets (`data/raw/`, `data/processed/`) — downloaded at run
  time; see the README.
- Model checkpoints (`checkpoints/`) — large and not needed to check the reported
  metrics.
- Full test-split timeseries trajectories — only aggregate metrics and stress-week evaluations are stored under results/tables/.
