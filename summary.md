# Safe Reinforcement Learning for Microgrid Energy Management: Executive Technical Summary

---

## 1. Executive Summary & Problem Overview

Real-time microgrid energy management balances three conflicting goals: minimizing grid energy purchase costs, limiting battery wear, and avoiding expensive load curtailment penalties—all while strictly respecting a contractual grid-import capacity limit (368.15 kW).

Deploying standard reinforcement learning (RL) in safety-critical power systems exposes three fundamental pathologies:
1. **Safety Violations:** Unconstrained and penalty-based policies treat physical capacity limits as economic trade-offs, routinely violating grid import caps during demand peaks.
2. **Moral Hazards & Exploration Breakdown:** Post-hoc environmental safety filters decouple the neural policy from constraint enforcement, triggering a **battery parking** pathology where the policy idles storage assets and shifts the entire burden onto costly load shedding.
3. **Temporal Credit Assignment Failures:** Averaging policy gradients across long operational episodes dilutes localized peak feedback, causing learning stagnation.

This benchmark shows that **in-the-loop differentiable analytical safety projections (`PPO-Diff-Shield`) resolve the gradient disconnect**, and fixing `SafePolicyNetwork.distribution()` to propagate Straight-Through Estimator (STE) gradients during policy updates plus PPO truncation/bootstrap handling (`T1`–`T3`) improved `clamped_ppo_diff_shield` Validation cost from **58,473.79 ± 1,180.66 EUR** (`v0-prototype`) to **56,696.94 ± 1,279.45 EUR (`final`) / 55,402.89 ± 1,212.20 EUR (`best`)** with **0.00% grid violations** (and causal 23-D `causal23_ppo_diff_shield` to **56,475.07 ± 1,311.65 EUR `final` / 55,268.23 ± 685.32 EUR `best`**)—passing the canonical $1.05 \times \text{PSM'}$ G2 gate (58,306.73 EUR) on mean, 4/5 `final` seeds, and 5/5 `best` seeds, while both `best` means and `final` Seed 101 (**55,045.60 EUR**) beat strict $1.00 \times \text{PSM'}$ (55,530.22 EUR). In addition, under **Regime 2** (C&I weekly peak demand charges, heterogeneous zone VoLL, and industrial trip penalties), `ci_ppo_diff_shield` (**69,784.24 ± 617.02 EUR**, 0.00% violations) outperforms `PSM'` (70,370.73 EUR), `Shield + PSM'` (70,363.16 EUR), and `ci_ppo_shielded` (70,581.07 ± 605.99 EUR), though the zero-battery `Shield + Idle` baseline (69,115.11 EUR) beats all `ci_*` RL 5-seed means (only Seed 105 at 68,906.11 EUR beats `Shield + Idle`) and `MPCPolicy(causal_noisy, H=24)` achieves 65,250.30 EUR. Under the **2019-W04 OOD contingency suite** (5 seeds, consistent noisy `CapShield` observations via `set_last_obs`, clean $t=0$ in `MPCPolicy`), `clamped_ppo_diff_shield` maintains **0.00% violations** across `nominal` (**7,980.30 ± 177.99 EUR**), `pv_inverter_trip`, `bess_derate`, and `grid_cap_reduction`, while `Shield + MPCPolicy(causal_noisy, H=24)` matches/beats `clamped_ppo_diff_shield` across all five contingencies; under `forecast_noise`, all shielded controllers incur ~33–36 violation hours (19.6%–21.4%, 0.3354 kW mean exc) because `CapShield` itself relies on noisy current PV/load readings.

---

## 2. Key Divergences from Prior Work (Salaorni et al., arXiv:2506.22995)

This benchmark builds upon foundational work in RL for microgrid dispatch—specifically the control setup of Salaorni et al. (IJCNN 2025, arXiv:2506.22995) based on the ErNESTO battery digital twin. However, that setup was designed for unconstrained economic trading and differs fundamentally from safety-critical distribution grids:

| Dimension | Prior Work (Salaorni et al., arXiv:2506.22995) | This Benchmark (Safe Microgrid RL) |
|---|---|---|
| **Grid Capacity Limits** | Unconstrained energy trading; infinite slack bus with bidirectional exchange. | Hard contractual import cap (368.15 kW); zero reverse feeding; strict zero-violation safety gate. |
| **Control Action Space** | 1-dimensional continuous action (battery charge/discharge ratio vs. grid). | 6-dimensional continuous action (battery power plus 5-sector fractional load shedding). |
| **Load Curtailment & Physics** | No demand shedding; passive load consumption. | Multi-zone priority shedding with a discontinuous industrial motor floor rule (70%–100% or full shutdown). |
| **Economic Penalty Structure** | Soft degradation penalties linearly combined into scalar reward. | Realistic 50:1 cost trade-off (1.00 EUR/kWh curtailment vs. 0.02 EUR/kWh battery wear). |
| **Safety Enforcement** | Standard unconstrained RL (SAC, TD3, PPO) optimizing economic reward. | Formal Constrained MDP benchmarking: Dual-Lagrangian, Residual, Post-Hoc Shield, and Differentiable Safety Layer. |
| **Pathology Identification** | None identified (unconstrained setting). | Identifies and resolves "battery parking" moral hazard, Lagrangian multiplier oscillation, and rollout credit assignment failure. |
| **Comparators & Oracles** | Untuned rule-based heuristics without peak-shaving coordination. | Tuned domain peak-shaving heuristic (PSM'), exact continuous LP oracle (LP'), and exact mixed-integer hindsight oracle (MILP' via HiGHS). |
| **Data & Stress Profiles** | Regional Italian power grid data without acute capacity bottlenecks. | Multi-source European data (OPSD, PVGIS, Open-Meteo), chronological splits, and acute stress bottlenecks (Week 49). |

### Practical Implications of These Divergences
In unconstrained settings, the agent treats the grid as an infinite buffer, and soft reward penalties suffice to balance costs. However, in physical distribution networks where exceeding capacity triggers substation transformer tripping or severe contractual penalties, soft penalties fail: RL agents exploit the 50:1 cost asymmetry to violate the grid cap during peak hours. Resolving these challenges requires coordinated multi-zone demand response, rigorous safety architectures, and analytical differentiable projections.

---

## 3. Microgrid Physical Architecture & Real-World Data

### 3.1 Microgrid Physical Subsystems

The simulated facility models an urban commercial/industrial microgrid operating on discrete hourly intervals:
- **Grid Interconnection:** Point of Common Coupling with a contractual import cap of **368.15 kW** (the empirical 90th percentile of net demand). Energy export back to the grid is barred.
- **Battery Energy Storage System (BESS):** 500 kWh nominal capacity, 250 kW continuous power rating (0.5C), 95% one-way charge/discharge efficiencies ($\eta_{\text{ch}} = \eta_{\text{dis}} = 0.95$, i.e., **90.25% round-trip efficiency**), and an operational state-of-charge safety envelope constrained to 10%–90% to prevent degradation.
- **Solar Photovoltaic (PV) Generation:** 250 kWp rooftop solar array modeled with empirical solar irradiance.
- **Five Heterogeneous Demand Sectors:** Residential (35%), Commercial (25%), Industrial (25%), Stadium (10%), and Street Lighting (5%).
- **Discontinuous Industrial Motor Rule:** Inductive motor drives cannot operate safely below 70% of continuous rating due to cooling and torque collapse. If curtailment reaches the industrial zone, it must operate between 70% and 100%, or undergo an immediate complete shutdown (0%); intermediate throttling (1%–69%) is physically disallowed.

### 3.2 The Economic Objective & The 50:1 Cost Asymmetry

Hourly microgrid operational costs combine three elements:
1. **Grid Electricity Purchases:** Wholesale day-ahead spot pricing (averaging 0.05–0.15 EUR/kWh).
2. **Battery Wear Degradation:** Fixed cell replacement wear cost of **0.02 EUR/kWh** (20 EUR/MWh).
3. **Customer Load Curtailment:** Regulatory Value of Lost Load penalty of **1.00 EUR/kWh** (1,000 EUR/MWh).

Because shedding 1 kWh costs fifty times more than cycling 1 kWh through the battery, optimal dispatch requires exhausting all available battery discharge before resorting to customer curtailment.

### 3.3 Observation Space, Action Space, and Evaluation Gates

- **State Space (17 dimensions):** Battery state-of-charge, solar generation, five sector demands, electricity spot price, ambient temperature, relative humidity, barometric pressure, cyclical hour of day and day of week ($\sin/\cos$), and stadium event + 1-hour event lookahead indicators ($\mathbb{I}_{\text{event}, t}, \mathbb{I}_{\text{event\_next\_1h}, t}$).
- **Action Space ($\mathcal{A} = [-1, 1]^6$):** Normalized 6-D continuous vector containing the battery charge/discharge command (-1 for 250 kW discharge, +1 for 250 kW charge) and five continuous load curtailment commands in $[-1, 1]$ mapped linearly via $s_z = (a_z + 1)/2 \in [0, 1]$ for each demand sector.
- **Safety Gate (G1):** Mean contract cap exceedance must not exceed 0.05 kW per step (0.00 kW for strict physical feasibility).
- **Economic Gate (G2):** Total operational cost must not exceed 105% of the domain heuristic PSM' (58,306.73 EUR; computed in `tools/report/load_results.py` as $1.05 \times 55{,}530.22\,\text{EUR}$ and stored in `project_constants.csv` as `g2_cost_multiplier = 1.05`). `per_seed_metrics.csv` tracks both `G2_pass` ($\le 58{,}306.73\,\text{EUR}$) and `G2_strict_pass` ($\le 55{,}530.22\,\text{EUR}$, $1.00\times \text{PSM'}$, beaten on mean by both `best` diff-shield configs as well as Seed 101 of `final` `clamped_ppo_diff_shield` at 55,045.60 EUR).
- **Regime 2 Extension (C&I Tariffs & Causal Forecasts):** `ci_*` configs enable 6h `causal_noisy` forecasts (`state_dim: 23`, exact D-1 prices with $\sqrt{h/H}$-scaled non-negative Gaussian PV/wind/load noise), a telescoping weekly peak demand charge (`2.333333 EUR/kW-week`, prorated from `~10 EUR/kW-month`), heterogeneous zone VoLL (`industrial: 3.0`, `residential: 1.5`, `commercial: 1.2`, `stadium: 0.4`, `street: 0.2 EUR/kWh`), and a `50.0 EUR` industrial trip switching penalty.

### 3.4 Real-World Datasets & Chronological Splitting

- **Data Sources (`proxy_DE` composite benchmark):** French national load from Open Power System Data / ENTSO-E (scaled to 500 kW microgrid peak), Sceaux PVGIS solar irradiance, Open-Meteo weather (temperature, humidity, pressure), and an `AT` $\to$ `DE-LU` day-ahead spot price splice at `2018-09-30 22:00 UTC` across the 2017–2019 bidding-zone split.
- **Chronological Partitions:** Shuffling time steps destroys physical battery charge continuity. Data is partitioned strictly chronologically:
  - **Training Split:** 2017–2018 (104 weeks / 17,472 operating hours).
  - **Validation Split:** First half of 2019 (Weeks 02–27 / 4,368 operating hours; winter-heavy, 468 violation hours unmanaged, net demand peaking at 456.87 kW).
  - **Test Split:** Second half of 2019 (Weeks 29–52 / 4,032 operating hours; summer/autumn).
- **Acute Bottleneck (Week 49):** The true stress bottleneck is Week 49 (December 2–8, 2019), where net demand surges to 426.32 kW, creating a 58.17 kW overrun across 66 hours.

---

## 4. How Each Agent Operates: Complete Directory

### 4.1 Heuristics and Mathematical Programming Baselines

1. **`GridOnlyPolicy` (Grid-Only Unmanaged Baseline):**
   Completely passive baseline. Battery dispatch and load shedding are locked to zero. All demand is served directly by available solar generation and transmission imports.
2. **`PeakShaveMinPolicy` (Peak-Shave-Minimum / PSM' Domain Heuristic):**
   Rule-based priority heuristic engineered around the 50:1 cost trade-off. Charges the battery during off-peak hours up to available grid cap headroom; discharges up to 250 kW when net demand exceeds 368.15 kW; and sheds load across zones in increasing order of economic criticality if storage is exhausted.
3. **`ShieldedIdlePolicy` (Shield + Idle Zero-Battery Baseline):**
   Locks battery dispatch to zero, but routes facility demand through the analytical safety shield to enforce grid capacity via load shedding.
4. **`ShieldedPSMPolicy` (Shield + PSM' Combined Heuristic):**
   Combines the tuned PSM' heuristic with the runtime safety shield to verify heuristic-shield compatibility.
5. **`LP'` / `MILP'` Theoretical Oracles (Continuous LP & Exact Mixed-Integer HiGHS Solvers):**
   Global linear program (`LP'`, continuous relaxation of motor floor) and exact mixed-integer linear program (`MILP'`, enforcing binary motor shutdown constraints) solved with perfect hindsight over each 168-hour weekly horizon (`tools/report/lp_oracle.py`), yielding 53,290.51 EUR on Regime 1 Val and 64,513.29 EUR on Regime 2 Val.
6. **`MPCPolicy` (Causal Receding-Horizon LP/MILP Controller):**
   24-hour receding-horizon optimizer (`agents/mpc.py`) supporting `perfect`, `causal_noisy`, and `persistence` forecasts with full Regime 2 parity (weekly peak-demand epigraph variable, zone VoLL, trip switching penalty). Achieves 53,404.83 EUR (`perfect`) and 53,514.28 EUR (`causal_noisy`) on Regime 1 Val at ~3.5 ms/step P50, degrading to 58,134.64 EUR under `persistence`.

### 4.2 Reinforcement Learning Agents

7. **`PPOAgent` [Unclamped] (Unconstrained Proximal Policy Optimization):**
   Standard Gaussian PPO policy trained with Generalized Advantage Estimation without physical interlocks or safety filters.
8. **`PPOAgent` [Clamped Full-Batch] (Clamped PPO with Full-Batch Updates):**
   Restricts battery charging power to available headroom below the grid cap. Trained using standard 168-hour full trajectory rollouts.
9. **`PPOAgent` [Clamped Minibatch] (Vanilla Minibatch PPO — Best Unconstrained RL):**
   Combines the physical charging interlock with minibatch stochastic policy updates (batch size 64, 5 epochs per rollout), enabling localized credit assignment.
10. **`PPOLagrangianAgent` (Dual-Lagrangian Constrained PPO):**
    Updates an adaptive penalty multiplier via subgradient ascent whenever capacity violations occur, adding a dynamic penalty to the reward signal.
11. **`PPOResidualAgent` (Residual Policy Learning on PSM' Heuristic):**
    A neural network outputs additive action adjustments superimposed onto the baseline PSM' heuristic decisions.
12. **`PPOShieldedAgent` (Post-Hoc Environmental Shielded PPO — Full-Batch & Minibatch):**
    The neural policy outputs raw dispatch commands, which an external runtime safety shield intercepts and modifies before environment execution.
13. **`PPOAuxShieldLossAgent` (`PPO-Aux-Shield`, Auxiliary Shield Loss PPO):**
    Adds an auxiliary supervised loss to the policy objective, penalizing the Euclidean distance between proposed policy actions and safe shielded actions.
14. **`PPODiffSafetyLayerAgent` (`PPO-Diff-Shield`, Differentiable Analytical Safety Layer PPO):**
    Embeds an analytical closed-form Euclidean projection with the T3 Straight-Through Estimator (STE) surrogate gradient directly into the policy neural network's forward pass, allowing non-vanishing analytical Jacobian gradients to flow through constraint boundaries during backpropagation. Also supports a 6-D continuous KKT QP layer (`ci_ppo_diff_6d`).

---

## 5. How the Safety Shield Works

The analytical runtime shield guarantees zero constraint violations through five deterministic execution steps:

1. **Battery Charging Headroom Clamping:** Clamps positive battery charging power to the remaining headroom below the grid cap, preventing charging commands from creating new violations.
2. **Net Grid Exceedance Evaluation:** Evaluates remaining net import against the 368.15 kW cap after solar generation and battery dispatch. If net import is within limits, the action passes unmodified.
3. **Priority-Ordered Customer Shedding:** If excess import remains, the shield curtails load in increasing order of social criticality:
   Street Lighting -> Stadium -> Commercial -> Residential -> Industrial.
4. **Discontinuous Industrial Motor Rule:** When curtailment reaches the industrial zone:
   - If required reduction is 30% or less, industrial load is reduced proportionally (operating between 70% and 100%).
   - If required reduction exceeds 30%, intermediate throttling is rejected and the shield commands a complete industrial shutdown (0%).
5. **Numerical Precision Protection:** Applies a small floating-point guard offset when mapping physical power reductions back to normalized action variables, preventing sub-milliwatt numerical rounding leaks.

---

## 6. Where and Why Each Agent Fails: Root Causes & Pathologies

| Controller Architecture | Primary Failure Mode | Underlying Root Cause |
|---|---|---|
| **Grid-Only** | Catastrophic Cap Violations | Passive baseline; zero storage dispatch or curtailment. |
| **PPO [Unclamped]** | Charging-Induced Violations | Charges aggressively during low-tariff hours, pushing facility load over the cap. |
| **PPO [Clamped Full-Batch]** | Idling / Stagnation | 168-hour rollout averaging dilutes localized peak signals, trapping policy in an idling saddle point. |
| **PPO [Clamped Minibatch]** | Intentional Constraint Leakage | Exploits the 50:1 cost trade-off: violating the cap is cheaper than paying 1.00 EUR/kWh curtailment. |
| **PPO-Lagrangian** | Under-Penalized Dual Drift | With 1×/rollout dual updates (`T2`, $\alpha_\lambda = 0.01$), $\lambda$ converges calmly (`0.314` MB / `0.369` FB) rather than oscillating wildly, yet remains unsafe (409.8h / 468.0h viol). |
| **PPO-Residual [Full-Batch]** | Soft-Gate Loophole / Peak Leakage | Unwinds `PSM'` curtailment during peaks (€46,168 cost, €570 unserved), leaking 465.6 violation hours. |
| **PPO-Residual [Minibatch]** | Insufficient Dual Penalty | Reduces violations slightly via curtailment (€7,265 unserved, €52,615 cost), still leaking 401.2 violation hours (1.51 kW mean exc). |
| **Shielded PPO [Full-Batch]** | "Battery Parking" Moral Hazard | External shield absorbs safety duties; actor gets zero gradient penalty for idling, leaving storage parked near 90% SoC ceiling (4.0% active SoC, 6.05 MWh). |
| **Shielded PPO [Minibatch]** | Suboptimal Coordination | Uncoordinated tug-of-war between policy exploration and shield overrides produces €58,987 cost (13,283 EUR unserved). |
| **PPO-Aux-Shield** | Suboptimal Curtailment | Indirect supervised divergence penalty (with `SoC`-masked discharge targets) yields €58,915 cost (1/5 `G2_pass`, 72.13 MWh). |
| **PPO-Diff-Shield (`final` / `best`)** | None (Best Safe RL Controller) | Active STE backprop in `distribution()` + `T1`–`T3` fixes pass G2 (€56,697 `final` / €55,403 `best`, 0.00% viol). |
| **PPO-Diff-6D (No `CapShield`)** | Conservative Continuous Motor Relaxation | Re-trained with `T1`'s `(1-0.70)/0.90` industrial slope, achieves 0.00% violations (`0.0h`), but continuous motor relaxation over-sheds load (86,380 EUR). |

### Key Pathologies Explained:

- **The "Battery Parking" Moral Hazard:**
  When a policy outputs actions into an external environment shield, the shield fixes unsafe actions by shedding load. Because the policy never observes physical cap violations, it receives no negative reinforcement for idling the battery during peaks. The policy learns that idling avoids battery cycling wear (0.02 EUR/kWh), while the external shield absorbs the grid violation by shedding customer load (1.00 EUR/kWh). Throughput collapses to 6.05 MWh, and curtailment costs surge to 14,193 EUR.
- **The Lagrangian Multiplier Oscillation:**
  Dual multipliers adjust with delay. During quiet hours, the multiplier decays toward zero. When a sudden demand spike occurs, the policy incurs violations, driving the multiplier up. By the time the multiplier is large enough to force conservative behavior, the peak has passed. The policy then aggressively curtails customer demand during normal hours before the multiplier decays again, repeating the cycle.
- **The In-the-Loop Differentiable Resolution:**
  By placing the safety projection inside the neural network forward pass with the T3 STE active surrogate gradient, the policy directly experiences constraint boundaries during gradient backpropagation. The agent learns proactive peak-shaving without parking storage or causing high-frequency chatter, while co-design with `CapShield` enforces the discontinuous industrial motor rule.

---

## 7. Systematic Comparison of "Safe" Approaches

| Dimension | Clamped Environment Interlock | Post-Hoc Environmental Shield | In-the-Loop Differentiable Safety Layer |
|---|---|---|---|
| **Enforcement Point** | Physical Environment (Battery) | Environment Wrapper (Post-Hoc Filter) | Policy Computational Graph (Internal Layer) |
| **Safety Scope** | Restricts battery charging only | Full grid cap protection (100% safe) | Full grid cap protection (100% safe with `CapShield`) |
| **Gradient Flow** | Indirect (via economic reward) | Disconnected (zero gradient to actor) | Non-vanishing analytical Jacobian + STE surrogate |
| **Susceptible to Moral Hazard?** | No | Yes ("Battery Parking" under full-batch) | No (policy directly penalized and guided) |
| **Asset Utilization** | Low–Moderate (5–25 MWh) | Bimodal (6.05 MWh parked or 58.00 MWh) | Active Peak Shaving (101.16 MWh `final` / 63.26 MWh `best`) |
| **Runtime Overhead** | Negligible | Minimal (0.15 ms P50) | Closed-form evaluation (1.14 ms P50) |
| **Validation Operating Cost** | 47,354 EUR (Unsafe: 450.4 violation hours) | 58,885 EUR – 58,987 EUR | **56,696.94 EUR `final` / 55,402.89 EUR `best` (Passes G2)** |

- **Clamped Interlock:** Essential physical baseline that prevents storage from charging into a capacity breach, but incapable of preventing load-driven overruns.
- **Environmental Shield:** Guarantees absolute safety at runtime, but distorts RL training dynamics by shielding the actor from the consequences of its decisions.
- **Differentiable Safety Layer:** The optimal paradigm for safe reinforcement learning, combining guaranteed runtime feasibility with direct gradient-based policy optimization.

> [!NOTE]
> **Co-Design with Environmental Shielding:** `PPODiffSafetyLayerAgent` embeds `SafePolicyNetwork` (closed-form 1-D battery projection with T3 STE surrogate active during gradient-enabled updates) to guide policy gradient updates with analytical constraint Jacobians while sampling raw actions cleanly. Because battery discharge is physically bounded by its 250 kW rating and 10% minimum state-of-charge, and industrial motor curtailment is discontinuous ($u_{\text{ind}} \in \{0\} \cup [0.70, 1.00]$), co-design with an outer `CapShield` guard remains essential economically—as shown by the `ci_ppo_diff_6d` ablation (re-trained with `T1`'s piecewise industrial slope, `ci_ppo_diff_6d` eliminates grid violations at 0.00%, yet continuous motor relaxation inflates total cost to 86,379.55 EUR).

---

## 8. Master Benchmark Results Tables

### 8.1 Regime 1 Validation Benchmark (Post-Fix T1–T3 80-Run Suite)

All figures evaluated across 5 random seeds on the 26-week Validation Split (H1 2019, 4,368 operating hours), matching `results/tables/per_seed_metrics.csv`, `results/tables/frozen_baselines.csv`, and `results/tables/regime_benchmarks_val.csv` verbatim:

| Controller Architecture | Total Cost (EUR) | Energy Purchase (EUR) | Battery Wear (EUR) | Unserved Penalty (EUR) | Throughput (MWh) | Active SoC (%) | Mean Exceedance (kW) | Violation Hours | Gate Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **LP' / MILP' Theoretical Oracle** | 53,290.51 | 44,192.97 | 436.73 | 8,660.86 | 21.84 | 37.98% / 37.96% | **0.0000** | **0.0h** | Global Lower Bound |
| **`MPCPolicy(perfect, H=24)`** | 53,404.83 | 44,261.26 | 383.39 | 8,760.19 | 19.17 | 30.0% | **0.0000** | **0.0h** | Non-Causal MPC (3.27 ms P50) |
| **`MPCPolicy(causal_noisy, H=24)`** | 53,514.28 | 44,251.67 | 395.56 | 8,867.05 | 19.78 | 33.6% | **0.0000** | **0.0h** | Causal MPC (3.41 ms P50) |
| **`causal23_ppo_diff_shield` (`best`)** | **55,268.23** | 44,531.36 | 992.29 | **9,744.57** | 49.61 | 33.8% | **0.0000** | **0.0h** | **Passes G1 & G2 (5/5; Mean < `Shield+PSM'`)** |
| **`clamped_ppo_diff_shield` (`best`)** | **55,402.89** | 44,609.72 | 1,265.17 | **9,528.01** | 63.26 | 40.7% | **0.0000** | **0.0h** | **Passes G1 & G2 (5/5; Mean < `PSM'`)** |
| **Peak-Shave-Minimum (PSM')** | 55,530.22 | 44,658.60 | 246.95 | 10,624.67 | 12.35 | 11.2% | **0.0000** | **0.0h** | **Passes G1 & G2** |
| **Shield + PSM'** | 55,500.35 | 44,660.23 | 246.95 | 10,593.17 | 12.35 | 11.2% | **0.0000** | **0.0h** | **Passes G1 & G2** |
| **`MPCPolicy(persistence, H=24)`** | 58,134.64 | 44,259.68 | 96.68 | 13,778.28 | 4.83 | 97.2% | **0.0000** | **0.0h** | Passes G1 & G2 (3.26 ms P50) |
| **Grid-Only (Unmanaged)** | 45,337.92 | 45,337.92 | 0.00 | 0.00 | 0.00 | 100.0% (static 50%) | 3.2571 | 468.0h | Fails G1 |
| **PPO [Unclamped] (`s3_ppo`, `v0`)** | 45,513.38 | 45,406.63 | 106.75 | 0.00 | 5.34 | 7.0% | 3.3497 | 475.4h | Fails G1 |
| **PPO [Clamped Full-Batch]** | 45,422.08 | 45,304.31 | 117.77 | 0.00 | 5.89 | 2.8% | 3.2338 | 464.8h | Fails G1 |
| **PPO [Clamped Minibatch]** | 47,354.23 | 45,308.74 | 506.46 | 1,539.02 | 25.32 | 16.8% | 3.0970 | 450.4h | Fails G1 |
| **PPO-Residual [Clamped FB]** | 46,167.91 | 45,429.36 | 168.44 | 570.12 | 8.42 | 12.6% | 2.8750 | 465.6h | Fails G1 |
| **PPO-Residual [Clamped MB]** | 52,615.24 | 44,978.38 | 372.28 | 7,264.57 | 18.61 | 31.5% | 1.5051 | 401.2h | Fails G1 |
| **`causal23_ppo_diff_shield` (`final`)** | **56,475.07** | 44,407.81 | 1,724.68 | 10,342.57 | 86.23 | 39.8% | **0.0000** | **0.0h** | **Passes G1 & G2 (4/5; 1.15 ms P50)** |
| **`clamped_ppo_diff_shield` (`final`)** | **56,696.94** | 44,668.49 | 2,023.20 | **10,005.25** | **101.16** | **50.5%** | **0.0000** | **0.0h** | **Passes G1 & G2 (4/5; 1.14 ms P50)** |
| **Shield + Idle (Zero-Battery)** | 58,644.70 | 44,417.53 | 0.00 | 14,227.17 | 0.00 | 100.0% (static 50%) | **0.0000** | **0.0h** | Passes G1; Fails G2 |
| **Shielded PPO [Clamped FB]** | 58,885.36 | 44,571.36 | 121.04 | 14,192.95 | 6.05 | 4.0% | **0.0000** | **0.0h** | Passes G1; Fails G2 |
| **PPO-Aux-Shield (Aux Loss)** | 58,914.89 | 44,542.36 | 1,442.52 | 12,930.01 | 72.13 | 35.4% | **0.0000** | **0.0h** | Passes G1; 1/5 G2 |
| **Shielded PPO [Clamped MB]** | 58,987.41 | 44,544.10 | 1,160.01 | 13,283.30 | 58.00 | 29.2% | **0.0000** | **0.0h** | Passes G1; 1/5 G2 |
| **PPO-Lagrangian [Clamped MB]** | 51,618.03 | 45,012.20 | 709.35 | 5,896.48 | 35.47 | 27.7% | 2.5058 | 409.8h | Fails G1 |

### 8.2 Regime 2 Validation: C&I Demand-Charge & Zone-VoLL Ablation (`regime_benchmarks_val.csv`)

Evaluated over 26 Val weeks with `2.333333 EUR/kW-week` weekly peak demand charge, zone VoLL (`industrial: 3.0`, `residential: 1.5`, `commercial: 1.2`, `stadium: 0.4`, `street: 0.2 EUR/kWh`), and `50.0 EUR` industrial trip switching penalty (`Shield + Idle` at 69,115.11 EUR beats the `ci_*` RL 5-seed means, while `MPCPolicy(causal_noisy, H=24)` achieves 65,250.30 EUR):

| Controller (Regime 2 Val) | Total Cost (EUR) | Demand Charge (EUR) | Unserved (EUR) | Wear (EUR) | Throughput (MWh) | Active SoC (%) | Trip (EUR) | Mean Exc (kW) | Viol. Rate (%) | Latency P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`LP'` / `MILP'`** | 64,513.29 | 17,111.13 | 1,957.80 | 1,252.28 | 62.61 | 72.07% / 72.21% | 0.00 | 0.0000 | 0.00% | — |
| **`MPCPolicy(causal_noisy, H=24)`** | 65,250.30 | 17,082.94 | 2,454.95 | 1,574.31 | 78.72 | 78.34% | 0.00 | 0.0000 | 0.00% | 3.69 |
| **`MPCPolicy(perfect, H=24)`** | 65,472.28 | 16,829.82 | 2,852.75 | 1,710.26 | 85.51 | 70.08% | 0.00 | 0.0000 | 0.00% | 3.52 |
| **`Grid-Only`** | 66,095.86 | 20,757.94 | 0.00 | 0.00 | 0.00 | 100.0% (static 50%) | 0.00 | 3.2571 | 10.71% | 0.002 |
| **`Shield + Idle`** | 69,115.11 | 19,660.75 | 5,036.83 | 0.00 | 0.00 | 100.0% (static 50%) | 0.00 | **0.0000** | **0.00%** | 0.002 |
| **`ci_ppo_diff_shield`** | **69,784.24 ± 617.02** | 20,261.46 | 4,456.67 | 635.13 | 31.76 | 16.71% | 0.00 | **0.0000** | **0.00%** | **1.18** |
| **`Shield + PSM'`** | 70,363.16 | 21,506.71 | 3,949.27 | 246.95 | 12.35 | 11.20% | 0.00 | 0.0000 | 0.00% | 0.036 |
| **`PSM'`** | 70,370.73 | 21,504.84 | 3,960.34 | 246.95 | 12.35 | 11.20% | 0.00 | 0.0000 | 0.00% | 0.035 |
| **`ci_ppo_shielded`** | 70,581.07 ± 605.99 | 20,594.16 | 4,894.68 | 654.80 | 32.74 | 17.38% | 0.00 | **0.0000** | **0.00%** | 0.15 |
| **`ci_ppo_diff_6d` (No `CapShield`)** | 86,379.55 ± 5,295.04 | 20,102.11 | 21,644.53 | 628.32 | 31.42 | 17.14% | 0.00 | **0.0000** | **0.00% (0.0h)** | 2.00 |

### 8.3 2019-W04 OOD Contingency & Tail-Risk Summary (`regime_benchmarks_val.csv`)

Evaluated across 5 contingencies with consistent noisy `CapShield` inputs (`set_last_obs`) and clean $t=0$ state/prices in `MPCPolicy`:

| Contingency (2019-W04) | `Shield + PSM'` Cost / Viol | `MPCPolicy(causal_noisy, H=24)` Cost / Viol | `Shield + MPCPolicy(causal_noisy, H=24)` Cost / Viol | `clamped_ppo_diff_shield` (5-seed Mean ± SD) Cost / $\text{CVaR}_{0.95}$ / Viol |
|---|---|---|---|---|
| **`nominal`** | 8,110.49 / 0h (0.00%) | **7,875.50** / 0h (0.00%) | **7,875.50** / 0h (0.00%) | 7,980.30 ± 177.99 / 120.42 / **0h (0.00%)** |
| **`pv_inverter_trip`** | 8,866.84 / 0h (0.00%) | **8,729.30** / 0h (0.00%) | **8,729.30** / 0h (0.00%) | 8,817.52 ± 131.23 / 121.35 / **0h (0.00%)** |
| **`bess_derate`** | 8,121.10 / 0h (0.00%) | **7,888.08** / 0h (0.00%) | **7,888.08** / 0h (0.00%) | 7,975.48 ± 169.96 / 120.42 / **0h (0.00%)** |
| **`grid_cap_reduction`** | **17,941.08** / 0h (0.00%) | **17,941.08** / 0h (0.00%) | **17,941.08** / 0h (0.00%) | 17,955.44 ± 30.18 / 186.62 / **0h (0.00%)** |
| **`forecast_noise`** | 8,105.01 / 36h (21.43%, 0.3354 kW) | **7,899.73** / 33h (19.64%, 0.3354 kW) | **7,899.73** / 34h (20.24%, 0.3354 kW) | 8,011.64 ± 151.06 / 120.45 / 35.2 ± 0.8h (20.95%, 0.3354 kW) |

*OOD Takeaway:* `Shield + MPCPolicy(causal_noisy, H=24)` matches or beats `clamped_ppo_diff_shield` across all five contingencies; under `forecast_noise`, all shielded controllers suffer ~33–36 violation hours (0.3354 kW mean exc, 120.45 EUR $\text{CVaR}_{0.95}$) because `CapShield` itself depends on noisy current PV/load observations (unshielded `PSM'` incurs 11.90% / 20h under `bess_derate`, 99.40% / 167h under `grid_cap_reduction`, and 11.31% / 19h under `forecast_noise`).

### 8.4 Locked One-Shot Test Split (`v0-prototype` Snapshot)

On the sealed 24-week H2 2019 Test Split (`results/tables/test_evaluation.csv`, 4,032 hours, deliberately NOT re-unlocked after T1–T13), milder autumn demand compresses unmanaged exceedances to 3.79% (153.0h vs. 468.0h on Val) and narrows the `PSM'`-to-`LP'` optimality headroom to 682.60 EUR (`LP'`: 36,991.67 EUR). Consequently, while `clamped_ppo_diff_shield` enforces strict safety (0.00% violations, 0.0000 kW exceedance), its mean Test cost of **39,884.02 ± 521.60 EUR** trails both **`PSM'` (37,674.27 EUR)** and the zero-battery **`Shield + Idle` baseline (39,759.47 EUR)** because active battery cycling incurs wear without recovering enough arbitrage value under mild load profiles.

> [!NOTE]
> **Reconciliation with Run Artifacts & Locked Test Provenance:**
> - **T1–T3 Post-Fix Proof:** `clamped_ppo_diff_shield` improved from €58,473.79 ± €1,180.66 (`v0-prototype`) to **€56,696.94 ± €1,279.45 (`final`)** (seeds 101–105: €55,045.60, €55,968.83, €56,649.27, €58,324.76, €57,496.22) and **€55,402.89 ± €1,212.20 (`best`)** (seeds 101–105: €54,543.46, €55,858.96, €54,337.06, €54,972.21, €57,302.78). Both 5-seed `best` means (`clamped_ppo_diff_shield` and `causal23_ppo_diff_shield`) and `final` Seed 101 beat strict 1.00× `PSM'` (€55,530.22).
> - **Locked Test Split Preserved:** `results/tables/test_evaluation.csv`, `results/tables/test_stress_weeks.csv`, and `results/runs/test_lock.json` remain the **locked one-shot `v0-prototype` Test snapshot** and were deliberately NOT unlocked or re-evaluated after T1–T13 fixes.

---

## 9. Report Figures & Evaluation Suite

Multi-regime and OOD Val benchmarks are generated via [`tools/report/eval_regimes_val.py`](tools/report/eval_regimes_val.py), and the 12 consolidated PNG benchmark figures are stored in `results/figures/presentation/`:
1. **[`01_ablation_pathologies.png`](results/figures/presentation/01_ablation_pathologies.png)**: 3-panel ablation comparing cost decomposition, cap violation rates, and active SoC across `Grid-Only`, `PPO (Full-Batch)`, `PPO (Minibatch)`, `PPO-Lagrangian`, `Shielded PPO`, and `PPO-Diff-Shield`.
2. **[`02_pareto_cost_safety.png`](results/figures/presentation/02_pareto_cost_safety.png)**: Broken-X 3-column Pareto view separating the 0.00 kW exact-zero safe lane, log-scale leaky/unsafe exceedance, and P50 step inference latency.
3. **[`03_learning_dynamics.png`](results/figures/presentation/03_learning_dynamics.png)**: 50k–300k step Validation learning curves (mean ± 1 SD across 5 seeds) tracking total cost, cap violation rate, and active unpinned SoC.
4. **[`04_regime1_causal_mpc.png`](results/figures/presentation/04_regime1_causal_mpc.png)**: Regime 1 Cleveland dot + per-seed interval benchmark comparing `LP'`/`MILP'`, `MPCPolicy` (`perfect`, `causal_noisy`, `persistence`), `PSM'`, `clamped_ppo_diff_shield`, and `causal23_ppo_diff_shield` on cost, $\text{CVaR}_{0.95}$, and P50 latency.
5. **[`05_regime2_ci_ablation.png`](results/figures/presentation/05_regime2_ci_ablation.png)**: Regime 2 C&I tariff stacked cost decomposition (energy, demand charge, zone VoLL, wear, trip penalty) and non-energy / motor-leakage ablation (`ci_ppo_diff_shield` vs. `PSM'`, `ci_ppo_shielded`, and unshielded `ci_ppo_diff_6d`).
6. **[`06_ood_stress_2019w04.png`](results/figures/presentation/06_ood_stress_2019w04.png)**: 2-panel 2019-W04 OOD contingency matrix (`nominal`, `pv_inverter_trip`, `bess_derate`, `grid_cap_reduction`, `forecast_noise`) comparing violation rate and weekly cost / $\text{CVaR}_{0.95}$ across `PSM'`, `MPCPolicy(causal_noisy, H=24)`, and `clamped_ppo_diff_shield`.
7. **[`07_dispatch_case_2019w04.png`](results/figures/presentation/07_dispatch_case_2019w04.png)**: 3-row synchronized full-168h dispatch case study on Val `2019-W04` (net grid import vs. cap, battery SoC cycling vs. parking, and cumulative unserved load).
8. **[`08_seasonal_test_context.png`](results/figures/presentation/08_seasonal_test_context.png)**: Paired seasonal comparison (Val H1 vs. locked `v0-prototype` Test H2 arbitrage headroom) and stress-week dumbbell from mild Test weeks (`W30`/`W45`/`W48`/`W51`) to cap-binding `2019-W49`.
9. **[`09_microgrid_architecture.png`](results/figures/presentation/09_microgrid_architecture.png)**: CMDP block diagram mapping the 17-D/23-D state space, physical assets (`368.15 kW` PCC cap, `250 kWp` PV, `500 kWh / ±250 kW` BESS, 5 demand zones with the discontinuous industrial motor rule), and 6-D safety projection into `G1`/`G2` gates.
10. **[`10_dataset_split_regimes.png`](results/figures/presentation/10_dataset_split_regimes.png)**: 154-week chronological split timeline (`104w` Train / `26w` Val / `24w` Test).
11. **[`11_safety_mechanisms.png`](results/figures/presentation/11_safety_mechanisms.png)**: 3-track forward and backward gradient flow schematic contrasting post-hoc `CapShield` (gradient disconnect and battery parking), `v0` 1-D analytical clipping (vanishing active Jacobian), and T3 STE (`α=0.1`, `forward_raw`) + 5-step `CapShield` co-design.
12. **[`12_seed_stability_tradeoff.png`](results/figures/presentation/12_seed_stability_tradeoff.png)**: 5-seed reproducibility spread (Seeds 101–105) across Regimes 1–2 alongside 24h `MPCPolicy` forecast sensitivity (`perfect` vs. `causal_noisy` vs. `persistence`) and unserved penalty spike.

---

## 10. Key Conclusions & Limitations

1. **Soft Penalties Fail in Cost-Asymmetric Regimes:** In systems with high curtailment penalties (1.00 EUR/kWh) relative to electricity prices, standard RL algorithms intentionally violate capacity limits during peak events.
2. **Post-Hoc Safety Filters Cause Moral Hazards:** Applying an external shield without gradient propagation incentivizes the neural policy to park storage assets and rely on expensive customer curtailment.
3. **Fixing Active Gradients & Truncation Unlocks G2 Compliance:** Propagating T3 STE gradients through `SafePolicyNetwork.distribution()` alongside T2 truncation/bootstrap fixes drops `clamped_ppo_diff_shield` Val cost to €56,696.94 (`final`) / €55,402.89 (`best`), passing both `1.05 × PSM'` (`4/5` `final`, `5/5` `best`) and `1.00 × PSM'` on `best` mean while executing ~3.0× faster than causal LP-MPC.
4. **Co-Designed Guards & Realistic Benchmarking:** In Regime 2, `ci_ppo_diff_shield` (€69,784.24) beats `PSM'` (€70,370.73), `Shield + PSM'` (€70,363.16), and `ci_ppo_shielded` (€70,581.07), while `Shield + Idle` (€69,115.11) leads the `ci_*` RL means and `MPCPolicy(causal_noisy, H=24)` reaches €65,250.30. Under 2019-W04 OOD contingencies, `Shield + MPCPolicy(causal_noisy, H=24)` matches/beats `clamped_ppo_diff_shield` across all five perturbations, and both suffer ~33–36h violations under `forecast_noise` because `CapShield` relies on noisy current PV/load readings.

### Limitations
- **Copperplate Single-Bus Network:** Omits AC branch power flows, bus voltage limits, line thermal ratings, and reactive power ($Q$).
- **Synthetic Zone Disaggregation & `proxy_DE` Prices:** Disaggregates French national load deterministically across 5 zones and splices `AT` $\to$ `DE-LU` spot prices at `2018-09-30 22:00 UTC`.
- **Linear Battery Wear & Hourly Discretization:** Uses constant `0.02 EUR/kWh` throughput wear (omitting temperature/DoD/SEI aging) and 1.0 h timesteps.
- **Locked Test Snapshot (`v0-prototype`):** Post-fix T1–T13 evaluation is reported on Validation and 2019-W04 OOD; locked Test tables (`test_evaluation.csv`, `test_stress_weeks.csv`, `test_lock.json`) retain the one-shot `v0-prototype` snapshot.
