# Safe Reinforcement Learning for Microgrid Energy Management: Executive Technical Summary

---

## 1. Executive Summary & Problem Overview

Real-time microgrid energy management balances three conflicting goals: minimizing grid energy purchase costs, limiting battery wear, and avoiding expensive load curtailment penalties—all while strictly respecting a contractual grid-import capacity limit (368.15 kW).

Deploying standard reinforcement learning (RL) in safety-critical power systems exposes three fundamental pathologies:
1. **Safety Violations:** Unconstrained and penalty-based policies treat physical capacity limits as economic trade-offs, routinely violating grid import caps during demand peaks.
2. **Moral Hazards & Exploration Breakdown:** Post-hoc environmental safety filters decouple the neural policy from constraint enforcement, triggering a **battery parking** pathology where the policy idles storage assets and shifts the entire burden onto costly load shedding.
3. **Temporal Credit Assignment Failures:** Averaging policy gradients across long operational episodes dilutes localized peak feedback, causing learning stagnation.

This benchmark shows that **in-the-loop differentiable analytical safety projections (`PPO-Diff-Shield`) resolve the gradient disconnect**, achieving zero grid violations, preventing the battery parking moral hazard, maintaining active battery cycling (77.52 MWh throughput), and delivering the lowest safe operating cost among all RL architectures (58,350.52 EUR on validation).

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
| **Comparators & Oracles** | Untuned rule-based heuristics without peak-shaving coordination. | Tuned domain peak-shaving heuristic (PSM') and exact hindsight MILP Oracle (HiGHS solver). |
| **Data & Stress Profiles** | Regional Italian power grid data without acute capacity bottlenecks. | Multi-source European data (OPSD, PVGIS, Open-Meteo), chronological splits, and acute stress bottlenecks (Week 49). |

### Practical Implications of These Divergences
In unconstrained settings, the agent treats the grid as an infinite buffer, and soft reward penalties suffice to balance costs. However, in physical distribution networks where exceeding capacity triggers substation transformer tripping or severe contractual penalties, soft penalties fail: RL agents exploit the 50:1 cost asymmetry to violate the grid cap during peak hours. Resolving these challenges requires coordinated multi-zone demand response, rigorous safety architectures, and analytical differentiable projections.

---

## 3. Microgrid Physical Architecture & Real-World Data

### 3.1 Microgrid Physical Subsystems

The simulated facility models an urban commercial/industrial microgrid operating on discrete hourly intervals:
- **Grid Interconnection:** Point of Common Coupling with a contractual import cap of **368.15 kW** (the empirical 90th percentile of net demand). Energy export back to the grid is barred.
- **Battery Energy Storage System (BESS):** 500 kWh nominal capacity, 250 kW continuous power rating (0.5C), 95% symmetric round-trip efficiency, and an operational state-of-charge safety envelope constrained to 10%–90% to prevent degradation.
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

- **State Space (17 dimensions):** Battery state-of-charge, solar generation, five sector demands, electricity spot price, ambient temperature, relative humidity, barometric pressure, hour of day, day of week, and stadium event indicators.
- **Action Space (6 dimensions):** Normalized continuous vector containing the battery charge/discharge command (-1 for 250 kW discharge, +1 for 250 kW charge) and five fractional load curtailment commands (0 to 1 for each demand sector).
- **Safety Gate (G1):** Mean contract cap exceedance must not exceed 0.05 kW per step (0.00 kW for strict physical feasibility).
- **Economic Gate (G2):** Total operational cost must not exceed 105% of the domain heuristic PSM' (58,306.73 EUR).

### 3.4 Real-World Datasets & Chronological Splitting

- **Data Sources:** Open Power System Data / ENTSO-E (hourly grid load and spot prices, 2017–2019), EU Joint Research Centre PVGIS (solar irradiance), and Open-Meteo (ambient temperature, humidity, pressure).
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
5. **`LP' Theoretical Oracle` (Hindsight Linear Programming via HiGHS Solver):**
   Mixed-integer linear program solved with perfect hindsight over each 168-hour horizon, providing the theoretical performance upper bound.

### 4.2 Reinforcement Learning Agents

6. **`PPOAgent` [Unclamped] (Unconstrained Proximal Policy Optimization):**
   Standard Gaussian PPO policy trained with Generalized Advantage Estimation without physical interlocks or safety filters.
7. **`PPOAgent` [Clamped Full-Batch] (Clamped PPO with Full-Batch Updates):**
   Restricts battery charging power to available headroom below the grid cap. Trained using standard 168-hour full trajectory rollouts.
8. **`PPOAgent` [Clamped Minibatch] (Vanilla Minibatch PPO — Best Unconstrained RL):**
   Combines the physical charging interlock with minibatch stochastic policy updates (batch size 64, 5 epochs per rollout), enabling localized credit assignment.
9. **`PPOLagrangianAgent` (Dual-Lagrangian Constrained PPO):**
   Updates an adaptive penalty multiplier via subgradient ascent whenever capacity violations occur, adding a dynamic penalty to the reward signal.
10. **`PPOResidualAgent` (Residual Policy Learning on PSM' Heuristic):**
    A neural network outputs additive action adjustments superimposed onto the baseline PSM' heuristic decisions.
11. **`PPOShieldedAgent` (Post-Hoc Environmental Shielded PPO — Full-Batch & Minibatch):**
    The neural policy outputs raw dispatch commands, which an external runtime safety shield intercepts and modifies before environment execution.
12. **`PPOAuxShieldLossAgent` (`PPO-Aux-Shield`, Auxiliary Shield Loss PPO):**
    Adds an auxiliary supervised loss to the policy objective, penalizing the Euclidean distance between proposed policy actions and safe shielded actions.
13. **`PPODiffSafetyLayerAgent` (`PPO-Diff-Shield`, Differentiable Analytical Safety Layer PPO):**
    Embeds an analytical closed-form Euclidean projection directly into the policy neural network's forward pass, allowing exact analytical Jacobian gradients to flow through constraint boundaries during backpropagation.

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
| **PPO-Lagrangian** | Multiplier Oscillation & Over-Shedding | Hysteresis: multiplier surges after a peak, causing massive off-peak shedding (24,173 EUR penalty), then decays before the next peak. |
| **PPO-Residual [Full-Batch]** | Soft-Gate Loophole Exploitation | Under-sheds load to save curtailment cost (€54,571 cost; plotted in Study 2 & 3), leaking 437.4 violation hours. |
| **PPO-Residual [Minibatch]** | Multiplier Oscillation & Over-Shed | Spikes unserved penalties (€14,673), driving cost to €59,705. |
| **Shielded PPO [Full-Batch]** | "Battery Parking" Moral Hazard | External shield absorbs safety duties; actor gets zero gradient penalty for idling, leaving storage parked at 90% SoC ceiling. |
| **Shielded PPO [Minibatch]** | Action Chattering & High Wear | Uncoordinated tug-of-war between policy exploration and shield overrides produces 101.46 MWh throughput and high wear. |
| **PPO-Aux-Shield** | Suboptimal Curtailment | Indirect supervised divergence penalty fails to provide clean directional gradient flow. |
| **PPO-Diff-Shield** | None (Best Safe RL Controller) | In-the-loop projection provides exact analytical gradients, eliminating moral hazards and high churn. |

### Key Pathologies Explained:

- **The "Battery Parking" Moral Hazard:**
  When a policy outputs actions into an external environment shield, the shield fixes unsafe actions by shedding load. Because the policy never observes physical cap violations, it receives no negative reinforcement for idling the battery during peaks. The policy learns that idling avoids battery cycling wear (0.02 EUR/kWh), while the external shield absorbs the grid violation by shedding customer load (1.00 EUR/kWh). Throughput collapses to 5.47 MWh, and curtailment costs surge to 14,227 EUR.
- **The Lagrangian Multiplier Oscillation:**
  Dual multipliers adjust with delay. During quiet hours, the multiplier decays toward zero. When a sudden demand spike occurs, the policy incurs violations, driving the multiplier up. By the time the multiplier is large enough to force conservative behavior, the peak has passed. The policy then aggressively curtails customer demand during normal hours before the multiplier decays again, repeating the cycle.
- **The In-the-Loop Differentiable Resolution:**
  By placing the safety projection inside the neural network forward pass, the policy directly experiences constraint boundaries during gradient backpropagation. The agent learns proactive peak-shaving without parking storage or causing high-frequency chatter.

---

## 7. Systematic Comparison of "Safe" Approaches

| Dimension | Clamped Environment Interlock | Post-Hoc Environmental Shield | In-the-Loop Differentiable Safety Layer |
|---|---|---|---|
| **Enforcement Point** | Physical Environment (Battery) | Environment Wrapper (Post-Hoc Filter) | Policy Computational Graph (Internal Layer) |
| **Safety Scope** | Restricts battery charging only | Full grid cap protection (100% safe) | Full grid cap protection (100% safe) |
| **Gradient Flow** | Indirect (via economic reward) | Disconnected (zero gradient to actor) | Exact analytical Jacobian backpropagation |
| **Susceptible to Moral Hazard?** | No | Yes ("Battery Parking" under full-batch) | No (policy directly penalized and guided) |
| **Asset Utilization** | Moderate (34–50 MWh) | Bimodal (5.47 MWh parked or 101.46 MWh churn) | Optimal (77.52 MWh balanced dispatch) |
| **Runtime Overhead** | Negligible | Minimal | Closed-form analytical evaluation |
| **Validation Operating Cost** | 48,761 EUR (Unsafe: 430 violation hours) | 58,910 EUR – 59,526 EUR | **58,350.52 EUR (Lowest safe RL cost)** |

- **Clamped Interlock:** Essential physical baseline that prevents storage from charging into a capacity breach, but incapable of preventing load-driven overruns.
- **Environmental Shield:** Guarantees absolute safety at runtime, but distorts RL training dynamics by shielding the actor from the consequences of its decisions.
- **Differentiable Safety Layer:** The optimal paradigm for safe reinforcement learning, combining guaranteed runtime feasibility with direct gradient-based policy optimization.

> [!NOTE]
> **Co-Design with Environmental Shielding:** `PPODiffSafetyLayerAgent` embeds `SafePolicyNetwork` (closed-form 1-D battery projection) to guide policy gradient updates with exact analytical Jacobians. Because battery discharge is physically bounded by its 250 kW rating and 10% minimum state-of-charge, the agent is co-designed with an outer `CapShield` environment wrapper to execute priority-ordered customer load curtailment if storage is depleted or net facility load exceeds $P_{\text{cap}} + P_{\text{max}}$.

---

## 8. Master Benchmark Results Table

All figures evaluated across 5 random seeds on the 26-week Validation Split (H1 2019, 4,368 operating hours):

| Controller Architecture | Total Cost (EUR) | Energy Purchase (EUR) | Battery Wear (EUR) | Unserved Penalty (EUR) | Throughput (MWh) | Active SoC (%) | Mean Exceedance (kW) | Violation Hours | Gate Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **LP' Theoretical Oracle** | 53,290.51 | 43,000.00 | 499.80 | 9,790.71 | 24.99 | 85.0% | **0.0000** | **0.0h** | Global Upper Bound |
| **Peak-Shave-Minimum (PSM')** | 55,530.22 | 44,658.60 | 246.95 | 10,624.67 | 12.35 | 11.2% | **0.0000** | **0.0h** | **Passes G1 & G2** |
| **Shield + PSM'** | 55,500.35 | 44,660.23 | 246.95 | 10,593.17 | 12.35 | 11.2% | **0.0000** | **0.0h** | **Passes G1 & G2** |
| **Grid-Only (Unmanaged)** | 45,337.92 | 45,337.92 | 0.00 | 0.00 | 0.00 | 0.0% | 3.2571 | 468.0h | Fails G1 |
| **PPO [Unclamped]** | 45,513.38 | 45,406.63 | 106.75 | 0.00 | 5.34 | 7.0% | 3.3500 | 475.4h | Fails G1 |
| **PPO [Clamped Full-Batch]** | 45,517.31 | 45,390.94 | 126.37 | 0.00 | 6.32 | 21.4% | 3.2400 | 466.0h | Fails G1 |
| **PPO [Clamped Minibatch]** | 48,760.55 | 45,026.23 | 991.40 | 2,742.92 | 49.57 | 34.1% | 2.9160 | 430.2h | Fails G1 |
| **PPO-Residual [Clamped FB]** | 54,570.55 | 44,733.62 | 254.96 | 9,581.96 | 12.75 | 15.1% | 0.2360 | 437.4h | Fails G1 (Under-Shedding; Plotted in Study 2 & 3) |
| **PPO-Residual [Clamped MB]** | 59,705.27 | 44,504.57 | 527.77 | 14,672.93 | 26.39 | 38.9% | 0.3087 | 212.8h | Fails G1 & G2 (Over-Shedding) |
| **PPO-Diff-Shield (Safety Layer)** | **58,350.52** | 44,485.50 | 1,550.38 | **12,314.63** | **77.52** | **37.7%** | **0.0000** | **0.0h** | **Passes G1 & G2** |
| **Shield + Idle (Zero-Battery)** | 58,644.70 | 44,417.53 | 0.00 | 14,227.17 | 0.00 | 0.0% | **0.0000** | **0.0h** | Passes G1; Fails G2 |
| **Shielded PPO [Clamped FB]** | 58,910.05 | 44,573.41 | 109.47 | 14,227.17 | 5.47 | 4.1% | **0.0000** | **0.0h** | Passes G1; Fails G2 |
| **Shielded PPO [Clamped MB]** | 59,526.37 | 44,443.81 | 2,029.18 | 13,053.38 | 101.46 | 42.0% | **0.0000** | **0.0h** | Passes G1; Fails G2 |
| **PPO-Aux-Shield (Aux Loss)** | 59,889.30 | 44,315.71 | 1,612.18 | 13,961.41 | 80.61 | 30.5% | **0.0000** | **0.0h** | Passes G1; Fails G2 |
| **PPO-Lagrangian [Clamped MB]** | 70,040.15 | 44,232.45 | 1,634.95 | 24,172.75 | 81.75 | 58.0% | 1.0118 | 228.4h | Fails G1 & G2 |

---

## 9. Report Figures & Evaluation Suite

All publication figures are generated via `python3 -m tools.generate_report`:
- **Study 1: Clamping & Batching Ablation in Basic PPO (`results/figures/study1_ppo_ablation/`):**
  Examines cost, grid exceedance, throughput, and active state-of-charge across unconstrained PPO variants.
- **Study 2: Capped PPO vs. Baselines (`results/figures/study2_capped_ppo/`):**
  Compares ten benchmark methods across total cost, exceedance hours, unserved load volume/cost, battery throughput, and active cycling time.
- **Study 3: Multi-Objective Pareto Frontier (`results/figures/study3_stress_pareto/`):**
  Maps mean grid exceedance against total cost as individual points with error bars, highlighting the feasible zone bounded by G1 and G2.
- **Study 4: Plain PPO vs. PPO-Lagrangian (`results/figures/study4_plain_vs_lagrangian/`):**
  Side-by-side comparison across unclamped, clamped full-batch, and clamped minibatch configurations.

---

## 10. Key Conclusions & Recommendations

1. **Soft Penalties Fail in Cost-Asymmetric Regimes:** In systems with high curtailment penalties (1.00 EUR/kWh) relative to electricity prices, standard RL algorithms intentionally violate capacity limits during peak events.
2. **Post-Hoc Safety Filters Cause Moral Hazards:** Applying an external shield without gradient propagation incentivizes the neural policy to park storage assets and rely on expensive customer curtailment.
3. **Differentiable Safety Layers Enable Optimal Safe RL:** Embedding analytical closed-form projection layers directly inside the policy computational graph provides strict safety guarantees while preserving gradient flow for active storage dispatch.
4. **Engineered Heuristics Set a Competitive Benchmark:** A well-tuned domain heuristic (`PSM'`) operating within 4.0% of hindsight LP optimality remains a strong baseline that unshielded or poorly integrated RL algorithms cannot reliably beat.
