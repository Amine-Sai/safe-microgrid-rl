# Safe Reinforcement Learning for Constrained Microgrid Energy Management

An empirical benchmark evaluating model-free reinforcement learning against tuned peak-shaving heuristics and mathematical programming oracles under contractual grid-import capacity constraints.

---

## Abstract

Real-time microgrid energy management requires balancing electricity acquisition costs, battery degradation, and customer curtailment penalties while strictly respecting contractual grid-import limits. While deep reinforcement learning (RL) has demonstrated effectiveness in unconstrained economic dispatch, its deployment in safety-critical power systems is challenged by constraint violations, exploration instability, and credit assignment failures. 

This repository provides an empirical benchmark evaluating continuous-action safe reinforcement learning paradigms on an urban microgrid with a 500 kWh / 250 kW battery energy storage system (BESS), 250 kW peak rooftop photovoltaic (PV) generation, and five heterogeneous demand zones. Under a contractual grid-import limit of $P_{\text{cap}} = 368.15\,\text{kW}$ (the 90th percentile of unmanaged net facility demand), we evaluate unconstrained Proximal Policy Optimization (PPO), dual-Lagrangian PPO, residual policy learning, post-hoc environmental safety shielding, and in-the-loop differentiable safety layers against a tuned domain heuristic (Peak-Shave-Minimum, PSM'), an exact continuous linear programming hindsight oracle (LP'), and an exact mixed-integer hindsight oracle (MILP' via HiGHS). Evaluations span 104 weeks of training (2017–2018), 26 weeks of validation (H1 2019, 4,368 h), and 24 weeks of locked out-of-sample testing (H2 2019, 4,032 h) across 5 independent random seeds.

The empirical analysis establishes five core quantitative conclusions:
1. **Unconstrained and dual-Lagrangian PPO fail to maintain grid feasibility**, incurring between 409 and 475 violation hours per semester with up to 3.35 kW mean exceedance. After fixing the dual-Lagrangian cadence to once per rollout (`T2`, $\lambda_{\text{end}} = 0.314\text{--}0.369$), full-batch `clamped_ppo_lagrangian` (`€45,660.86`, `3.25 kW`, `468.0h`) and minibatch `clamped_mb_ppo_lagrangian` (`€51,618.03`, `2.51 kW`, `409.8h`) remain unsafe at the baseline dual step size ($\alpha_\lambda = 0.01$).
2. **Residual policy learning passes soft mean-magnitude thresholds via an empirical loophole** or leaks peaks, distributing low-magnitude overruns across 335.8–465.6 operating hours per semester (7.7%–10.7% of the semester) while under-shedding critically during acute demand surges.
3. **Post-hoc environmental shielding induces an acute "battery parking" moral hazard under full-batch policy gradients**: downstream safety overrides absorb constraint handling, causing the policy network to park the battery near its state-of-charge ceiling (3.97% active cycling time, 6.05 MWh throughput), incurring €14,192.95 in unserved load penalties and nearly matching a zero-battery control baseline (€58,644.70 vs. €58,885.36).
4. **In-the-loop differentiable safety projections resolve the gradient disconnect**, and fixing `SafePolicyNetwork.distribution()` to propagate active-constraint STE gradients ($\alpha_{\text{ste}} = 0.1$) into PPO updates plus truncation/bootstrap handling (`T1`–`T3`) improved `clamped_ppo_diff_shield` Validation cost from **€58,473.79 ± €1,180.66** (`v0-prototype`) to **€56,696.94 ± €1,279.45 (`final`) / €55,402.89 ± €1,212.20 (`best`)** with **0.00% violations** (and causal 23-D `causal23_ppo_diff_shield` to **€56,475.07 ± €1,311.65 `final` / €55,268.23 ± €685.32 `best`**). Mean `final` performance passes the canonical $G_2 \le 1.05 \times \text{PSM'}$ gate (€58,306.73) with 4/5 `final` and 5/5 `best` seeds passing, both `best` means and `final` Seed 101 (**€55,045.60**) beat strict `1.00× PSM'` (€55,530.22), and execution takes **1.14 ms P50** (~3.0× faster than causal LP-MPC at 3.41 ms).
5. **Generalization to C&I tariffs (Regime 2) and OOD contingencies (2019-W04):** Under weekly peak demand charges, heterogeneous zone VoLL, and industrial trip penalties, `ci_ppo_diff_shield` (**€69,784.24 ± €617.02**, 0.00% violations) outperforms `PSM'` (€70,370.73), `Shield + PSM'` (€70,363.16), and `ci_ppo_shielded` (€70,581.07 ± €605.99), though the zero-battery `Shield + Idle` baseline (€69,115.11) beats the `ci_*` RL 5-seed means (only `ci_ppo_diff_shield` Seed 105 at €68,906.11 beats `Shield + Idle`) and causal `MPCPolicy(causal_noisy, H=24)` reaches €65,250.30. Under 2019-W04 OOD contingencies (5 seeds, consistent noisy `CapShield` observations via `set_last_obs`, clean $t=0$ in `MPCPolicy`), `clamped_ppo_diff_shield` maintains 0.00% violations on `nominal` (**€7,980.30 ± €177.99**), `pv_inverter_trip`, `bess_derate`, and `grid_cap_reduction`, while `Shield + MPCPolicy(causal_noisy, H=24)` matches/beats `clamped_ppo_diff_shield` across all five contingencies; under `forecast_noise`, all shielded controllers suffer ~33–36 violation hours (19.6%–21.4%, 0.3354 kW mean exc) because `CapShield` itself relies on noisy current PV/load readings.

---

## Problem Formulation

The microgrid dispatch problem is formulated as an infinite-horizon discounted Constrained Markov Decision Process (CMDP), defined by the tuple $\langle \mathcal{S}, \mathcal{A}, P, R, C, d, \gamma \rangle$, where:
- $\mathcal{S} \subset \mathbb{R}^{17}$ denotes the continuous observation state space.
- $\mathcal{A} = [-1, 1]^6 \subset \mathbb{R}^6$ represents the continuous action space (battery command plus 5-zone shedding commands mapped linearly via $s_z = (a_z + 1)/2 \in [0, 1]$).
- $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ denotes the environmental transition probability distribution.
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the economic return function ($R(s_t, a_t) = -c_t$).
- $C: \mathcal{S} \times \mathcal{A} \to \mathbb{R}_{\ge 0}$ is the instantaneous constraint cost function measuring contractual grid-import capacity exceedance.
- $d \in \mathbb{R}_{\ge 0}$ is the constraint violation tolerance budget ($d = 0.05\,\text{kW/step}$ soft $G_1$ threshold; $d = 0.00\,\text{kW/step}$ for strict physical feasibility).
- $\gamma \in [0, 1)$ is the temporal discount factor ($\gamma = 0.99$).

The operational decision interval is discrete hourly steps ($\Delta t = 1.0\text{ h}$) evaluated over weekly episodes ($T = 168\text{ h}$).

### Formal State and Action Spaces

The 17-dimensional state vector $s_t \in \mathcal{S}$ exposes physical asset states, meteorological features, and calendar signals:

$$
s_t = \begin{bmatrix}
\text{SoC}_t,\; P_{\text{pv}, t},\; P_{\text{load}, 1, t},\; \dots,\; P_{\text{load}, 5, t},\; p_t,\; T_{\text{amb}, t},\; RH_t,\; BP_t,\;
\sin\left(\frac{2\pi h_t}{24}\right),\; \cos\left(\frac{2\pi h_t}{24}\right),\; \sin\left(\frac{2\pi D_t}{7}\right),\; \cos\left(\frac{2\pi D_t}{7}\right),\;
\mathbb{I}_{\text{event}, t},\; \mathbb{I}_{\text{event-next-1h}, t}
\end{bmatrix}^\top
$$

where $\text{SoC}_t \in [0, 1]$ is the battery state-of-charge, $P_{\text{pv}, t} \ge 0$ is current rooftop solar power (kW), $P_{\text{load}, z, t} \ge 0$ are the five zonal facility loads (kW), $p_t$ is the day-ahead spot electricity price (EUR/kWh), $T_{\text{amb}, t}$, $RH_t$, and $BP_t$ denote ambient temperature ($^\circ\text{C}$), relative humidity (%), and barometric pressure (hPa), $h_t \in [0, 23]$ and $D_t \in [0, 6]$ encode diurnal and weekly periodicity, $\mathbb{I}_{\text{event}, t} \in \{0, 1\}$ indicates active stadium match events, and $\mathbb{I}_{\text{event-next-1h}, t} \in \{0, 1\}$ is the 1-hour event lookahead indicator.

The 6-dimensional continuous control action $a_t \in \mathcal{A}$ coordinates energy storage and demand response:

$$
a_t = \begin{bmatrix} a_{\text{bat}, t}, a_{\text{shed}, 1, t}, a_{\text{shed}, 2, t}, a_{\text{shed}, 3, t}, a_{\text{shed}, 4, t}, a_{\text{shed}, 5, t} \end{bmatrix}^\top \in [-1, 1]^6
$$

where $a_{\text{bat}, t} \in [-1, 1]$ commands battery continuous power ($-1.0 \implies 250\,\text{kW}$ discharge, $+1.0 \implies 250\,\text{kW}$ charge), and $a_{\text{shed}, z, t} \in [-1, 1]$ maps linearly via $s_{z, t} = (a_{\text{shed}, z, t} + 1)/2 \in [0, 1]$ to the fractional curtailment ratio applied to demand sector $z$ (residential, commercial, stadium, street, industrial).

### Physical Microgrid Components

The modeled infrastructure comprises four coupled physical subsystems:

1. **Utility Grid Connection & Power Balance:**
   The point of common coupling (PCC) connects the facility to the transmission grid. Net served load after customer curtailment and solar self-consumption is:

   $$
   P_{\text{net-served}, t} = \sum_{z=1}^5 (1 - s_{z, t}) P_{\text{load}, z, t} - P_{\text{pv}, t}
   $$

   Real-time grid power import $P_{\text{grid}, t}$ balances facility demand and battery power:

   $$
   P_{\text{grid}, t} = \max\left(0, P_{\text{net-served}, t} + P_{\text{bat}, t}\right)
   $$

   Reverse power injection back into the utility grid is barred ($P_{\text{grid}, t} \ge 0$).

2. **Battery Energy Storage System (BESS):**
   The storage system possesses a nominal energy capacity $E_{\text{nom}} = 500\text{ kWh}$ and continuous power rating $P_{\text{bat}}^{\text{max}} = 250\text{ kW}$ (a 0.5C storage architecture). The internal state-of-charge $\text{SoC}_t \in [0, 1]$ evolves according to:

   $$\text{SoC}_{t+1} = \text{SoC}_t + \frac{\Delta t}{E_{\text{nom}}} \left( \eta_{\text{ch}} P_{\text{ch}, t} - \frac{1}{\eta_{\text{dis}}} P_{\text{dis}, t} \right)$$

   where one-way charge and discharge efficiencies are parameterized as $\eta_{\text{ch}} = \eta_{\text{dis}} = 0.95$ (90.25% round-trip efficiency). Operational boundaries are strictly confined to $\text{SoC} \in [\text{SoC}_{\text{min}}, \text{SoC}_{\text{max}}] = [0.10, 0.90]$ to prevent accelerated cell degradation.

3. **Solar Photovoltaic Generation:**
   Rooftop solar generation is synthesized from empirical solar irradiance $G_t$ ($\text{W/m}^2$) collected from the EU Joint Research Centre PVGIS database for Sceaux, France:

   $$P_{\text{pv}, t} = G_t \cdot A_{\text{pv}} \cdot \eta_{\text{pv}} \cdot PR \cdot 10^{-3}$$

   with an effective aperture area $A_{\text{pv}} = 1{,}471\,\text{m}^2$, nominal module conversion efficiency $\eta_{\text{pv}} = 0.20$, and balance-of-system performance ratio $PR = 0.85$, yielding a peak generation rating of $250\,\text{kW}$.

4. **Heterogeneous Demand Sectors & Discontinuous Industrial Motor Rule:**
   The aggregate facility electrical load $P_{\text{load}, t}$ is partitioned across five distinct demand sectors:
   - Residential: 35% of total base demand.
   - Commercial: 25% of total base demand.
   - Industrial: 25% of total base demand.
   - Stadium: 10% of total base demand (with intermittent evening match events).
   - Street Lighting: 5% of total base demand (subject to astronomical solar zenith cycles).
   
   Crucially, the industrial zone features heavy inductive motor and machining loads subject to non-linear physical operating limits: continuous operational throttling between 0% and 70% is physically disallowed due to minimum motor cooling and torque stability thresholds. The industrial load throughput fraction $u_{\text{ind}, t} = 1 - s_{5, t}$ must satisfy:

   $$u_{\text{ind}, t} \in \{0\} \cup [0.70, 1.00]$$

   Consequently, the microgrid controller must either maintain at least 70% baseline industrial throughput or execute a full sectoral process shutdown.

### Multi-Component Economic Objective & The 50:1 Cost Asymmetry

The microgrid operational cost $c_t$ at step $t$ is composed of energy import acquisition, battery cell wear degradation, and customer curtailment penalties:

$$c_t = \underbrace{p_t \cdot P_{\text{grid}, t} \cdot \Delta t}_{\text{Grid Purchase Cost}} + \underbrace{c_w \cdot |P_{\text{bat}, t}| \cdot \Delta t}_{\text{Battery Wear Degradation}} + \underbrace{c_u \cdot \sum_{z=1}^5 s_{z, t} P_{\text{load}, z, t} \cdot \Delta t}_{\text{Unserved Load Penalty}}$$

where:
- $p_t$ is the day-ahead spot electricity price (EUR/kWh) obtained from Open Power System Data (OPSD) under the `proxy_DE` configuration (`configs/config.yaml`): French national load scaled to a 500 kW microgrid peak, Sceaux PVGIS solar irradiance and Open-Meteo weather, and an `AT` $\to$ `DE-LU` day-ahead spot price splice at `2018-09-30 22:00 UTC` across the 2017–2019 bidding-zone split.
- $c_w = 0.02\text{ EUR/kWh}$ (€20/MWh) denotes the marginal battery wear degradation cost per unit of energy throughput.
- $c_u = 1.00\text{ EUR/kWh}$ (€1,000/MWh) represents the regulatory Value of Lost Load (VoLL) curtailment penalty applied uniformly across unserved demand.

The policy maximizes the discounted economic return $R(s_t, a_t) = -c_t$:

$$\max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right] \quad \text{s.t.} \quad \mathbb{E}_{\tau \sim \pi} \left[ \frac{1}{T} \sum_{t=0}^{T-1} \max(0, P_{\text{grid}, t} - P_{\text{cap}}) \right] \le d$$

The economic coefficients establish an acute 50:1 cost trade-off: curtailing 1 kWh of electrical load costs €1.00, whereas cycling 1 kWh through the battery incurs only €0.02 in wear degradation. Any cost-minimizing control policy must prioritize battery discharge during demand surges before shedding customer load, yielding a net economic benefit of €0.98 per kWh of load preserved.

### Safety Constraint

The microgrid operates under a contractual grid-import capacity limit:

$$P_{\text{cap}} = 368.15\,\text{kW}$$

This threshold is fixed at the 90th percentile (P90) of unmanaged net load ($P_{\text{net}} = \max(0, P_{\text{load}} - P_{\text{pv}})$) evaluated across the 104-week training split. The constraint cost is defined as the hourly grid-import exceedance:

$$C(s_t, a_t) = \max(0, P_{\text{grid}, t} - P_{\text{cap}})$$

The threshold $d = 0.05\,\text{kW/step}$ serves as the soft $G_1$ gate in CMDP formulations, while physical system reliability requires strict feasibility (0 violation hours, 0.00 kW exceedance).

### Environment Families Introduced Upfront

To systematically isolate the physical and algorithmic drivers of control failure, three experimental environment families are benchmarked:

1. `unclamped`:
   The unconstrained legacy physical formulation where battery charging commands are executed without reference to the grid-import cap. Under this formulation, battery charging can independently push total site demand over $P_{\text{cap}}$, inducing charging-driven capacity violations.
2. `clamped`:
   The reference physical formulation implementing an automated physical operator interlock. The environment dynamically restricts battery charging power to the instantaneous available contractual headroom:

   $$
   P_{\text{ch}, t}^{\text{allowed}} = \min\left( P_{\text{ch}, t}^{\text{req}}, \max\left(0, P_{\text{cap}} - P_{\text{net-served}, t}\right) \right)
   $$

   This interlock guarantees that battery charging can never induce a grid-cap violation by construction, isolating load shedding as the sole mechanism for resolving capacity emergencies.
3. `clamped_mb`:
   Combines the clamped physical environment with a minibatch policy gradient optimizer (`batch_size=64`, 5 value epochs per rollout, raw unshielded action storage) on the base 17-dimensional state space. Introduced to resolve temporal credit assignment failures and escape stationary policy equilibria.

![Microgrid CMDP Architecture & 5-Zone Demand Physics](results/figures/presentation/09_microgrid_architecture.png)

---
## Previous Project Approaches & Hardcoded Baselines

### Prior Work (Salaorni et al., IJCNN 2025 on ErNESTO-DT)

Previous investigations in RL-based microgrid energy management (Salaorni et al., IJCNN 2025; Journal of Energy Storage 2025) formulated the control task using the ErNESTO-DT simulation framework. However, that formulation differs fundamentally from safety-critical industrial settings:
- **Absence of Capacity Constraints:** Prior work optimized unconstrained profit without a contractual grid-import cap ($P_{\text{cap}}$).
- **Soft Battery Degradation Penalties:** Battery degradation was addressed solely via linear penalty terms added to the scalar reward function.
- **Untuned Comparators:** RL agents were compared against untuned rule-based heuristics that lacked peak-shaving coordination.
- **No Multi-Zone Shedding:** The action space was restricted to battery power dispatch, omitting load curtailment mechanisms.

In real-world power distribution, soft penalties fail to enforce grid-import boundaries. Because electricity prices and battery wear are orders of magnitude smaller than infrastructure capacity violation penalties, an RL agent optimizing a soft objective treats constraint compliance as an economic trade-off. During rare, high-magnitude net load spikes, unconstrained or soft-penalized agents reliably choose to violate physical import limits.

### Hardcoded & Classical Baselines (Pros, Cons, and Why They Fail)

To contextualize the performance of reinforcement learning, five classical baselines were implemented in [`agents/heuristics.py`](agents/heuristics.py) and [`tools/report/lp_oracle.py`](tools/report/lp_oracle.py):

| Baseline Controller | Total Cost (€, Val) | Grid Purchase (€) | Battery Wear (€) | Unserved Penalty (€) | Throughput (MWh) | Mean Exceedance (kW) | Violation Hours | Strict Feasible? |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **LP' Theoretical Oracle** | *53,290.51* | *44,192.97* | *436.73* | *8,660.86* | *21.84* | *0.0000* | *0h* | Theoretical Limit |
| **Peak-Shave-Minimum (PSM')** | *55,530.22* | *44,658.60* | *246.95* | *10,624.67* | *12.35* | *0.0000* | *0h* | Yes |
| **Shield + PSM'** | *55,500.35* | *44,660.23* | *246.95* | *10,593.17* | *12.35* | *0.0000* | *0h* | Yes |
| **Shield + Idle (Zero-Battery)** | *58,644.70* | *44,417.53* | *0.00* | *14,227.17* | *0.00* | *0.0000* | *0h* | Yes |
| **Grid-Only (Unmanaged)** | *45,337.92* | *45,337.92* | *0.00* | *0.00* | *0.00* | *3.2571* | *468.0h* | No |

#### 1. Grid-Only (Unmanaged Baseline)
- **Description:** Battery remains completely idle ($P_{\text{bat}} = 0$), and zero load shedding is executed ($P_{\text{shed}} = 0$).
- **Pros:** Incurs the lowest nominal energy acquisition cost (€45,337.92 on Val, €36,562.65 on Test) with zero wear and zero customer curtailment.
- **Cons & Failure Mode:** Fails safety completely. Exceeds $P_{\text{cap}}$ in 468.0 hours on the Validation split (10.71% of the time), generating a mean exceedance of 3.2571 kW and net load peaking at 456.87 kW (88.72 kW over the contractual limit). It is operationally unacceptable.

#### 2. Peak-Shave-Minimum (PSM' Domain Heuristic)
- **Description:** A domain-engineered priority heuristic that charges the battery during low-tariff off-peak periods, discharges aggressively whenever net facility demand exceeds $P_{\text{cap}}$, and curtails lowest-priority demand sectors when storage is exhausted.
- **Pros:** Maintains strict zero violations (0h violations, 0.0000 kW exceedance across both splits) while providing a competitive economic baseline (€55,530.22 on Val, €37,674.27 on Test, 12.35 MWh throughput).
- **Cons & Failure Mode:** Governed by static threshold rules. It cannot execute dynamic price arbitrage, adapt to fluctuating spot tariffs, or perform predictive multi-hour storage management.

#### 3. Shield + Idle (Zero-Battery Safety Baseline)
- **Description:** Fixes battery dispatch to zero ($a_{\text{bat}} = 0$) and passes demand through an analytical post-hoc safety shield ([`env/cap_shield.py`](env/cap_shield.py)), shedding load whenever unmanaged demand exceeds $P_{\text{cap}}$.
- **Pros:** Guarantees strict feasibility (0h violations, 0.0000 kW exceedance).
- **Cons & Failure Mode:** Zero storage utilization (0.00 MWh throughput) results in severe unserved load curtailment penalties (€14,227.17 on Val, €58,644.70 total cost). This confirms empirically that deploying a safety shield without active energy storage dispatch is economically non-viable.

#### 4. Shield + PSM'
- **Description:** Evaluates the tuned PSM' heuristic wrapped with the runtime safety shield.
- **Outcome:** Yields €55,500.35 on Val (12.35 MWh throughput, 0h violations), confirming that safety shielding does not distort a well-tuned heuristic policy.

#### 5. LP' Theoretical Oracle (Hindsight Linear Programming) & Exact MILP'
- **Description:** Formulates the exact discrete microgrid dynamics as a global linear program over each 168-hour weekly horizon, solved with perfect hindsight using the HiGHS solver via [`tools/report/lp_oracle.py`](tools/report/lp_oracle.py).
- **Pros:** Establishes the absolute theoretical lower bound on attainable operating costs (€53,290.51 on Val: €44,192.97 grid purchases, €436.73 battery wear, €8,660.86 unserved load penalties; €36,991.67 on Test), achieving 21.84 MWh (21.837 MWh) throughput and 37.98% active SoC while guaranteeing 0h violations. Additionally, [`tools/report/lp_oracle.py::solve_week_milp`](tools/report/lp_oracle.py) provides the exact Mixed-Integer Linear Programming formulation (`MILP'`) that explicitly models the non-convex binary motor shutdown constraint ($u_{\text{ind}} \in \{0\} \cup [0.70, 1.00]$).
- **Cons & Failure Mode:** Non-causal. It requires 168 hours of perfect, deterministic future knowledge of solar insolation, sector loads, and spot prices, rendering it physically unrealizable in online operations.

---

## Why Reinforcement Learning

Reinforcement learning provides two distinct advantages over traditional optimization frameworks for microgrid energy management:

1. **Sub-Second Real-Time Causal Execution:**
   Online Mixed-Integer Linear Programming (MILP) or non-linear Model Predictive Control (MPC) solvers require iterative numerical optimization at each time step. Solving high-dimensional receding-horizon formulations with mixed-integer constraints under short dispatch intervals is computationally intensive and vulnerable to solver timeouts. In contrast, an evaluated RL policy network executes in $< 1\text{ ms}$ on commodity CPU hardware, enabling sub-second real-time control without requiring future forecast knowledge.

2. **Non-Linear Coordination Under Uncertainty:**
   Deep neural networks can learn non-linear policies that simultaneously coordinate dynamic time-of-use tariff arbitrage, battery state-of-charge tracking, and multi-zone fractional shedding under non-convex operational constraints (such as the disjoint industrial motor constraint $u_{\text{ind}} \in \{0\} \cup [0.70, 1.00]$). RL maps raw environmental observations directly to multi-asset control actions without requiring linearization heuristics.

---

## Why Each Safe RL Architecture Was Selected

To investigate constraint enforcement in continuous action spaces ($\mathcal{A} \subset \mathbb{R}^6$), five distinct safe RL paradigms were selected:

1. **Unconstrained PPO (Baseline Formulation):**
   Standard Proximal Policy Optimization (Schulman et al., 2017) maximizing discounted economic returns via Generalized Advantage Estimation (GAE; Schulman et al., 2016). Selected to establish the baseline economic performance of RL and test whether high economic penalties (the €1.00/kWh VoLL) implicitly enforce physical capacity limits.

2. **Dual-Lagrangian PPO (PPO-Lagrangian):**
   The canonical formulation for Constrained MDPs (Tessler et al., 2019; Spoor et al., 2025). It defines a saddle-point Lagrangian objective $\mathcal{L}(\theta, \lambda) = J(\pi_\theta) - \lambda (\bar{C}(\pi_\theta) - d)$, where the non-negative dual multiplier $\lambda \ge 0$ is dynamically adapted via subgradient ascent:

   $$
   \lambda_{k+1} = \max\left(0, \lambda_k + \alpha_\lambda (\bar{C}_k - d)\right)
   $$

   Selected to evaluate whether adaptive constraint weighting can reliably balance cost minimization against capacity exceedance.

3. **Residual Policy Learning (PPO-Residual):**
   Following Silver et al. (2018), a parameterized policy network $\pi_\theta(s)$ is superimposed onto the tuned domain heuristic $\pi_{\text{base}}(s) = \text{PSM'}(s)$:

   $$
   a_t = \pi_{\text{base}}(s_t) + \alpha_{\text{res}} \cdot \pi_\theta(s_t)
   $$

   Selected to test whether RL can refine a safe, high-performing heuristic to extract dynamic price arbitrage without destabilizing safety.

4. **Post-Hoc Environmental Safety Shield (Shielded PPO):**
   An external runtime verification wrapper ([`env/cap_shield.py`](env/cap_shield.py)) placed at the environment boundary. If an action proposed by the policy network violates $P_{\text{cap}}$, the shield deterministically clamps battery charging and sheds minimal necessary load according to priority rankings. Selected to enforce a zero-violation safety guarantee at execution time while testing whether external shielding distorts policy gradient updates.

5. **In-the-Loop Safe RL (Differentiable Safety Layer & Auxiliary Shield Loss):**
   Following Dalal et al. (2018), an analytical closed-form Euclidean projection layer ([`agents/safety_layer.py`](agents/safety_layer.py)) is embedded directly inside the policy computational graph:

   $$\tilde{a}_t = \arg\min_{\tilde{a} \in \mathcal{A}_{\text{safe}}(s_t)} \frac{1}{2} \|\tilde{a} - a_t\|_2^2 \quad \text{s.t.} \quad P_{\text{net}}(s_t) + P_{\text{bat}}^{\text{max}} \cdot \tilde{a}_{\text{bat}} \le P_{\text{cap}}$$

   - **Historical Snapshot (`v0-prototype`):** The committed benchmark runs used a 1-D analytical battery projection:

     $$\tilde{a}_{\text{bat}} = \operatorname{clip}\left( a_{\text{bat}} - \frac{\operatorname{ReLU}(P_{\text{net}} + P_{\text{bat}}^{\text{max}} a_{\text{bat}} - P_{\text{cap}})}{P_{\text{bat}}^{\text{max}}}, -1.0, 1.0 \right)$$

     While computationally efficient, this formulation exhibited an active-manifold gradient pathology: when the constraint bound ($g > 0$), the analytical derivative $\frac{\partial \tilde{a}_{\text{bat}}}{\partial a_{\text{bat}}} = 1 - 1 = 0$ vanished, starving the policy of directional gradient feedback during acute overruns. It relied on an outer `CapShield` wrapper for multi-zone demand curtailment and battery SoC boundary enforcement. An auxiliary loss variant (`PPO-Aux-Shield`) penalizing policy divergence from safe actions ($\mathcal{L}_{\text{aux}} = \frac{1}{2} \|a - \tilde{a}\|_2^2$) was also evaluated.
   - **Post-Snapshot T3 Upgrades:** In task T3, `agents/safety_layer.py` was structurally enhanced with:
     a) **Straight-Through Estimator (STE) Active Gradient:** Applies a leaky STE surrogate derivative (Bengio et al., 2013; $\frac{\partial \tilde{a}}{\partial a} = \alpha_{\text{ste}} = 0.1$) on active constraints, restoring non-vanishing gradient flow back to the actor during peak demand surges.
     b) **Hard SoC Envelopes:** Analytical projection boundaries directly enforcing $\text{SoC} \in [\text{SoC}_{\text{min}}, \text{SoC}_{\text{max}}] = [0.10, 0.90]$.
     c) **Joint 6-D KKT QP Projection:** Formulates the full 6-dimensional Quadratic Program over battery dispatch and 5-zone load shedding with exact KKT dual multipliers and full-rank Jacobian backpropagation.
     d) **Distribution Consistency:** `SafePolicyNetwork` evaluates action log-probabilities using raw outputs (`forward_raw`), eliminating double-projection distortions.
   - *Provenance Note:* All 80 post-fix Regime 1 & Regime 2 runs (`clamped_*`, `clamped_mb_*`, `clamped_ppo_diff_shield`, `clamped_ppo_shield_aux`, `causal23_*`, `ci_*`) were executed and evaluated on Validation (`results/tables/per_seed_metrics.csv`, `runs.csv`, `regime_benchmarks_val.csv`). Only the legacy unclamped `s3_*` runs (20 runs) and the locked one-shot Test tables (`results/tables/test_evaluation.csv`, `test_stress_weeks.csv`, `results/runs/test_lock.json`) retain the historical `v0-prototype` snapshot.

![Safety Mechanisms & Gradient Flow Comparison — Post-Hoc Shield vs. T3 STE Co-Design](results/figures/presentation/11_safety_mechanisms.png)

6. **Post-Snapshot Codebase & Benchmark Upgrades (Tasks T1–T13):**
   Beyond the T3 safety layer overhaul, tasks T1–T13 introduced structural upgrades and new experimental regimes across the repository:
   - **T1 (`env/cap_shield.py`):** Inverted relu/deadzone shed mappings, re-clamped positive battery charging to 0 on any shedding event to block post-blackout recharging, and safely guarded the 0.90 SoC ceiling against unintended trips.
   - **T2 (`agents/ppo.py`, `experiments/train_core.py`):** Reused denormalized value and cost-value estimates for bootstrap returns, and properly zeroed GAE advantages across episode truncations.
   - **T5 (`tools/report/lp_oracle.py`, `load_results.py`):** Added exact Mixed-Integer Linear Programming (`solve_week_milp`, `MILP'`) enforcing the non-convex binary motor shutdown constraint, and centralized the canonical 1.05x G2 gate helper (58,306.73 EUR).
   - **T6 & T9–T10 (`env/smartgrid_env.py`, `agents/mpc.py`, `tools/report/lp_oracle.py`):** Added Regime 1 asymmetric causal forecasts (`oracle` / `causal_noisy` / `persistence` with exact D-1 day-ahead prices), Regime 2 C&I tariffs (telescoping weekly peak demand charge, heterogeneous zone VoLL, industrial trip switching penalty), and full Regime 2 parity in `MPCPolicy` and `LP'`/`MILP'`.
   - **T7 & T11–T13 (`configs/config.yaml`, `tools/report/evaluate_test_split.py`, `eval_regimes_val.py`):** Added $\text{CVaR}_{0.95}$ tail-risk metrics, per-step latency tracking, 4 OOD contingencies, 80 post-fix 300k-step RL runs (`T12`), and unified Regime 1 / Regime 2 / 2019-W04 OOD Validation benchmarking (`T13`).

---

## Data Splits & Experimental Methodology

### Mandatory Chronological Splitting

In reinforcement learning for physical infrastructure, random splitting across time steps destroys the temporal Markovian continuity of physical storage states ($\text{SoC}_{t+1} = \text{SoC}_t + \dots$), thermal building inertia, and multi-day meteorological weather patterns. Shuffling time steps allows future weather and price sequences to leak into the training distribution. Consequently, the dataset is partitioned strictly along chronological boundaries:
- **Training Split:** 2017–2018 (104 continuous weeks = 17,472 operating hours).
- **Validation Split:** H1 2019 (Weeks 02–27, 26 continuous weeks = 4,368 operating hours).
- **Test Split:** H2 2019 (Weeks 29–52, 24 continuous weeks = 4,032 operating hours).

The Test split was kept strictly sealed throughout all model development and hyperparameter tuning, unlocked exactly once via [`tools/report/evaluate_test_split.py`](tools/report/evaluate_test_split.py) for final evaluation.

### Seasonal Distribution Shift Between Validation and Test Splits

A documented distribution shift exists between the Validation and Test splits:
1. **Validation Split (H1 2019):** Winter-heavy period characterized by high resistive and heat-pump space heating demand and low solar insolation. Under unmanaged Grid-Only dispatch, grid import exceeds $P_{\text{cap}}$ in **10.71% of hours (468 violation hours)**, with net load peaking at **456.87 kW (88.72 kW above the cap)**. Peak shaving is required frequently, forcing PSM' to incur €10,624.67 in customer load shedding penalties.
2. **Test Split (H2 2019):** Mild late-summer and autumn period. Unmanaged Grid-Only dispatch exceeds $P_{\text{cap}}$ in only **3.79% of hours (153 violation hours)**, with net load peaking at **426.32 kW (58.17 kW above the cap)**. Because unmanaged exceedances are rare, PSM' incurs only €868.00 in load curtailment penalties over the entire 24 weeks. This seasonal shift naturally compresses absolute cost differentials between all controllers on the test set.

![154-Week Chronological Dataset Splits (Train / Val / Test)](results/figures/presentation/10_dataset_split_regimes.png)

### Week 49 (2019-W49) as the Acute Peak-Stress Bottleneck

Standard meteorological stress criteria—lowest solar irradiance (W48), highest electricity price (W45), highest temperature (W30), and lowest barometric pressure (W51)—fail to bind the microgrid capacity constraint. On each of these four weather weeks, total net site demand remains strictly below 368.15 kW, meaning even the unmanaged Grid-Only baseline incurs exactly 0.00 kW exceedance and 0 violation hours.

Isolating true capacity stress requires evaluating controllers on the **cap-binding peak-exceedance week: Week 49 (2019-W49, December 2–8, 2019)** from the locked `v0-prototype` Test snapshot (`results/tables/test_stress_weeks.csv`):
- **Net Demand Surge:** Peak site demand reaches **426.32 kW**, producing an acute **58.17 kW overrun** above $P_{\text{cap}}$ across 66 operating hours.
- **Controller Performance Breakdown on W49 (168 Operating Hours):**

| Controller Architecture | W49 Total Cost (€) | Mean Exceedance (kW) | Violation Hours | Active Cycling |
|---|---:|---:|:---:|:---:|
| **Grid-Only (Unmanaged)** | 2,149.71 | 10.25 | 66.0h (39.3%) | None |
| **PPO-Lagrangian (Clamped)** | 2,158.79 ± 6.74 | 10.25 | 66.0h (39.3%) | Ineffective |
| **PPO-Residual (Clamped)** | 2,864.69 ± 129.35 | 0.83 | 60.8h (36.2%) | Under-sheds Load |
| **Shield + Idle (Zero-Battery)** | 3,785.40 | **0.00** | **0.0h (0.0%)** | Passive Parking |
| **Full-Batch Shielded PPO** | 3,797.08 ± 0.01 | **0.00** | **0.0h (0.0%)** | Defensive Shedding |
| **Minibatch Shielded PPO** | 3,599.47 ± 140.56 | **0.00** | **0.0h (0.0%)** | Active Peak Shaving |
| **Peak-Shave-Minimum (PSM')** | **2,990.43** | **0.00** | **0.0h (0.0%)** | Heuristic Benchmark |
| **Shield + PSM'** | **2,985.58** | **0.00** | **0.0h (0.0%)** | Heuristic Benchmark |

On the cap-binding stress week, Minibatch Shielded PPO actively deploys stored battery energy to shave demand peaks, outperforming Full-Batch Shielded PPO by **€197.61**, whereas unconstrained and Lagrangian controllers fail completely.

---

## Multi-Angle Empirical Results & Comparative Evaluation

To systematically evaluate continuous-action microgrid energy management under contractual grid-import boundaries ($P_{\text{cap}} = 368.15\text{ kW}$), the empirical evaluation is structured into two sequential stages:
1. **Phase 1: PPO Environment & Optimizer Ablation (Clamping, Batching):** Isolating the physical charging interlock and policy gradient minibatching to identify the best-performing unconstrained PPO baseline.
2. **Phase 2: Comprehensive Multi-Angle Safe RL Benchmark (Best PPO vs. Safe Architectures & Baselines):** Benchmarking the representative PPO baseline against tuned heuristics, linear programming hindsight oracles, and the spectrum of constrained and shielded safe RL paradigms across three core analytical angles: **Economical Performance**, **Battery Dynamics & Asset Utilization**, and **Reliability & Safety Enforcement**.

---

### Phase 1: PPO Optimizer & Environment Ablation (Batching, Clamping)

We evaluate three systematic PPO formulations across 5 independent random seeds on the 26-week Validation Split (H1 2019, 4,368 operating hours), varying batch size first, then clamping:
- **`clamped_ppo` (`PPO [Full-Batch]`):** Standard 168-hour full-batch trajectory rollouts ($B=2048$, 10 epochs) with the automated physical charging interlock.
- **`unclamped_ppo` / `s3_ppo` (`PPO [Mini-Batch]`):** Unclamped baseline formulation without charging constraints.
- **`clamped_mb_ppo` (`PPO [Clamped Minibatch]`):** Combines the clamped physical environment with minibatch policy optimization (`batch_size=64`, 5 value epochs per rollout) on the base 17-dimensional state space.

#### Phase 1 Empirical Comparison Table (Validation Split, Mean ± SD across 5 Seeds)

| PPO Formulation | Total Cost (€) | Grid Purchases (€) | Battery Wear (€) | Unserved Penalty (€) | Throughput (MWh) | Active SoC (%) | Mean Exceedance (kW) | Violation Hours | Violation Rate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`PPO [Full-Batch]` (`clamped_ppo`, T12)** | 45,415.58 ± 177 | 45,312.51 | 103.07 | 0.00 | 5.15 | 3.8% | 3.2420 | 465.8h | 10.66% |
| **`PPO [Unclamped]` (`s3_ppo`, `v0-prototype`)** | 45,513.38 ± 192 | 45,406.63 | 106.75 | 0.00 | 5.34 | 7.0% | 3.3497 | 475.4h | 10.88% |
| **`PPO [Clamped Minibatch]` (`clamped_mb_ppo`, ★ Best PPO, T12)** | **47,354.23 ± 378** | 45,308.74 | **506.46** | **1,539.02** | **25.32** | **16.8%** | **3.0970** | **450.4h** | **10.31%** |

![Ablation & Safe RL Pathologies — Cost, Cap Violations, and Battery Activity](results/figures/presentation/01_ablation_pathologies.png)

#### Why `clamped_mb_ppo` is the Best-Performing Unconstrained PPO Baseline

1. **Escaping the Passive "Battery Parking" Trap:** In full-batch updates (`clamped_ppo` and `s3_ppo`), sparse reward signals averaged over 168-hour weekly trajectories dilute gradient feedback. The policy network collapses into a near-zero dispatch saddle point, cycling storage for only 5.15–5.34 MWh over the entire semester (3.8% to 7.0% active SoC) and executing zero load shedding. Switching to minibatch optimization (`clamped_mb_ppo`) provides frequent, localized stochastic gradient steps, increasing battery throughput nearly fivefold to **25.32 MWh** (16.8% active cycling) and actively deploying stored energy to shave net demand peaks.
2. **Representative Baseline Selection:** Consequently, `clamped_mb_ppo` represents the optimal unconstrained learning policy, escaping algorithmic stagnation while maintaining true unconstrained behavior, and serves as the benchmark comparator in Phase 2.
---

### Phase 2: Comprehensive Multi-Angle Safe RL Benchmark (Best PPO vs. Safe Architectures & Baselines)

In Phase 2, the best unconstrained PPO baseline (`PPO [Clamped Minibatch]`) is compared against five reference baselines and six constrained Safe RL architectures across the 26-week Validation Split (H1 2019, 4,368 operating hours, 5 independent random seeds per learning method):
- **Reference Baselines:** `Grid-Only` (unmanaged connection), `PSM'` (tuned priority heuristic), `LP' Theoretical Oracle` (HiGHS hindsight LP), `Shield + Idle` (zero-battery shield baseline), and `Shield + PSM'` (heuristic + shield).
- **Unconstrained Representative:** `PPO [Clamped Minibatch]`.
- **Safe RL Architectures:** `PPO-Lagrangian [Clamped MB]`, `PPO-Residual [Clamped Full-Batch]`, `Shielded PPO [Clamped Full-Batch]`, `Shielded PPO [Clamped Minibatch]`, `PPO-Aux-Shield` (auxiliary shield loss), and `PPO-Diff-Shield` (differentiable safety layer).

#### 1. Economical Angle: Total Expenditure & Cost Composition (Regime 1 Val, Post-Fix T12 Runs)

The economic objective evaluates total operational expenditure alongside its three physical constituents: energy purchase from the transmission grid, cell wear degradation (€20/MWh), and customer curtailment penalties (VoLL = €1,000/MWh). Table values match `results/tables/per_seed_metrics.csv` and `results/tables/frozen_baselines.csv` (`results/tables/project_constants.csv` sets `g2_cost_multiplier = 1.05`, giving `g2_cost_eur = 58,306.73 EUR`, alongside `G2_strict_pass` at `55,530.22 EUR` in `per_seed_metrics.csv`):

| Controller Architecture | Total Cost (€) | Grid Energy Purchase (€) | Battery Degradation Wear (€) | Unserved Load Penalty (€) | Economic Feasibility ($G_2 \le 58,307$ €) |
|---|---:|---:|---:|---:|:---:|
| **LP' / MILP' Theoretical Oracle** | *53,290.51* | *44,192.97* | *436.73* | *8,660.86* | Theoretical Bound |
| **Peak-Shave-Minimum (PSM')** | *55,530.22* | *44,658.60* | *246.95* | *10,624.67* | **Passed** |
| **Shield + PSM'** | *55,500.35* | *44,660.23* | *246.95* | *10,593.17* | **Passed** |
| **Grid-Only (Unmanaged Baseline)** | *45,337.92* | *45,337.92* | *0.00* | *0.00* | Invalid (Safety Violations) |
| **PPO [Clamped Minibatch] (`clamped_mb_ppo`)** | 47,354.23 ± 378 | 45,308.74 | 506.46 | 1,539.02 | Invalid (Safety Violations) |
| **PPO-Residual [Clamped FB] (`clamped_ppo_residual`)** | 46,167.91 ± 150 | 45,429.36 | 168.44 | 570.12 | Invalid (Safety Violations) |
| **PPO-Lagrangian [Clamped MB] (`clamped_mb_ppo_lagrangian`)** | 51,618.03 ± 576 | 45,012.20 | 709.35 | 5,896.48 | Invalid (Safety Violations) |
| **PPO-Residual [Clamped MB] (`clamped_mb_ppo_residual`)** | 52,615.24 ± 2,174 | 44,978.38 | 372.28 | 7,264.57 | Invalid (Safety Violations) |
| **PPO-Diff-Shield (`clamped_ppo_diff_shield`, `best` ckpt)** | **55,402.89 ± 1,212** | 44,609.72 | 1,265.17 | **9,528.01** | **Passed (Mean < `PSM'` + 5/5 Seeds)** |
| **PPO-Diff-Shield (`clamped_ppo_diff_shield`, `final` ckpt)** | **56,696.94 ± 1,279** | 44,668.49 | 2,023.20 | **10,005.25** | **Passed (Mean + 4/5 Seeds; Seed 101 < `PSM'`)** |
| **Shield + Idle (Zero-Battery Control)** | *58,644.70* | *44,417.53* | *0.00* | *14,227.17* | Exceeds Gate by 0.6% |
| **Shielded PPO [Clamped FB] (`clamped_ppo_shielded`)** | 58,885.36 ± 45 | 44,571.36 | 121.04 | 14,192.95 | Exceeds Gate by 1.0% |
| **PPO-Aux-Shield (`clamped_ppo_shield_aux`)** | 58,914.89 ± 920 | 44,542.36 | 1,442.52 | 12,930.01 | Exceeds Gate by 1.0% (1/5 Pass) |
| **Shielded PPO [Clamped MB] (`clamped_mb_ppo_shielded`)** | 58,987.41 ± 711 | 44,544.10 | 1,160.01 | 13,283.30 | Exceeds Gate by 1.2% |

![Five-Seed Cost Stability & MPC Forecast Sensitivity](results/figures/presentation/12_seed_stability_tradeoff.png)

![Validation Learning Dynamics — Cost, Violations, and Active SoC (50k–300k Steps)](results/figures/presentation/03_learning_dynamics.png)

**Key Economic Findings (Direct Empirical Proof of T1–T3 Fixes):**
- **T2 + T3 Fixes Unlock G2 Gate Compliance:** In `v0-prototype` (before fixing `SafePolicyNetwork.distribution()` to propagate STE gradients during updates and PPO truncation/bootstrap handling), `clamped_ppo_diff_shield` averaged **€58,473.79 ± €1,180.66** on Validation (failing the canonical $1.05 \times \text{PSM'}$ G2 gate of €58,306.73 on mean, with only Seed 101 passing at €56,411.62). After retraining under the fixed T1–T3 codebase, `clamped_ppo_diff_shield` improves to **€56,696.94 ± €1,279.45 (`final`)** and **€55,402.89 ± €1,212.20 (`best`)**, comfortably passing the G2 gate (€58,306.73) on mean (`final` and `best`), 4/5 `final` seeds (€55,045.60, €55,968.83, €56,649.27, €57,496.22; Seed 104 is €58,324.76 `final` / €54,972.21 `best`), and 5/5 `best` seeds. Moreover, the 5-seed `best` means of both `clamped_ppo_diff_shield` (**€55,402.89**) and `causal23_ppo_diff_shield` (**€55,268.23**) as well as `final` Seed 101 (**€55,045.60**) outperform even the strict 1.00× `PSM'` heuristic (**€55,530.22**).
- **The 50:1 Asymmetry Dictates Cost:** `clamped_ppo_diff_shield` cuts unserved load penalties from €12,336.49 (`v0-prototype`) down to **€10,005.25 (`final`) / €9,528.01 (`best`)**, beating `PSM'` (€10,624.67) and saving €2,924.76–€4,221.94 compared to post-hoc `Shielded PPO` variants.
- **Lagrangian & Aux-Shield Stabilization:** Fixing the Lagrangian dual cadence to once per rollout eliminates `clamped_mb_ppo_lagrangian`'s explosive multiplier overshoot ($\lambda_{\text{end}} = 0.314$ vs. `2.354`), reducing cost from €64,380.18 to **€51,618.03 ± €575.51** (€5,896.48 unserved) while still violating grid caps (`2.51 kW` / `409.8h`) at `lr_dual = 0.01`. Masking `PPO-Aux-Shield`'s auxiliary imitation target by feasible `SoC` discharge lowers its cost from €59,481.95 to **€58,914.89 ± €920.29**.

#### 2. Battery Dynamics & Asset Utilization Angle

Storage asset utilization quantifies total throughput dispatched, active cycling time within operational bounds ($\text{SoC} \in [0.10, 0.90]$), and resultant cell wear costs:

| Controller Architecture | Battery Energy Throughput (MWh) | Active Unpinned SoC Time (%) | Battery Wear Degradation (€) | Operating Regime / Asset Diagnosis |
|---|---:|---:|---:|---|
| **LP' Theoretical Oracle** (`frozen_baselines.csv` / `regime_benchmarks_val.csv`) | 21.84 | 37.98% | 436.73 | Global Multi-Day Arbitrage Bound |
| **Peak-Shave-Minimum (PSM')** | 12.35 | 11.20% | 246.95 | Targeted Peak Shaving |
| **Shield + PSM'** | 12.35 | 11.20% | 246.95 | Targeted Peak Shaving |
| **Grid-Only (Unmanaged Baseline)** | 0.00 | 100.0% (static 50% SoC) | 0.00 | Complete Inaction |
| **Shield + Idle (Zero-Battery Control)**| 0.00 | 100.0% (static 50% SoC) | 0.00 | Zero Battery Utilization |
| **PPO [Clamped Minibatch] (`clamped_mb_ppo`)** | 25.32 | 16.8% | 506.46 | Active Unconstrained Arbitrage |
| **Shielded PPO [Clamped Full-Batch]** | 6.05 | 4.0% | 121.04 | **"Battery Parking" Moral Hazard** |
| **PPO-Residual [Clamped Full-Batch]** | 8.42 | 12.6% | 168.44 | Uncoordinated Cap Leakage |
| **PPO-Diff-Shield (`final` / `best`)** | **101.16 / 63.26** | **50.5% / 40.7%** | **2,023.20 / 1,265.17** | **Optimal Coordinated Safe Dispatch** |
| **PPO-Aux-Shield (Aux. Shield Loss)** | 72.13 | 35.4% | 1,442.52 | SoC-Masked Aux Learning |
| **PPO-Lagrangian [Clamped Minibatch]** | 35.47 | 27.7% | 709.35 | Under-Penalized Dual Drift |
| **Shielded PPO [Clamped Minibatch]** | 58.00 | 29.2% | 1,160.01 | Active Cycling with Higher Unserved Cost |

![2019-W04 Dispatch Case Study — Full Cycling vs. Battery Parking](results/figures/presentation/07_dispatch_case_2019w04.png)

**Key Asset Utilization Findings:**
- **The "Battery Parking" Pathology:** Under full-batch policy gradients (`clamped_ppo_shielded`), downstream safety shielding absorbs constraint enforcement. The actor network learns that idling the battery saves wear costs (€0.02/kWh) while the shield curtails load, parking the battery at its ceiling for 96.0% of the time (4.0% active SoC, 6.05 MWh throughput) and incurring €14,192.95 in unserved penalties.
- **Targeted Differentiable Dispatch:** Post-fix `clamped_ppo_diff_shield` sustains active storage cycling (`101.16 MWh` / `50.5%` active SoC `final`, `63.26 MWh` / `40.7%` active SoC `best`) to eliminate idle parking and cut unserved penalties from €12,336.49 (`v0-prototype`) down to **€10,005.25 (`final`) / €9,528.01 (`best`)** (€2,331.24–€2,808.48 lower curtailment cost).

#### 3. Reliability & Safety Angle: Feasibility & Constraint Enforcement

Safety evaluates compliance with the contractual import threshold ($P_{\text{cap}} = 368.15\text{ kW}$), comparing mean exceedance, violation hours, and curtailment rates on post-fix Validation (H1 2019, 4,368h) alongside the locked `v0-prototype` Test violation hours (H2 2019, 4,032h):

| Controller Architecture | Unserved Energy (%, Val) | Curtailment Penalty (€, Val) | Val Violation Hours (4,368h, T12) | Locked Test Viol. Hours (`v0-prototype`, 4,032h) | Mean Exceedance (kW, Val) | Strict Feasible? |
|---|---:|---:|---:|---:|---:|:---:|
| **LP' / MILP' Theoretical Oracle** | 0.67% | 8,660.86 | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | Theoretical Benchmark |
| **Peak-Shave-Minimum (PSM')** | 0.83% | 10,624.67 | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | **Strict Feasible** |
| **Shield + PSM'** | 0.83% | 10,593.17 | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | **Strict Feasible** |
| **Shield + Idle (Zero-Battery)** | 1.11% | 14,227.17 | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | **Strict Feasible** |
| **PPO-Diff-Shield (`final` / `best`)** | **0.78% / 0.74%** | **10,005.25 / 9,528.01** | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | **Strict Feasible (Best Safe RL)** |
| **Shielded PPO [Clamped Full-Batch]** | 1.10% | 14,192.95 | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | **Strict Feasible (Parked)** |
| **Shielded PPO [Clamped Minibatch]** | 1.03% | 13,283.30 | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | **Strict Feasible** |
| **PPO-Aux-Shield (Aux. Shield Loss)** | 1.01% | 12,930.01 | **0.0h (0.0%)** | **0.0h (0.0%)** | **0.0000** | **Strict Feasible** |
| **PPO-Residual [Clamped FB]** | 0.04% | 570.12 | 465.6h (10.7%) | 134.8h (3.3%) | 2.8750 | Fails Safety |
| **PPO-Residual [Clamped MB]** | 0.57% | 7,264.57 | 401.2h (9.2%) | 66.4h (1.6%) | 1.5051 | Fails Safety |
| **PPO-Lagrangian [Clamped MB]** | 0.46% | 5,896.48 | 409.8h (9.4%) | 58.5h (1.5%) | 2.5058 | Fails Feasibility |
| **PPO [Clamped Minibatch] (Best PPO)** | 0.12% | 1,539.02 | 450.4h (10.3%) | 132.2h (3.3%) | 3.0970 | Fails Safety |
| **Grid-Only (Unmanaged Baseline)** | 0.00% | 0.00 | 468.0h (10.7%) | 153.0h (3.8%) | 3.2571 | Fails Safety |

![Broken-X Cost vs. Safety & Inference Latency Pareto](results/figures/presentation/02_pareto_cost_safety.png)

---

### Regime 1: Causal Forecast & Control Baselines (`results/tables/regime_benchmarks_val.csv`)

To place post-fix safe RL against realistic causal receding-horizon optimization (as well as hindsight oracles), [`tools/report/eval_regimes_val.py`](tools/report/eval_regimes_val.py) benchmarks `LP'`, `MILP'`, 24-hour `MPCPolicy` across `perfect`, `causal_noisy` (exact D-1 spot prices; $\sqrt{h/H}$-scaled clipped non-negative Gaussian noise on PV 15%, wind 20%, load 8%), and `persistence` forecasts, alongside 17-D `clamped_ppo_diff_shield` and 23-D causal-forecast `causal23_ppo_diff_shield` (6h `causal_noisy` lookahead) on the 26-week Validation Split:

| Controller / Forecast Mode | Val Cost (€) | Mean Exc (kW) | Viol. Rate (%) | $\text{CVaR}_{0.95}$ Cost (€) | Throughput (MWh) | Active SoC (%) | Latency P50 / P99 (ms) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`LP'` (Perfect Foresight, 168h)** | *53,290.51* | 0.0000 | 0.00% | — | 21.84 | 37.98% | — | Continuous hindsight bound |
| **`MILP'` (Exact Motor Floor, 168h)** | *53,290.51* | 0.0000 | 0.00% | — | 21.84 | 37.96% | — | Exact binary industrial constraint |
| **`MPCPolicy(perfect, H=24)`** | 53,404.83 | 0.0000 | 0.00% | 63.55 | 19.17 | 29.99% | 3.27 / 3.78 | Non-causal 24h oracle forecast |
| **`MPCPolicy(causal_noisy, H=24)`** | 53,514.28 | 0.0000 | 0.00% | 63.90 | 19.78 | 33.63% | 3.41 / 3.68 | Causal noisy 24h forecast |
| **`causal23_ppo_diff_shield` (`best`)** | **55,268.23 ± 685** | **0.0000** | **0.00%** | **65.77** | 49.61 | 33.81% | **1.18 / 1.34** | Mean < `Shield + PSM'`; Seed 103 **€54,545.38** |
| **`clamped_ppo_diff_shield` (`best`)** | **55,402.89 ± 1,212** | **0.0000** | **0.00%** | **65.20** | 63.26 | 40.67% | **1.14 / 1.27** | Mean < `PSM'`; Seed 103 **€54,337.06** |
| **`PSM'` (Domain Heuristic)** | 55,530.22 | 0.0000 | 0.00% | 68.49 | 12.35 | 11.20% | 0.03 / 0.05 | Myopic priority rules |
| **`causal23_ppo_diff_shield` (`final`)** | **56,475.07 ± 1,312** | **0.0000** | **0.00%** | **67.18** | 86.23 | 39.81% | **1.15 / 1.30** | Seed 103 €55,361.98 (< `PSM'`) |
| **`clamped_ppo_diff_shield` (`final`)** | **56,696.94 ± 1,279** | **0.0000** | **0.00%** | **66.71** | 101.16 | 50.53% | **1.14 / 1.30** | ~3.0× faster than LP-MPC |
| **`MPCPolicy(persistence, H=24)`** | 58,134.64 | 0.0000 | 0.00% | 73.98 | 4.83 | 97.21% | 3.26 / 3.53 | Naive persistence hurts MPC |
| **`Grid-Only` (Unmanaged)** | 45,337.92 | 3.2571 | 10.71% | 29.20 | 0.00 | 100.00% (static 50%) | 0.002 / 0.004 | 468 violation hours |

*(Additional 23-D causal configs in `results/tables/per_seed_metrics.csv`: `causal23_ppo_shielded` achieves €59,155.29 ± €565.27 with 0.00% violations, and `causal23_ppo_residual` incurs €55,529.66 ± €3,452.49 with 7.69% violations / 335.8h.)*

![Regime 1 Causal Forecast & MPC Benchmark](results/figures/presentation/04_regime1_causal_mpc.png)

**Takeaway:** While receding-horizon `MPCPolicy(causal_noisy, H=24)` achieves €53,514.28 when forecasts are well-calibrated, its performance degrades sharply under naive `persistence` forecasts (€58,134.64). Model-free `clamped_ppo_diff_shield` requires **no multi-step forecast model**, executes at **1.14 ms P50 (~3.0× faster than LP-MPC at 3.41 ms)**, beats `MPCPolicy(persistence, H=24)` by €1,438–€2,732, and both `clamped_ppo_diff_shield` (`best`: €55,402.89) and `causal23_ppo_diff_shield` (`best`: €55,268.23) beat `PSM'` (€55,530.22) on 5-seed mean (while `final` Seed 101 reaches €55,045.60 and `causal23` `final` Seed 103 reaches €55,361.98).

---

### Regime 2: C&I Demand-Charge & Zone-VoLL Ablation (`results/tables/regime_benchmarks_val.csv`)

To evaluate controllers beyond uniform curtailment penalties, Regime 2 (`ci_*` in [`configs/config.yaml`](configs/config.yaml)) introduces realistic commercial & industrial (C&I) tariff structures over the 26-week Validation Split:
- **Weekly Peak Demand Charge:** `2.333333 EUR/kW-week` (prorated from `~10 EUR/kW-month`) charged via telescoping weekly peak increments (`cost_demand_charge_eur`).
- **Heterogeneous Zone VoLL (`EUR/kWh`):** `industrial: 3.0`, `residential: 1.5`, `commercial: 1.2`, `stadium: 0.4`, `street: 0.2`.
- **Industrial Trip Switching Penalty:** `50.0 EUR` incurred whenever industrial curtailment crosses the 30% motor-shutdown threshold from a non-tripped state.
- **Causal Noisy Forecasts:** `state_dim: 23`, `forecast_horizon_hours: 6`, `forecast_mode: causal_noisy`.

| Controller (Regime 2 Val, 26 Weeks) | Total Cost (€) | Energy (€) | Wear (€) | Unserved (€) | Demand Charge (€) | Trip (€) | Throughput (MWh) | Active SoC (%) | Mean Exc (kW) | Viol. Rate (%) | Latency P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`LP'` (Hindsight Bound)** | *64,513.29* | 44,192.08 | 1,252.28 | 1,957.80 | 17,111.13 | 0.00 | 62.61 | 72.07% | 0.0000 | 0.00% | — |
| **`MILP'` (Exact Motor Floor)** | *64,513.29* | 44,192.08 | 1,252.28 | 1,957.80 | 17,111.13 | 0.00 | 62.61 | 72.21% | 0.0000 | 0.00% | — |
| **`MPCPolicy(causal_noisy, H=24)`** | 65,250.30 | 44,138.11 | 1,574.31 | 2,454.95 | 17,082.94 | 0.00 | 78.72 | 78.34% | 0.0000 | 0.00% | 3.69 |
| **`MPCPolicy(perfect, H=24)`** | 65,472.28 | 44,079.45 | 1,710.26 | 2,852.75 | 16,829.82 | 0.00 | 85.51 | 70.08% | 0.0000 | 0.00% | 3.52 |
| **`Grid-Only` (Unmanaged)** | *66,095.86* | 45,337.92 | 0.00 | 0.00 | 20,757.94 | 0.00 | 0.00 | 100.0% (static 50%) | 3.2571 | 10.71% | 0.002 |
| **`Shield + Idle` (Zero-Battery)** | *69,115.11* | 44,417.53 | 0.00 | 5,036.83 | 19,660.75 | 0.00 | 0.00 | 100.0% (static 50%) | **0.0000** | **0.00%** | 0.002 |
| **`ci_ppo_diff_shield` (1-D + `CapShield`)** | **69,784.24 ± 617** | 44,430.98 | 635.13 | 4,456.67 | 20,261.46 | 0.00 | 31.76 | 16.71% | **0.0000** | **0.00%** | **1.18** |
| **`Shield + PSM'`** | 70,363.16 | 44,660.23 | 246.95 | 3,949.27 | 21,506.71 | 0.00 | 12.35 | 11.20% | 0.0000 | 0.00% | 0.036 |
| **`PSM'` (Domain Heuristic)** | 70,370.73 | 44,658.60 | 246.95 | 3,960.34 | 21,504.84 | 0.00 | 12.35 | 11.20% | 0.0000 | 0.00% | 0.035 |
| **`ci_ppo_shielded` (Post-Hoc Shield)** | 70,581.07 ± 606 | 44,437.42 | 654.80 | 4,894.68 | 20,594.16 | 0.00 | 32.74 | 17.38% | **0.0000** | **0.00%** | 0.15 |
| **`ci_ppo_diff_6d` (6-D QP, No `CapShield`)** | 86,379.55 ± 5,295 | 44,004.60 | 628.32 | 21,644.53 | 20,102.11 | 0.00 | 31.42 | 17.14% | **0.0000** | **0.00% (0.0h)** | 2.00 |

![Regime 2 C&I Demand-Charge & Zone-VoLL Ablation](results/figures/presentation/05_regime2_ci_ablation.png)

**Key Regime 2 Insights:**
- **`ci_ppo_diff_shield` Beats `PSM'`, `Shield + PSM'`, and `ci_ppo_shielded`, While `Shield + Idle` Leads on Mean:** Under C&I tariffs, `ci_ppo_diff_shield` achieves **€69,784.24 ± €617.02** (`31.76 MWh`, `16.71%` active SoC, **0.00% violations**, €0.00 trip penalties), outperforming `PSM'` (€70,370.73, cutting €1,243.38 in peak demand charges), `Shield + PSM'` (€70,363.16), and `ci_ppo_shielded` (€70,581.07 ± €605.99), with Seed 105 reaching **€68,906.11** (below `Shield + Idle`). On 5-seed mean, however, the zero-battery **`Shield + Idle` baseline (€69,115.11)** remains lower (primarily by avoiding battery charging peak-demand exposure, keeping demand charges at €19,660.75 vs. €20,261.46), while causal `MPCPolicy(causal_noisy, H=24)` reaches **€65,250.30**.
- **Honest Ablation (`ci_ppo_diff_6d`) — Fixed Industrial Slope & Costly Continuous Relaxation:** Before T1, `ci_ppo_diff_6d` leaked **8.80%** because the 6-D QP hardcoded a linear `0.5*(1+a_z)` shedding map that overestimated industrial curtailment by 3× below the `0.90` motor shutdown threshold. Re-training `ci_ppo_diff_6d` with T1's exact `(1 - motor_min)/thresh` piecewise industrial slope and 6 active-set iterations reduces grid exceedance to **0.0000 kW (`0.0h` violations, max `0.0001 kW`)** across all 5 seeds without `CapShield`. However, continuous relaxation of the discontinuous motor rule still triggers heavy over-shedding (€21,644.53 unserved, **€86,379.55 ± €5,295.04** total cost), confirming why pairing the 1-D battery safety layer with `CapShield` remains strongly preferable economically.

---

### 2019-W04 OOD Contingency & Tail-Risk Suite (`results/tables/regime_benchmarks_val.csv`)

To test robustness beyond nominal Validation profiles, [`tools/report/eval_regimes_val.py`](tools/report/eval_regimes_val.py) evaluates `PSM'`, `Shield + PSM'`, `MPCPolicy(causal_noisy, H=24)`, `Shield + MPCPolicy(causal_noisy, H=24)`, and all 5 seeds (101–105, `final`) of `clamped_ppo_diff_shield` on the representative cap-binding Validation week **2019-W04** across four non-destructive out-of-distribution (OOD) contingencies (`evaluate_ood_stress_suite`, with `CapShield` receiving perturbed `obs` via `set_last_obs` and `MPCPolicy` preserving clean $t=0$ state/prices):

| Contingency (2019-W04, 168h) | Controller | Cost (€) | $\text{CVaR}_{0.95}$ Cost (€) | Mean / Max Exc (kW) | Viol. Rate (%) / Hours | Shield Rate | Latency P50 / P99 (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`nominal`** | `PSM'` | 8,115.88 | 120.46 | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 0.03 / 0.05 |
| **`nominal`** | `Shield + PSM'` | 8,110.49 | 120.42 | 0.0000 / 0.00 | 0.00% (0h) | 0.589 | 0.04 / 0.05 |
| **`nominal`** | `MPCPolicy(causal_noisy, H=24)` | **7,875.50** | **120.32** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.56 / 5.08 |
| **`nominal`** | `Shield + MPCPolicy(causal_noisy, H=24)` | **7,875.50** | **120.32** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.57 / 4.08 |
| **`nominal`** | `clamped_ppo_diff_shield` (5-seed Mean ± SD) | 7,980.30 ± 177.99 | 120.42 | 0.0000 / 0.00 | **0.00% (0h)** | 0.561 | **1.17 / 1.32** |
| **`pv_inverter_trip`** | `PSM'` | 8,872.52 | 121.39 | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 0.03 / 0.05 |
| **`pv_inverter_trip`** | `Shield + PSM'` | 8,866.84 | 121.35 | 0.0000 / 0.00 | 0.00% (0h) | 0.619 | 0.04 / 0.05 |
| **`pv_inverter_trip`** | `MPCPolicy(causal_noisy, H=24)` | **8,729.30** | **121.35** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.57 / 3.76 |
| **`pv_inverter_trip`** | `Shield + MPCPolicy(causal_noisy, H=24)` | **8,729.30** | **121.35** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.58 / 3.84 |
| **`pv_inverter_trip`** | `clamped_ppo_diff_shield` (5-seed Mean ± SD) | 8,817.52 ± 131.23 | **121.35** | 0.0000 / 0.00 | **0.00% (0h)** | 0.600 | **1.17 / 1.42** |
| **`bess_derate`** | `PSM'` | 7,941.62 | 120.46 | 1.1932 / 36.62 | 11.90% (20h) | 0.000 | 0.03 / 0.05 |
| **`bess_derate`** | `Shield + PSM'` | 8,121.10 | 120.42 | 0.0000 / 0.00 | 0.00% (0h) | 0.631 | 0.04 / 0.05 |
| **`bess_derate`** | `MPCPolicy(causal_noisy, H=24)` | **7,888.08** | **120.32** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.55 / 3.88 |
| **`bess_derate`** | `Shield + MPCPolicy(causal_noisy, H=24)` | **7,888.08** | **120.32** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.61 / 3.93 |
| **`bess_derate`** | `clamped_ppo_diff_shield` (5-seed Mean ± SD) | 7,975.48 ± 169.96 | 120.42 | 0.0000 / 0.00 | **0.00% (0h)** | 0.561 | **1.18 / 1.37** |
| **`grid_cap_reduction`** | `PSM'` | 8,115.09 | 120.46 | 62.7137 / 73.58 | 99.40% (167h) | 0.000 | 0.03 / 0.04 |
| **`grid_cap_reduction`** | `Shield + PSM'` | **17,941.08** | **186.62** | 0.0000 / 0.00 | 0.00% (0h) | 0.994 | 0.04 / 0.05 |
| **`grid_cap_reduction`** | `MPCPolicy(causal_noisy, H=24)` | **17,941.08** | **186.62** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.65 / 4.13 |
| **`grid_cap_reduction`** | `Shield + MPCPolicy(causal_noisy, H=24)` | **17,941.08** | **186.62** | 0.0000 / 0.00 | 0.00% (0h) | 0.000 | 3.69 / 3.87 |
| **`grid_cap_reduction`** | `clamped_ppo_diff_shield` (5-seed Mean ± SD) | 17,955.44 ± 30.18 | **186.62** | 0.0000 / 0.00 | **0.00% (0h)** | 0.977 | **1.18 / 1.36** |
| **`forecast_noise`** | `PSM'` | 8,110.55 | 120.49 | 0.3298 / 8.24 | 11.31% (19h) | 0.000 | 0.03 / 0.05 |
| **`forecast_noise`** | `Shield + PSM'` | 8,105.01 | 120.45 | 0.3354 / 8.29 | 21.43% (36h) | 0.583 | 0.04 / 0.05 |
| **`forecast_noise`** | `MPCPolicy(causal_noisy, H=24)` | **7,899.73** | **120.45** | 0.3354 / 8.29 | 19.64% (33h) | 0.000 | 3.62 / 3.94 |
| **`forecast_noise`** | `Shield + MPCPolicy(causal_noisy, H=24)` | **7,899.73** | **120.45** | 0.3354 / 8.29 | 20.24% (34h) | 0.000 | 3.70 / 3.91 |
| **`forecast_noise`** | `clamped_ppo_diff_shield` (5-seed Mean ± SD) | 8,011.64 ± 151.06 | **120.45** | 0.3354 / 8.29 | 20.95% (35.2 ± 0.8h) | 0.559 | **1.17 / 1.33** |

![2019-W04 OOD Contingency & Tail-Risk Matrix](results/figures/presentation/06_ood_stress_2019w04.png)

**Key OOD Findings (Calibrated 5-Seed Evaluation):**
- **Consistent Observation Perturbation & Runtime Adaptation:** Under `bess_derate` and `grid_cap_reduction`, unshielded `PSM'` suffers **11.90% (20h)** and **99.40% (167h)** cap violations, whereas `Shield + PSM'`, `MPCPolicy(causal_noisy, H=24)`, `Shield + MPCPolicy(causal_noisy, H=24)`, and `clamped_ppo_diff_shield` (all 5 seeds) maintain **0.00% violations (0h)** across `nominal`, `pv_inverter_trip`, `bess_derate`, and `grid_cap_reduction`.
- **`Shield + MPCPolicy` Matches or Beats RL Across Contingencies:** When sensor observations are perturbed consistently across controllers (`CapShield` uses noisy `obs` via `set_last_obs`) and `MPCPolicy` keeps $t=0$ state/prices clean, `Shield + MPCPolicy(causal_noisy, H=24)` matches or outperforms `clamped_ppo_diff_shield` across all five contingencies (e.g., €7,875.50 vs. €7,980.30 ± €177.99 nominal; €7,899.73 vs. €8,011.64 ± €151.06 under `forecast_noise`), while RL retains a ~3.1× inference speed advantage (`1.17 ms` vs. `3.60 ms` P50).
- **Sensor-Noise Limitation Under `forecast_noise`:** Under `forecast_noise` on 17-D observations (which injects noise directly into current PV and spot-price readings), **all** shielded controllers—`Shield + PSM'` (36h / 21.43%), `MPCPolicy` (33h / 19.64%), `Shield + MPCPolicy` (34h / 20.24%), and `clamped_ppo_diff_shield` (35.2 ± 0.8h / 20.95%)—incur identical 0.3354 kW mean exceedance (120.45 EUR $\text{CVaR}_{0.95}$) because the analytical `CapShield` itself relies on noisy current PV/load readings.

---

### Cross-Configuration Ablation Synthesis (Plain PPO vs. PPO-Lagrangian)

The full cross-configuration comparison across unclamped, clamped full-batch, and clamped minibatch setups is synthesized in [`01_ablation_pathologies.png`](results/figures/presentation/01_ablation_pathologies.png), [`03_learning_dynamics.png`](results/figures/presentation/03_learning_dynamics.png), and [`12_seed_stability_tradeoff.png`](results/figures/presentation/12_seed_stability_tradeoff.png) (covering cost decomposition, cap violations, active SoC trajectories, and 5-seed spread).

---

### Locked One-Shot Out-of-Sample Test Evaluation (`v0-prototype` Snapshot)

> [!IMPORTANT]
> **Test Lock Hygiene (`test_lock.json` Untouched):** `results/tables/test_evaluation.csv`, `results/tables/test_stress_weeks.csv`, and `results/runs/test_lock.json` remain the **locked one-shot `v0-prototype` Test snapshot**. To preserve strict out-of-sample evaluation integrity, the Test split was **deliberately NOT unlocked or re-evaluated** after the T1–T13 fixes and new training runs. All post-fix empirical gains are established strictly on the Validation split above.

Evaluated across the 24-week sealed Test Split (H2 2019, 4,032 operating hours):

| Controller Architecture | Test Cost (€, Mean ± SD) | Min / Max Cost (€) | Mean Exceedance (kW) | Violation Rate (%) | Violation Hours | Unserved Energy (%) | Mean Cap-Clipped (kW) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LP' Theoretical Oracle** | *36,991.67* | *36,991.67* | **0.0000** | **0.00%** | **0h** | 0.0158% | 0.0000 |
| **Peak-Shave-Minimum (PSM')**| *37,674.27* | *37,674.27* | **0.0000** | **0.00%** | **0h** | 0.0803% | 0.0000 |
| **Shield + PSM'** | *37,661.61* | *37,661.61* | **0.0000** | **0.00%** | **0h** | 0.0791% | 0.0000 |
| **`clamped_ppo_residual`** | **37,633.81 ± 281** | 37,368.89 / 38,096.80 | 0.0702 | 3.34% | 134.8h ± 43 | 0.0763% | 0.0472 |
| **Shield + Idle (Zero-Battery)**| *39,759.47* | *39,759.47* | **0.0000** | **0.00%** | **0h** | 0.3117% | 0.0000 |
| **`clamped_ppo_diff_shield`** | 39,884.02 ± 522 | 39,136.74 / 40,323.05 | **0.0000** | **0.00%** | **0h** | 0.1760% | 0.0000 |
| **`clamped_ppo_shielded`** | 39,996.21 ± 9 | 39,991.87 / 40,012.22 | **0.0000** | **0.00%** | **0h** | 0.3117% | 0.0083 |
| **`clamped_ppo_shield_aux`** | 40,969.05 ± 320 | 40,573.42 / 41,432.04 | **0.0000** | **0.00%** | **0h** | 0.3028% | 1.0341 |
| **`clamped_mb_ppo_shielded`** | 40,984.24 ± 354 | 40,693.83 / 41,472.30 | **0.0000** | **0.00%** | **0h** | 0.2593% | 4.1518 |
| **`clamped_mb_ppo_residual`** | 41,653.90 ± 1026 | 40,238.37 / 43,083.04 | 0.1269 | 1.65% | 66.4h ± 24 | 0.4370% | 0.3410 |
| **`clamped_mb_ppo`** | 38,824.36 ± 946 | 37,646.69 / 40,218.37 | 0.6798 | 3.28% | 132.2h ± 10 | 0.1472% | 1.3254 |
| **`clamped_ppo`** | 36,732.99 ± 182 | 36,534.95 / 36,907.04 | 0.8302 | 3.78% | 152.4h ± 1 | 0.0000% | 0.0591 |
| **`clamped_ppo_lagrangian`** | 36,767.17 ± 134 | 36,533.51 / 36,870.81 | 0.8355 | 3.79% | 153.0h ± 0 | 0.0000% | 0.0339 |
| **`clamped_mb_ppo_lagrangian`**| 50,468.52 ± 4387 | 46,337.62 / 57,121.92 | 0.1722 | 1.45% | 58.5h ± 25 | 1.1848% | 2.3971 |
| **Grid-Only (Unmanaged)** | *36,562.65* | *36,562.65* | 0.8355 | 3.79% | 153.0h | 0.0000% | 0.0000 |

![Seasonal Arbitrage Context (Val H1 vs. Locked Test H2) & W49 Stress](results/figures/presentation/08_seasonal_test_context.png)

**Candid Analysis of Negative Comparisons on the Locked Test Split:**
- **Failure to Beat Heuristic & Zero-Battery Baselines:** On the locked out-of-sample Test split (H2 2019, 4,032 hours), `clamped_ppo_diff_shield` guarantees strict safety (0.0000 kW exceedance, 0h violations), but its mean operating cost of **€39,884.02 ± €521.60** is higher than both the domain heuristic **PSM' (€37,674.27)** and the zero-battery safety baseline **Shield + Idle (€39,759.47)**.
- **Root Cause (Seasonal Compression of Arbitrage Value):** In H2 2019 (late summer and autumn), grid-cap stress is substantially milder than in winter: unmanaged Grid-Only exceeds $P_{\text{cap}}$ in only 3.79% of hours (153h) versus 10.71% (468h) on the Validation split. Because capacity emergencies are infrequent, the room for economic improvement via battery peak shaving is compressed to just €682.60 (the gap between PSM' and the theoretical LP' oracle). Active cycling under learned policies incurs battery wear (€0.02/kWh) and occasional defensive curtailment without recovering sufficient tariff savings, allowing the passive `Shield + Idle` policy to win on raw cost.
- **Model-Free Shielding Overhead:** Similarly, `clamped_ppo_shield_aux` (€40,969.05) and `clamped_mb_ppo_shielded` (€40,984.24) incur higher costs due to residual action chatter and over-curtailment, while unconstrained `clamped_mb_ppo` (€38,824.36) fails feasibility (0.68 kW exceedance across 132.2 hours). These results confirm that under low-stress operating regimes, tuned heuristics and simple physical overrides outperform uncalibrated safe RL policies.

---

## In-Depth Methodological Breakdown: Strengths, Failures, and Root Causes

Adhering to the two-stage evaluation framework, the methodological breakdown examines the operational trade-offs, theoretical advantages, and failure modes of each evaluated approach.

### Stage 1: PPO Environment & Optimizer Ablation

#### 1. Unclamped PPO (`unclamped_ppo`)
- **Strengths:** Achieves low nominal energy acquisition cost on paper (€45,513.38 on Val).
- **Failures:** Fails safety catastrophically, producing 475.4 violation hours (10.9% of the semester) and 3.35 kW mean exceedance.
- **Root Cause:** In the unclamped environment, battery charging commands can independently push facility net load over $P_{\text{cap}}$. Coupled with the 50:1 cost asymmetry between customer load curtailment (€1,000/MWh) and electricity purchases (~€50–€150/MWh), the unconstrained policy learns to charge aggressively during low tariffs without regard to capacity headroom, triggering charging-induced capacity violations.

#### 2. Clamped Full-Batch PPO (`clamped_ppo`)
- **Strengths:** Eliminates charging-induced violations by construction via the automated physical charging interlock ($P_{\text{ch}}^{\text{allowed}} \le \max(0, P_{\text{cap}} - P_{\text{net-served}})$).
- **Failures:** Still incurs 464.8 violation hours and 3.23 kW mean exceedance, while battery throughput remains parked at only 5.89 MWh (2.8% active SoC).
- **Root Cause:** While the charging interlock guarantees charging cannot cause violations, capacity emergencies from uncontrollable facility demand require active load curtailment or battery discharge. Under full-batch updates, policy gradients averaged over 168 hours fail to associate localized peak surges with shedding actions. The policy executes zero load shedding and leaves the battery essentially idle.

#### 3. Clamped Minibatch PPO (`clamped_mb_ppo` — Representative Best PPO)
- **Strengths:** Escapes the full-batch saddle point, increasing battery throughput more than fourfold to **25.32 MWh** (16.8% active cycling) and actively shaving demand peaks.
- **Failures:** Still leaks 450.4 violation hours and 3.10 kW mean exceedance because it lacks an explicit safety enforcement mechanism.
- **Root Cause:** Minibatching (`batch_size=64`, 5 value epochs) provides frequent localized gradient signals that enable the policy to discover profitable peak-shaving storage actions. However, because unserved load penalties are fifty times more costly than grid purchases, the policy deliberately accepts capacity violations during extreme surges. It is the best-performing unconstrained PPO baseline.

---

### Stage 2: Cross-Architecture Benchmark (Best PPO vs. Safe RL & Baselines)

#### 5. Dual-Lagrangian PPO (`clamped_mb_ppo_lagrangian`)
- **Strengths:** Directly formalizes the constrained MDP using adaptive dual multipliers ($\lambda$).
- **Failures:** Even after fixing dual updates to once per rollout (`T2`, $\lambda_{\text{end}} = 0.314$ vs. `2.354`), operating cost reaches €51,618.03 with €5,896.48 in unserved load penalties while still permitting 2.51 kW mean exceedance across 409.8 violation hours.
- **Root Cause:** With a single dual step per 2,048-step rollout at $\alpha_\lambda = 0.01$, $\lambda$ adapts smoothly rather than oscillating wildly, yet remains too conservative over 146 rollouts to suppress peak exceedances under the 50:1 economic asymmetry.

#### 6. Residual Policy Learning (`clamped_ppo_residual`)
- **Strengths:** Achieves low nominal costs (€46,167.91 on Val; €37,633.81 on locked `v0-prototype` Test) by perturbing the high-performing PSM' heuristic baseline.
- **Failures:** Fails strict feasibility, incurring 465.6 violation hours (2.88 kW mean exceedance) on Val and 134.8h on locked `v0-prototype` Test. Critically under-sheds load during acute stress surges (Week 49).
- **Root Cause:** Exploitation of the soft $G_1$ gate loophole or peak leakage. Unwinding PSM' curtailment during peaks trades capacity overruns for lower immediate curtailment penalties while still violating physical distribution limits.

#### 7. Post-Hoc Environmental Shielding (Shielded PPO — Full-Batch vs. Minibatch)
- **Strengths:** Deterministically guarantees strict zero violations (0h violations, 0.0000 kW exceedance) across all operating hours and random seeds.
- **Failures:** 
  - Under full-batch PPO (`clamped_ppo_shielded`), it collapses into the **"battery parking" moral hazard**: throughput falls to 6.05 MWh (4.0% active SoC), incurring €14,192.95 in unserved penalties and nearly matching the zero-battery `Shield + Idle` baseline (€58,885.36 vs. €58,644.70).
  - Under minibatch PPO (`clamped_mb_ppo_shielded`), asset utilization is restored (58.00 MWh), yet unserved load penalties remain high (€13,283.30, €58,987.41 total cost).
- **Root Cause:** The gradient disconnect between the unshielded policy network and the external runtime safety shield. Because the external filter resolves violations downstream via load shedding, the policy receives zero gradient penalty for idling during peaks, shifting the burden onto customer curtailment.

#### 8. In-the-Loop Safe RL (Differentiable Safety Layer — `PPO-Diff-Shield` & `PPO-Aux-Shield`)
- **Strengths:**
  - **Lowest Safe Operating Cost & G2 Gate Pass:** After T1–T3 fixes and full retraining, `clamped_ppo_diff_shield` improves from **€58,473.79 ± €1,180.66** (`v0-prototype`) to **€56,696.94 ± €1,279.45 (`final`)** and **€55,402.89 ± €1,212.20 (`best`)** on Validation—passing the canonical $G_2 \le 1.05 \times \text{PSM'}$ gate (€58,306.73) on mean, 4/5 `final` seeds, and 5/5 `best` seeds, while both `clamped_ppo_diff_shield` (`best`) and `causal23_ppo_diff_shield` (`best`: €55,268.23) plus `final` Seed 101 (**€55,045.60**) beat strict `1.00× PSM'` (€55,530.22).
  - **Strict Feasibility:** Guarantees 0h violations and 0.0000 kW exceedance across all splits, seeds, and non-sensor-noise 2019-W04 OOD contingencies.
  - **Resolves the Moral Hazard:** Activates sustained battery throughput (**101.16 MWh `final` / 63.26 MWh `best`**, 50.5%–40.7% active SoC) and slashes unserved load curtailment penalties to **€10,005.25 (`final`) / €9,528.01 (`best`)**, beating `PSM'` (€10,624.67).
  - **Auxiliary Shield Loss (`PPO-Aux-Shield`):** Provides a model-free alternative (**€58,914.89 ± €920.29**, 72.13 MWh throughput, 1/5 `G2_pass`) with SoC-masked auxiliary discharge targets.
- **Root Cause of Success:** Routing backward gradients in `SafePolicyNetwork.distribution()` through `safety_layer` with the T3 Straight-Through Estimator (STE, $\alpha_{\text{ste}} = 0.1$) while evaluating log-probabilities on raw means (`forward_raw`) lets constraint-boundary Jacobians flow into policy updates without double-projection bias, teaching coordinated peak-shaving without passive parking. Meanwhile, `ci_ppo_diff_6d` retrained under T1's piecewise `(1 - motor_min)/thresh` industrial slope eliminates grid violations (`0.00%`, max `0.0001 kW`), yet continuous motor relaxation incurs heavy over-shedding (€86,379.55 ± €5,295.04), confirming that co-designing the 1-D battery safety layer with `CapShield` remains essential under non-convex motor rules.

#### 9. Classical Heuristics & Mathematical Programming Oracles
- **Peak-Shave-Minimum (PSM'):** Achieves €55,530.22 on Val and €37,674.27 on Test with 0h violations. Its domain-engineered priority queues explicitly exploit the 50:1 cost asymmetry by exhausting storage discharge before shedding customer load, remaining exceptionally hard to beat causally.
- **LP' Theoretical Oracle:** Establishes the global continuous hindsight lower bound (€53,290.51 on Val: €44,192.97 purchase, €436.73 wear, €8,660.86 unserved; €36,991.67 on Test, 0h violations) via linear programming over 168-hour foresight windows. Additionally, exact Mixed-Integer Linear Programming (`MILP'`) via [`tools/report/lp_oracle.py::solve_week_milp`](tools/report/lp_oracle.py) enforces the non-convex industrial motor floor. PSM' operates within **4.03% (€2,240)** of global hindsight LP optimality on Val and **1.81% (€683)** on Test.
- **Shield + Idle (Zero-Battery Control):** Incurs €58,644.70 on Val with €14,227.17 in unserved penalties, proving empirically that runtime safety shielding without active storage dispatch is economically non-viable.

---

## Limitations & Future Work

1. **Copperplate Transmission & Distribution Network Modeling:**
   The environment adopts a single-bus (copperplate) microgrid assumption. All generation, storage, and demand components connect to a common point without modeling distribution line impedances, AC branch power flows, bus voltage bounds ($V_{\text{min}} \le |V| \le V_{\text{max}}$), line thermal current capacities, or reactive power compensation ($Q$). In realistic distribution networks, localized voltage rise from solar reverse feed and branch line congestion often bind before aggregate substation limits.
2. **Synthetic Multi-Zone Load Disaggregation:**
   While total site demand is grounded in empirical French national load curves, sectoral disaggregation into five demand zones (35% residential, 25% commercial, 25% industrial, 10% stadium, 5% street lighting) relies on synthetic deterministic apportioning rather than physically metered distinct customer feeders. Real-world microgrid zones exhibit uncorrelated demand spikes, stochastic machine cycling, and non-simultaneous peak periods.
3. **Linear Throughput Degradation vs. Non-Linear Electrochemistry:**
   Battery aging is approximated via a constant marginal throughput cost ($c_w = 0.02\,\text{EUR/kWh}$). Real-world lithium-ion degradation depends non-linearly on operational temperature, cell chemistry, cycling depth-of-discharge (DoD), high C-rate mechanical stress, calendar aging at elevated SoC, and solid electrolyte interphase (SEI) growth.
4. **Uniform Value of Lost Load (VoLL) in Regime 1 vs. Regime 2 C&I Extension:**
   Regime 1 penalizes curtailment uniformly at $c_u = 1.00\,\text{EUR/kWh}$. Regime 2 (`ci_*`) partially addresses this limitation by introducing heterogeneous sector VoLL (`industrial: 3.0`, `residential: 1.5`, `commercial: 1.2`, `stadium: 0.4`, `street: 0.2 EUR/kWh`), weekly peak demand charges (`2.333333 EUR/kW-week`), and a `50.0 EUR` industrial trip switching penalty, though multi-hour interruption duration curves remain simplified.
5. **Discrete Hourly Time Step Dynamics:**
   The simulation advances in discrete 1-hour time intervals ($\Delta t = 1.0\,\text{h}$). Intra-hour power volatility, sub-second inverter ramping, spinning reserve dispatch, and frequency stabilization dynamics are not captured.
6. **Post-Fix Validation Runs (T9–T13) vs. Locked `v0-prototype` Test Snapshot:**
   All 80 post-fix 300k-step runs (`clamped_*`, `clamped_mb_*`, `clamped_ppo_diff_shield`, `clamped_ppo_shield_aux`, `causal23_*`, `ci_*`) have been re-trained and evaluated on Validation (`results/tables/per_seed_metrics.csv`, `runs.csv`, `regime_benchmarks_val.csv`). However, the legacy unclamped `s3_*` runs (20 runs) and the locked one-shot Test tables (`results/tables/test_evaluation.csv`, `test_stress_weeks.csv`, `results/runs/test_lock.json`) deliberately preserve the initial `v0-prototype` snapshot without re-unlocking the Test split.

---

## Repository Architecture & Reproducibility

### Codebase Organization

The codebase is organized into modular packages separating environment physics, policy architectures, execution pipelines, and evaluation tooling:

| Subdirectory | Core Responsibility | Key Modules |
|---|---|---|
| [`configs/`](configs) | Centralized configuration management | [`config.yaml`](configs/config.yaml) (hyperparameters, physics constants, Regime 1 & Regime 2 configs) |
| [`env/`](env) | Physical microgrid simulation | [`smartgrid_env.py`](env/smartgrid_env.py) (Gymnasium physics, causal forecasts, Regime 2 C&I tariffs), [`cap_shield.py`](env/cap_shield.py) (analytical safety shield) |
| [`agents/`](agents) | OOP policy hierarchy & safety projections | [`ppo.py`](agents/ppo.py) (`PPOAgent`, `PPOLagrangianAgent`, `PPOResidualAgent`, `PPOShieldedAgent`, `PPOAuxShieldLossAgent`, `PPODiffSafetyLayerAgent`), [`safety_layer.py`](agents/safety_layer.py) (`DifferentiableSafetyLayer`, `SafePolicyNetwork`), [`heuristics.py`](agents/heuristics.py) (`PSM'`, `GridOnly`), [`mpc.py`](agents/mpc.py) (`MPCPolicy`) |
| [`experiments/`](experiments) | Training harnesses & batch execution | [`train_core.py`](experiments/train_core.py) (inner loop), [`run_shield_variants.py`](experiments/run_shield_variants.py) (comparative evaluation suites) |
| [`tools/report/`](tools/report) | Evaluation, oracles, & figure generation | [`lp_oracle.py`](tools/report/lp_oracle.py) (HiGHS LP & MILP oracles), [`eval_regimes_val.py`](tools/report/eval_regimes_val.py) (Regime 1, Regime 2, & 2019-W04 OOD Val benchmark), [`evaluate_test_split.py`](tools/report/evaluate_test_split.py) (locked test runner, CVaR tail risk, OOD stress testing), [`plots.py`](tools/report/plots.py) (report & stress-trajectory plotting), [`load_results.py`](tools/report/load_results.py) (canonical G2, regime, & training-curve helpers) |
| [`tests/`](tests) | Deterministic unit and integration tests | 127 automated tests covering physics, causal forecasts, Regime 2 tariffs, buffer mechanics, GAE advantages, agent serialization, LP/MILP oracles, MPC, stress evaluation, and report loaders |
| [`results/`](results) | Committed experimental artifacts | `tables/` (`runs.csv`, `per_seed_metrics.csv`, `regime_benchmarks_val.csv`, `frozen_baselines.csv`, locked `test_evaluation.csv`), `figures/` (12 consolidated PNG benchmark figures in `results/figures/presentation/`), `runs/` (per-seed JSON logs) |

### Quickstart Execution Guide

To replicate the empirical benchmark from a clean repository checkout:

```bash
# 1. Environment initialization
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Automated data ingestion and preprocessing (OPSD, PVGIS, Open-Meteo)
python data/download.py
python data/prepare.py

# 3. Execute comprehensive unit and integration test suite (127 tests)
pytest tests/ -v

# 4. Train an agent configuration (e.g., Differentiable Safety Layer, Seed 101)
python experiments/run_shield_variants.py --methods diff_shield --seeds 101

# 5. Export summary metric tables, run Regime 1/2/OOD Val benchmarks, and generate trajectory plots
python tools/report/export_csv.py
python tools/report/eval_regimes_val.py
python tools/report/plot_stress_trajectory.py
```

---

## Citations & References

```bibtex
@article{salaorni2025ernesto,
  title   = {A novel digital twin for battery energy storage systems in micro-grids},
  author  = {Salaorni, Davide and Bianchi, Federico and Colnago, Silvia
             and Barisione, Andrea and Trov{\`o}, Francesco and Restelli, Marcello},
  journal = {Journal of Energy Storage},
  volume  = {132},
  pages   = {117745},
  year    = {2025},
  doi     = {10.1016/j.est.2025.117745}
}

@inproceedings{salaorni2025reinforcement,
  title         = {A Reinforcement Learning Approach for Optimal Control in Microgrids},
  author        = {Salaorni, Davide and Bianchi, Federico and Trov{\`o}, Francesco
                   and Restelli, Marcello},
  booktitle     = {2025 International Joint Conference on Neural Networks (IJCNN)},
  year          = {2025},
  eprint        = {2506.22995},
  archivePrefix = {arXiv},
  doi           = {10.1109/IJCNN64981.2025.11227643}
}

@article{dalal2018safe,
  title         = {Safe Exploration in Continuous Action Spaces},
  author        = {Dalal, Gal and Dvijotham, Krishnamurthy and Vecerik, Matej
                   and Hester, Todd and Paduraru, Cosmin and Tassa, Yuval},
  journal       = {arXiv preprint arXiv:1801.08757},
  year          = {2018},
  eprint        = {1801.08757},
  archivePrefix = {arXiv}
}

@article{bengio2013estimating,
  title         = {Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation},
  author        = {Bengio, Yoshua and L{\'e}onard, Nicholas and Courville, Aaron},
  journal       = {arXiv preprint arXiv:1308.3432},
  year          = {2013},
  eprint        = {1308.3432},
  archivePrefix = {arXiv}
}

@article{schulman2017proximal,
  title         = {Proximal Policy Optimization Algorithms},
  author        = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla
                   and Radford, Alec and Klimov, Oleg},
  journal       = {arXiv preprint arXiv:1707.06347},
  year          = {2017},
  eprint        = {1707.06347},
  archivePrefix = {arXiv}
}

@inproceedings{schulman2016high,
  title         = {High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  author        = {Schulman, John and Moritz, Philipp and Levine, Sergey
                   and Jordan, Michael I. and Abbeel, Pieter},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2016},
  eprint        = {1506.02438},
  archivePrefix = {arXiv}
}

@article{silver2018residual,
  title         = {Residual Policy Learning},
  author        = {Silver, Tom and Allen, Kelsey R. and Tenenbaum, Joshua B.
                   and Kaelbling, Leslie Pack},
  journal       = {arXiv preprint arXiv:1812.06298},
  year          = {2018},
  eprint        = {1812.06298},
  archivePrefix = {arXiv}
}

@article{spoor2025towards,
  title         = {Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning},
  author        = {Spoor, Lindsay and Serra-G{\'o}mez, {\'A}lvaro and Plaat, Aske and Moerland, Thomas M.},
  journal       = {arXiv preprint arXiv:2510.17564},
  year          = {2025},
  eprint        = {2510.17564},
  archivePrefix = {arXiv}
}

@inproceedings{tessler2019reward,
  title         = {Reward Constrained Policy Optimization},
  author        = {Tessler, Chen and Mankowitz, Daniel J. and Mannor, Shie},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2019},
  eprint        = {1805.11074},
  archivePrefix = {arXiv}
}
```

---

Data Sources: Open Power System Data (CC BY 4.0), PVGIS / EU Joint Research Centre (EC reuse policy), Open-Meteo (CC BY 4.0).
