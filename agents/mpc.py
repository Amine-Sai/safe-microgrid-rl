"""Receding-Horizon Model Predictive Control (MPC) policy for microgrid dispatch.

Solves a causal receding-horizon optimization problem over a sliding window H (default H=24)
using HiGHS linear programming (LP) or mixed-integer linear programming (MILP).
Guarantees physical admissibility, respects battery limits and grid import cap,
and supports perfect, persistence, and noisy forecast modes with per-step latency measurement.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from env.cap_shield import shed_actions_for
from env.smartgrid_env import default_config, resolve_grid_import_cap


class MPCPolicy:
    r"""Causal receding-horizon Model Predictive Control (MPC) policy.

    Formulates and solves a multi-step lookahead optimization problem using HiGHS LP
    (continuous relaxation) or MILP (with binary industrial motor tripping) to compute
    optimal battery dispatch and load shedding actions while strictly enforcing the
    grid import capacity limit $P_{\text{cap}}$.

    Supported forecast modes:
        - "perfect": ground-truth lookahead window extracted from environment time series.
        - "persistence": flat persistence forecast repeating current observation across the horizon.

    Supports additive or relative Gaussian forecast noise via `forecast_noise_std`.
    Tracks per-step solve latency in milliseconds (`last_solve_ms`, `solve_times_ms`).
    """

    name: str = "mpc"

    def __init__(
        self,
        env: Optional[Any] = None,
        horizon: int = 24,
        forecast_mode: str = "perfect",
        forecast_noise_std: float = 0.0,
        forecast_noise_relative: bool = False,
        use_milp: bool = False,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize receding-horizon MPCPolicy.

        Args:
            env: Optional SmartGridEnv or wrapper to extract physical constants and time series.
            horizon: Lookahead horizon in hours (default: 24).
            forecast_mode: Forecast regime, either 'perfect' or 'persistence'.
            forecast_noise_std: Optional Gaussian noise standard deviation for forecasts.
            forecast_noise_relative: If True, scale noise standard deviation by feature magnitude.
            use_milp: If True, solve exact MILP with industrial motor trip variable; else fast LP.
            params: Optional parameter dictionary overriding physical/cost constants.
            seed: Optional random seed for reproducible forecast noise sampling.
        """
        self.env = env
        self.horizon: int = int(horizon)
        self.forecast_mode: str = str(forecast_mode).strip().lower()
        self.forecast_noise_std: float = float(forecast_noise_std)
        self.forecast_noise_relative: bool = bool(forecast_noise_relative)
        self.use_milp: bool = bool(use_milp)
        self.params: Dict[str, Any] = params or {}
        self._rng: np.random.RandomState = np.random.RandomState(seed)

        # Step solve latency metrics (in milliseconds)
        self.last_solve_ms: float = 0.0
        self.solve_times_ms: List[float] = []

        # Synchronize physical and operational parameters dynamically
        self._sync_parameters()

    def _sync_parameters(self, env: Optional[Any] = None) -> None:
        """Synchronize physical, cost, and constraint parameters dynamically."""
        if env is not None:
            self.env = env

        target_env = self.env
        cfg = default_config()

        # Physical battery parameters
        self.c_bat: float = float(
            self.params.get(
                "battery_capacity_kwh",
                getattr(target_env, "bat_capacity_kwh", cfg["battery_capacity_kwh"]),
            )
        )
        self.p_max: float = float(
            self.params.get(
                "battery_max_power_kw",
                getattr(target_env, "bat_max_power_kw", cfg["battery_max_power_kw"]),
            )
        )
        self.eta_c: float = float(
            self.params.get(
                "battery_eta_charge",
                getattr(target_env, "bat_eta_charge", cfg["battery_eta_charge"]),
            )
        )
        self.eta_d: float = float(
            self.params.get(
                "battery_eta_discharge",
                getattr(target_env, "bat_eta_discharge", cfg["battery_eta_discharge"]),
            )
        )
        self.soc_min: float = float(
            self.params.get(
                "battery_soc_min",
                getattr(target_env, "bat_soc_min", cfg["battery_soc_min"]),
            )
        )
        self.soc_max: float = float(
            self.params.get(
                "battery_soc_max",
                getattr(target_env, "bat_soc_max", cfg["battery_soc_max"]),
            )
        )
        self.dt: float = float(
            self.params.get("dt_hours", getattr(target_env, "dt_hours", cfg.get("dt_hours", 1.0)))
        )

        # Grid import capacity limit
        if "grid_import_cap_kw" in self.params and self.params["grid_import_cap_kw"] is not None:
            self.cap: float = float(self.params["grid_import_cap_kw"])
        elif target_env is not None and hasattr(target_env, "grid_import_cap_kw"):
            self.cap = float(target_env.grid_import_cap_kw)
        else:
            self.cap = float(resolve_grid_import_cap(cfg))

        # Reward cost weights
        self.w_wear: float = float(
            self.params.get(
                "cost_wear_eur_kwh",
                getattr(target_env, "cost_wear_eur_kwh", cfg["cost_wear_eur_kwh"]),
            )
        )
        self.w_uns: float = float(
            self.params.get(
                "cost_unserved_eur_kwh",
                getattr(target_env, "cost_unserved_eur_kwh", cfg["cost_unserved_eur_kwh"]),
            )
        )

        # Industrial motor and shedding parameters
        raw_motor = self.params.get(
            "motor_min_pct",
            getattr(target_env, "motor_min_pct", cfg["motor_min_pct"]),
        )
        raw_motor_f = float(raw_motor)
        self.motor_min_pct: float = raw_motor_f / 100.0 if raw_motor_f > 1.0 else raw_motor_f

        self.industrial_shutdown_threshold: float = float(
            self.params.get(
                "industrial_shutdown_threshold",
                getattr(
                    target_env,
                    "industrial_shutdown_threshold",
                    cfg.get("industrial_shutdown_threshold", 0.90),
                ),
            )
        )
        self.shed_mapping: str = str(
            self.params.get(
                "shed_mapping",
                getattr(target_env, "shed_mapping", cfg.get("shed_mapping", "linear")),
            )
        )
        self.violation_tolerance_kw: float = float(
            self.params.get(
                "violation_tolerance_kw",
                getattr(
                    target_env,
                    "violation_tolerance_kw",
                    cfg.get("violation_tolerance_kw", 1e-4),
                ),
            )
        )

    def _get_forecast(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
        K: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate lookahead forecasts of PV, zone loads, and price for horizon K."""
        mode = self.forecast_mode
        if mode == "persistence":
            pv_val = float(obs[1])
            loads_val = np.array(obs[2:7], dtype=np.float32)
            price_kwh_val = max(0.0, float(obs[7])) / 1000.0

            pv = np.full(K, pv_val, dtype=np.float32)
            loads = np.tile(loads_val, (K, 1)).astype(np.float32)
            price_kwh = np.full(K, price_kwh_val, dtype=np.float32)
        elif mode == "perfect":
            target_env = env if env is not None else self.env
            base_env = getattr(target_env, "unwrapped", target_env)

            curr_idx: Optional[int] = None
            if base_env is not None and hasattr(base_env, "_step_idx"):
                curr_idx = int(base_env._start_idx + base_env._step_idx)
            elif info is not None and "data_idx" in info:
                curr_idx = int(info["data_idx"])

            if (
                base_env is not None
                and hasattr(base_env, "_pv_kw_series")
                and curr_idx is not None
            ):
                pv = base_env._pv_kw_series[curr_idx : curr_idx + K].astype(np.float32).copy()
                loads = base_env._zone_loads[curr_idx : curr_idx + K].astype(np.float32).copy()
                raw_price = base_env._price[curr_idx : curr_idx + K]
                price_kwh = (np.maximum(0.0, raw_price) / 1000.0).astype(np.float32)
            else:
                raise ValueError(
                    "forecast_mode='perfect' requires environment or info['data_idx'] "
                    "with access to time-series data."
                )
        else:
            raise ValueError(f"Unknown forecast_mode: '{mode}'. Supported: 'perfect', 'persistence'.")

        # Optional Gaussian forecast noise
        if self.forecast_noise_std > 0.0:
            if self.forecast_noise_relative:
                pv = np.maximum(
                    0.0,
                    pv + self._rng.normal(0.0, self.forecast_noise_std * (pv + 1e-3), size=pv.shape),
                )
                loads = np.maximum(
                    0.0,
                    loads
                    + self._rng.normal(
                        0.0, self.forecast_noise_std * (loads + 1e-3), size=loads.shape
                    ),
                )
                price_kwh = np.maximum(
                    0.0,
                    price_kwh
                    + self._rng.normal(
                        0.0, self.forecast_noise_std * (price_kwh + 1e-4), size=price_kwh.shape
                    ),
                )
            else:
                pv = np.maximum(
                    0.0, pv + self._rng.normal(0.0, self.forecast_noise_std, size=pv.shape)
                )
                loads = np.maximum(
                    0.0, loads + self._rng.normal(0.0, self.forecast_noise_std, size=loads.shape)
                )
                price_kwh = np.maximum(
                    0.0,
                    price_kwh
                    + self._rng.normal(
                        0.0, self.forecast_noise_std / 1000.0, size=price_kwh.shape
                    ),
                )

        return pv.astype(np.float32), loads.astype(np.float32), price_kwh.astype(np.float32)

    def _solve_lp(
        self,
        soc_0: float,
        pv: np.ndarray,
        loads: np.ndarray,
        price_kwh: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Solve continuous HiGHS LP relaxation over horizon K.

        Variables per step t: [p_ch, p_dis, unserved, grid_import]
        Returns:
            Tuple of (p_ch_0, p_dis_0, unserved_0)
        """
        K = len(pv)
        tot_loads = loads.sum(axis=1) if loads.ndim == 2 else loads
        net = tot_loads - pv

        n_vars = 4 * K
        c = np.zeros(n_vars)
        bounds = []
        for t in range(K):
            c[4 * t : 4 * t + 4] = [self.w_wear, self.w_wear, self.w_uns, price_kwh[t]]
            bounds.extend(
                [
                    (0.0, self.p_max),
                    (0.0, self.p_max),
                    (0.0, float(tot_loads[t])),
                    (0.0, self.cap),
                ]
            )

        A_ub = []
        b_ub = []

        # 1. Power balance: import + p_dis + pv + unserved >= load + p_ch
        # <=> p_ch - p_dis - unserved - import <= pv - load = -net
        for t in range(K):
            row = np.zeros(n_vars)
            row[4 * t : 4 * t + 4] = [1.0, -1.0, -1.0, -1.0]
            A_ub.append(row)
            b_ub.append(-float(net[t]))

        # 2. State-of-Charge dynamics: soc_k in [soc_min, soc_max]
        for k in range(K):
            row_hi = np.zeros(n_vars)
            row_lo = np.zeros(n_vars)
            for t in range(k + 1):
                row_hi[4 * t + 0] = self.eta_c * self.dt / self.c_bat
                row_hi[4 * t + 1] = -self.dt / (self.eta_d * self.c_bat)
                row_lo[4 * t + 0] = -self.eta_c * self.dt / self.c_bat
                row_lo[4 * t + 1] = self.dt / (self.eta_d * self.c_bat)
            A_ub.append(row_hi)
            b_ub.append(self.soc_max - soc_0)
            A_ub.append(row_lo)
            b_ub.append(soc_0 - self.soc_min)

        res = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=bounds,
            method="highs",
        )
        if not res.success or res.x is None:
            return 0.0, 0.0, 0.0

        return float(res.x[0]), float(res.x[1]), float(res.x[2])

    def _solve_milp(
        self,
        soc_0: float,
        pv: np.ndarray,
        loads: np.ndarray,
        price_kwh: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Solve exact HiGHS MILP enforcing piecewise industrial motor constraint.

        Variables per step t: [p_ch, p_dis, u_other, u_ind, z_ind, grid_import]
        Returns:
            Tuple of (p_ch_0, p_dis_0, unserved_total_0)
        """
        K = len(pv)
        if loads.ndim == 2:
            load_other = loads[:, :4].sum(axis=1)
            load_ind = loads[:, 4]
            tot_loads = loads.sum(axis=1)
        else:
            load_other = loads * 0.8
            load_ind = loads * 0.2
            tot_loads = loads

        net = tot_loads - pv
        n_vars = 6 * K
        c = np.zeros(n_vars)
        integrality = np.zeros(n_vars)
        lb = np.zeros(n_vars)
        ub = np.zeros(n_vars)

        for t in range(K):
            base = 6 * t
            c[base : base + 6] = [
                self.w_wear,
                self.w_wear,
                self.w_uns,
                self.w_uns,
                0.0,
                price_kwh[t],
            ]
            integrality[base + 4] = 1  # z_ind is binary
            lb[base : base + 6] = 0.0
            ub[base : base + 6] = [
                self.p_max,
                self.p_max,
                float(load_other[t]),
                float(load_ind[t]),
                1.0,
                self.cap,
            ]

        bounds = Bounds(lb, ub)
        A_rows = []
        lhs = []
        rhs = []

        # 1. Power balance
        for t in range(K):
            base = 6 * t
            row = np.zeros(n_vars)
            row[base : base + 6] = [1.0, -1.0, -1.0, -1.0, 0.0, -1.0]
            A_rows.append(row)
            lhs.append(-np.inf)
            rhs.append(-float(net[t]))

        # 2. SoC dynamics
        for k in range(K):
            row_hi = np.zeros(n_vars)
            row_lo = np.zeros(n_vars)
            for t in range(k + 1):
                base = 6 * t
                row_hi[base + 0] = self.eta_c * self.dt / self.c_bat
                row_hi[base + 1] = -self.dt / (self.eta_d * self.c_bat)
                row_lo[base + 0] = -self.eta_c * self.dt / self.c_bat
                row_lo[base + 1] = self.dt / (self.eta_d * self.c_bat)
            A_rows.append(row_hi)
            lhs.append(-np.inf)
            rhs.append(self.soc_max - soc_0)
            A_rows.append(row_lo)
            lhs.append(-np.inf)
            rhs.append(soc_0 - self.soc_min)

        # 3. Piecewise industrial motor operational constraints:
        # u_ind - (motor_min * L_ind) * z_ind <= (1 - motor_min) * L_ind
        # -u_ind + L_ind * z_ind <= 0
        for t in range(K):
            base = 6 * t
            l_i = float(load_ind[t])
            row_ub = np.zeros(n_vars)
            row_ub[base + 3] = 1.0
            row_ub[base + 4] = -self.motor_min_pct * l_i
            A_rows.append(row_ub)
            lhs.append(-np.inf)
            rhs.append((1.0 - self.motor_min_pct) * l_i)

            row_lb = np.zeros(n_vars)
            row_lb[base + 3] = -1.0
            row_lb[base + 4] = l_i
            A_rows.append(row_lb)
            lhs.append(-np.inf)
            rhs.append(0.0)

        constraints = LinearConstraint(np.array(A_rows), lhs, rhs)
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if not res.success or res.x is None:
            return 0.0, 0.0, 0.0

        p_ch_0 = float(res.x[0])
        p_dis_0 = float(res.x[1])
        u_other_0 = float(res.x[2])
        u_ind_0 = float(res.x[3])
        return p_ch_0, p_dis_0, u_other_0 + u_ind_0

    def get_action(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
    ) -> np.ndarray:
        r"""Compute receding-horizon MPC continuous control action $\in [-1, 1]^6$.

        Args:
            obs: Observation vector from environment.
            info: Optional auxiliary environment info dictionary.
            env: Optional environment reference for dynamic synchronization.

        Returns:
            Continuous action vector [battery_cmd, shed_res, shed_com, shed_stad, shed_street, shed_ind] in [-1, 1]^6.
        """
        t0 = time.perf_counter()

        if env is not None:
            self._sync_parameters(env)

        soc_0 = float(obs[0])
        current_pv = float(obs[1])
        current_loads = np.array(obs[2:7], dtype=np.float32)
        total_current_load = float(current_loads.sum())

        target_env = env if env is not None else self.env
        base_env = getattr(target_env, "unwrapped", target_env)

        rem_steps = self.horizon
        if (
            base_env is not None
            and hasattr(base_env, "episode_length")
            and hasattr(base_env, "_step_idx")
        ):
            rem_steps = max(1, int(base_env.episode_length - base_env._step_idx))
        K = min(self.horizon, rem_steps)

        pv_f, loads_f, price_f = self._get_forecast(obs, info=info, env=target_env, K=K)

        if self.use_milp:
            p_ch_0, p_dis_0, u_0 = self._solve_milp(soc_0, pv_f, loads_f, price_f)
        else:
            p_ch_0, p_dis_0, u_0 = self._solve_lp(soc_0, pv_f, loads_f, price_f)

        # Map battery power to continuous action in [-1, 1]
        if self.p_max > 0:
            bat_cmd = float(np.clip((p_ch_0 - p_dis_0) / self.p_max, -1.0, 1.0))
        else:
            bat_cmd = 0.0

        # Safety: check potential grid import exceedance on real current observation
        # p_bat > 0 represents charging (draws power), p_bat < 0 discharging (injects power)
        p_bat_est = bat_cmd * self.p_max
        est_import = max(0.0, total_current_load - current_pv + p_bat_est)
        excess_kw = max(0.0, est_import - self.cap)
        required_shed_kw = max(u_0, excess_kw)

        shed_acts = shed_actions_for(
            excess_kw=required_shed_kw,
            obs=obs,
            motor_min_fraction=self.motor_min_pct,
            industrial_shutdown_threshold=self.industrial_shutdown_threshold,
            shed_mapping=self.shed_mapping,
        )

        action = np.empty(6, dtype=np.float32)
        action[0] = bat_cmd
        action[1:] = shed_acts

        # Measure solve latency in milliseconds
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        self.last_solve_ms = float(dt_ms)
        self.solve_times_ms.append(float(dt_ms))

        return np.clip(action, -1.0, 1.0)

    def act(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
    ) -> np.ndarray:
        """Alias for get_action matching heuristic policy convention."""
        return self.get_action(obs, info=info, env=env)

    def predict(
        self,
        obs: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
        env: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """SB3-compatible policy interface returning (action, state)."""
        return self.get_action(obs, env=env), state

    def reset(self, env: Optional[Any] = None) -> None:
        """Reset policy solve time telemetry at episode boundary."""
        if env is not None:
            self._sync_parameters(env)
        self.last_solve_ms = 0.0
        self.solve_times_ms.clear()

    @property
    def mean_solve_latency_ms(self) -> float:
        """Mean solve latency across recorded steps in ms."""
        return float(np.mean(self.solve_times_ms)) if self.solve_times_ms else 0.0

    @property
    def max_solve_latency_ms(self) -> float:
        """Maximum solve latency across recorded steps in ms."""
        return float(np.max(self.solve_times_ms)) if self.solve_times_ms else 0.0

    @property
    def total_solve_latency_ms(self) -> float:
        """Cumulative solve latency across recorded steps in ms."""
        return float(np.sum(self.solve_times_ms)) if self.solve_times_ms else 0.0


def mpc_prime_policy(env: Optional[Any] = None, **kwargs: Any) -> MPCPolicy:
    """Construct a configured instance of MPCPolicy."""
    return MPCPolicy(env=env, **kwargs)
