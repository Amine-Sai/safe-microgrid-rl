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

from env.cap_shield import SHED_ORDER, shed_actions_for
from env.smartgrid_env import (
    NUM_ZONES,
    ZONE_NAMES,
    default_config,
    resolve_grid_import_cap,
    shed_action_to_fractions,
)


_SHED_TIE_RANK: Dict[int, int] = {
    entry["obs_idx"] - 2: rank for rank, entry in enumerate(SHED_ORDER)
}
_INDUSTRIAL_ZONE_IDX: int = ZONE_NAMES.index("industrial")


class MPCPolicy:
    r"""Causal receding-horizon Model Predictive Control (MPC) policy.

    Formulates and solves a multi-step lookahead optimization problem using HiGHS LP
    (continuous relaxation) or MILP (with binary industrial motor tripping) to compute
    optimal battery dispatch and load shedding actions while strictly enforcing the
    grid import capacity limit $P_{\text{cap}}$.

    Supported forecast modes:
        - "perfect" / "oracle": ground-truth lookahead window extracted from environment time series.
        - "persistence": flat persistence forecast repeating current observation across the horizon
          (or exact D-1 prices with persisted PV/wind/loads when matching env.forecast_mode).
        - "causal_noisy": causal forecast via env.get_horizon_data(...) with exact D-1 prices
          and lead-time-scaled noisy PV/wind/load predictions.

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
            forecast_mode: Forecast regime ('perfect', 'oracle', 'persistence', or 'causal_noisy').
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
        base_env = getattr(target_env, "unwrapped", target_env)
        cfg = default_config()

        # Physical battery parameters
        self.c_bat: float = float(
            self.params.get(
                "battery_capacity_kwh",
                getattr(base_env, "bat_capacity_kwh", cfg["battery_capacity_kwh"]),
            )
        )
        self.p_max: float = float(
            self.params.get(
                "battery_max_power_kw",
                getattr(base_env, "bat_max_power_kw", cfg["battery_max_power_kw"]),
            )
        )
        self.eta_c: float = float(
            self.params.get(
                "battery_eta_charge",
                getattr(base_env, "bat_eta_charge", cfg["battery_eta_charge"]),
            )
        )
        self.eta_d: float = float(
            self.params.get(
                "battery_eta_discharge",
                getattr(base_env, "bat_eta_discharge", cfg["battery_eta_discharge"]),
            )
        )
        self.soc_min: float = float(
            self.params.get(
                "battery_soc_min",
                getattr(base_env, "bat_soc_min", cfg["battery_soc_min"]),
            )
        )
        self.soc_max: float = float(
            self.params.get(
                "battery_soc_max",
                getattr(base_env, "bat_soc_max", cfg["battery_soc_max"]),
            )
        )
        self.dt: float = float(
            self.params.get("dt_hours", getattr(base_env, "dt_hours", cfg["dt_hours"]))
        )

        # Grid import capacity limit
        if "grid_import_cap_kw" in self.params and self.params["grid_import_cap_kw"] is not None:
            self.cap: float = float(self.params["grid_import_cap_kw"])
        elif base_env is not None and hasattr(base_env, "grid_import_cap_kw"):
            self.cap = float(base_env.grid_import_cap_kw)
        else:
            self.cap = float(resolve_grid_import_cap(cfg))

        # Reward cost weights & Regime 2 C&I tariffs
        self.w_wear: float = float(
            self.params.get(
                "cost_wear_eur_kwh",
                getattr(base_env, "cost_wear_eur_kwh", cfg["cost_wear_eur_kwh"]),
            )
        )
        self.w_uns: float = float(
            self.params.get(
                "cost_unserved_eur_kwh",
                getattr(base_env, "cost_unserved_eur_kwh", cfg["cost_unserved_eur_kwh"]),
            )
        )
        self.demand_charge_eur_per_kw_week: float = float(
            self.params.get(
                "demand_charge_eur_per_kw_week",
                getattr(
                    base_env,
                    "demand_charge_eur_per_kw_week",
                    cfg["demand_charge_eur_per_kw_week"],
                ),
            )
        )
        self.industrial_trip_penalty_eur: float = float(
            self.params.get(
                "industrial_trip_penalty_eur",
                getattr(
                    base_env,
                    "industrial_trip_penalty_eur",
                    cfg["industrial_trip_penalty_eur"],
                ),
            )
        )
        self.zone_voll: np.ndarray = self._resolve_zone_voll(base_env)

        # Industrial motor and shedding parameters
        raw_motor = self.params.get(
            "motor_min_pct",
            getattr(base_env, "motor_min_pct", cfg["motor_min_pct"]),
        )
        raw_motor_f = float(raw_motor)
        self.motor_min_pct: float = raw_motor_f / 100.0 if raw_motor_f > 1.0 else raw_motor_f

        self.industrial_shutdown_threshold: float = float(
            self.params.get(
                "industrial_shutdown_threshold",
                getattr(
                    base_env,
                    "industrial_shutdown_threshold",
                    cfg["industrial_shutdown_threshold"],
                ),
            )
        )
        self.shed_mapping: str = str(
            self.params.get(
                "shed_mapping",
                getattr(base_env, "shed_mapping", "linear"),
            )
        )
        self.cap_clamp_tolerance_kw: float = float(
            self.params.get(
                "cap_clamp_tolerance_kw",
                getattr(base_env, "cap_clamp_tolerance_kw", cfg["cap_clamp_tolerance_kw"]),
            )
        )
        self.violation_tolerance_kw: float = float(
            self.params.get(
                "violation_tolerance_kw",
                getattr(base_env, "violation_tolerance_kw", cfg["violation_tolerance_kw"]),
            )
        )

    def _resolve_zone_voll(self, base_env: Optional[Any]) -> np.ndarray:
        """Resolve 5-element per-zone VoLL vector matching ZONE_NAMES."""
        raw_param = self.params.get("zone_voll_eur_kwh")
        if raw_param is not None:
            if isinstance(raw_param, dict):
                return np.array([float(raw_param[z]) for z in ZONE_NAMES], dtype=np.float64)
            arr = np.asarray(raw_param, dtype=np.float64)
            if arr.shape != (NUM_ZONES,):
                raise ValueError(f"zone_voll_eur_kwh must have length {NUM_ZONES}, got {arr.shape}")
            return arr
        if (
            "cost_unserved_eur_kwh" not in self.params
            and base_env is not None
            and hasattr(base_env, "get_zone_voll")
        ):
            return np.asarray(base_env.get_zone_voll(), dtype=np.float64)
        if base_env is not None and getattr(base_env, "zone_voll_eur_kwh", None) is not None:
            return np.asarray(base_env.get_zone_voll(), dtype=np.float64)
        return np.full(NUM_ZONES, self.w_uns, dtype=np.float64)

    @staticmethod
    def _resolve_curr_idx(
        base_env: Optional[Any], info: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        """Resolve current dataset index from environment state or info dict."""
        if base_env is not None and hasattr(base_env, "_step_idx"):
            return int(base_env._start_idx + base_env._step_idx)
        if info is not None and "data_idx" in info:
            return int(info["data_idx"])
        return None

    def _get_forecast(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
        K: int = 24,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate lookahead forecasts of PV, zone loads, and price for horizon K."""
        mode = self.forecast_mode
        target_env = env if env is not None else self.env
        base_env = getattr(target_env, "unwrapped", target_env)
        env_forecast_mode = getattr(base_env, "forecast_mode", None)

        use_env_horizon = (
            base_env is not None
            and hasattr(base_env, "get_horizon_data")
            and (
                mode == "causal_noisy"
                or mode == "oracle"
                or (mode == "persistence" and env_forecast_mode == "persistence")
            )
        )

        if use_env_horizon:
            curr_idx = self._resolve_curr_idx(base_env, info)
            if curr_idx is None:
                raise ValueError(
                    f"forecast_mode='{mode}' requires environment or info['data_idx'] "
                    "with access to get_horizon_data()."
                )
            pv_0 = np.array([max(0.0, float(obs[1]))], dtype=np.float32)
            loads_0 = np.maximum(0.0, np.array(obs[2:7], dtype=np.float32))[None, :]
            price_0 = np.array([max(0.0, float(obs[7])) / 1000.0], dtype=np.float32)
            if K > 1:
                h_data = base_env.get_horizon_data(
                    idx=curr_idx, horizon=K - 1, forecast_mode=mode
                )
                pv = np.concatenate([pv_0, h_data["pv_kw"].astype(np.float32)])
                loads = np.vstack([loads_0, h_data["zone_loads_kw"].astype(np.float32)])
                price_kwh = np.concatenate(
                    [price_0, (np.maximum(0.0, h_data["price_eur_mwh"]) / 1000.0).astype(np.float32)]
                )
            else:
                pv, loads, price_kwh = pv_0, loads_0, price_0
        elif mode == "causal_noisy":
            raise ValueError(
                "forecast_mode='causal_noisy' requires environment with get_horizon_data()."
            )
        elif mode == "persistence":
            pv_val = float(obs[1])
            loads_val = np.array(obs[2:7], dtype=np.float32)
            price_kwh_val = max(0.0, float(obs[7])) / 1000.0

            pv = np.full(K, pv_val, dtype=np.float32)
            loads = np.tile(loads_val, (K, 1)).astype(np.float32)
            price_kwh = np.full(K, price_kwh_val, dtype=np.float32)
        elif mode == "perfect":
            curr_idx = self._resolve_curr_idx(base_env, info)
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
            raise ValueError(
                f"Unknown forecast_mode: '{mode}'. "
                "Supported: 'perfect', 'oracle', 'persistence', 'causal_noisy'."
            )

        # Optional Gaussian forecast noise (future steps t >= 1 only; t=0 obs & cleared prices stay exact)
        if self.forecast_noise_std > 0.0 and pv.shape[0] > 1:
            pv = pv.copy()
            loads = loads.copy()
            if self.forecast_noise_relative:
                pv[1:] = np.maximum(
                    0.0,
                    pv[1:]
                    + self._rng.normal(
                        0.0, self.forecast_noise_std * (pv[1:] + 1e-3), size=pv[1:].shape
                    ),
                )
                loads[1:] = np.maximum(
                    0.0,
                    loads[1:]
                    + self._rng.normal(
                        0.0, self.forecast_noise_std * (loads[1:] + 1e-3), size=loads[1:].shape
                    ),
                )
            else:
                pv[1:] = np.maximum(
                    0.0,
                    pv[1:] + self._rng.normal(0.0, self.forecast_noise_std, size=pv[1:].shape),
                )
                loads[1:] = np.maximum(
                    0.0,
                    loads[1:] + self._rng.normal(0.0, self.forecast_noise_std, size=loads[1:].shape),
                )

        return pv.astype(np.float32), loads.astype(np.float32), price_kwh.astype(np.float32)

    def _normalize_zone_loads(self, loads: np.ndarray, K: int, milp_split: bool = False) -> np.ndarray:
        """Ensure lookahead loads have shape (K, NUM_ZONES)."""
        arr = np.asarray(loads, dtype=np.float64)
        if arr.ndim == 2:
            return arr
        out = np.zeros((K, NUM_ZONES), dtype=np.float64)
        if milp_split:
            out[:, 0] = arr * 0.8
            out[:, _INDUSTRIAL_ZONE_IDX] = arr * 0.2
        else:
            out[:, 0] = arr
        return out

    def _build_common_constraints(
        self,
        n_vars: int,
        K: int,
        stride: int,
        idx_u0: int,
        idx_imp: int,
        idx_peak: Optional[int],
        net: np.ndarray,
        soc_0: float,
    ) -> Tuple[list, list]:
        """Build power balance, SoC limits, and optional peak epigraph constraints (A x <= b)."""
        A_rows = []
        rhs = []

        # 1. Power balance: import + p_dis + pv + sum_z u_{t,z} >= load + p_ch
        #    <=> p_ch - p_dis - sum_z u_{t,z} - import <= -net
        for t in range(K):
            base = stride * t
            row = np.zeros(n_vars)
            row[base + 0] = 1.0
            row[base + 1] = -1.0
            row[base + idx_u0 : base + idx_u0 + NUM_ZONES] = -1.0
            row[base + idx_imp] = -1.0
            A_rows.append(row)
            rhs.append(-float(net[t]))

        # 2. State-of-Charge dynamics: soc_k in [soc_min, soc_max]
        for k in range(K):
            row_hi = np.zeros(n_vars)
            row_lo = np.zeros(n_vars)
            for t in range(k + 1):
                base = stride * t
                row_hi[base + 0] = self.eta_c * self.dt / self.c_bat
                row_hi[base + 1] = -self.dt / (self.eta_d * self.c_bat)
                row_lo[base + 0] = -self.eta_c * self.dt / self.c_bat
                row_lo[base + 1] = self.dt / (self.eta_d * self.c_bat)
            A_rows.append(row_hi)
            rhs.append(self.soc_max - soc_0)
            A_rows.append(row_lo)
            rhs.append(soc_0 - self.soc_min)

        # 3. Weekly peak demand epigraph: grid_import_t - P_peak <= 0
        if idx_peak is not None:
            for t in range(K):
                row = np.zeros(n_vars)
                row[stride * t + idx_imp] = 1.0
                row[idx_peak] = -1.0
                A_rows.append(row)
                rhs.append(0.0)

        return A_rows, rhs

    def _solve_lp(
        self,
        soc_0: float,
        pv: np.ndarray,
        loads: np.ndarray,
        price_kwh: np.ndarray,
        initial_peak: float = 0.0,
    ) -> Tuple[float, float, np.ndarray]:
        """Solve continuous HiGHS LP relaxation over horizon K.

        Variables per step t: [p_ch, p_dis, u_0..u_4, grid_import] (+ optional P_peak).
        Returns:
            Tuple of (p_ch_0, p_dis_0, u_zones_0)
        """
        K = len(pv)
        zone_loads = self._normalize_zone_loads(loads, K, milp_split=False)
        tot_loads = zone_loads.sum(axis=1)
        net = tot_loads - pv

        idx_u0 = 2
        idx_imp = 2 + NUM_ZONES
        stride = 3 + NUM_ZONES
        has_demand = self.demand_charge_eur_per_kw_week > 0.0
        n_vars = stride * K + (1 if has_demand else 0)
        idx_peak = stride * K if has_demand else None

        c = np.zeros(n_vars)
        bounds = []
        for t in range(K):
            base = stride * t
            c[base + 0] = self.w_wear * self.dt
            c[base + 1] = self.w_wear * self.dt
            c[base + idx_u0 : base + idx_u0 + NUM_ZONES] = self.zone_voll * self.dt
            c[base + idx_imp] = float(price_kwh[t]) * self.dt
            bounds.append((0.0, self.p_max))
            bounds.append((0.0, self.p_max))
            for z in range(NUM_ZONES):
                bounds.append((0.0, float(zone_loads[t, z])))
            bounds.append((0.0, self.cap))

        if idx_peak is not None:
            c[idx_peak] = self.demand_charge_eur_per_kw_week
            bounds.append((float(initial_peak), max(self.cap, float(initial_peak))))

        A_ub, b_ub = self._build_common_constraints(
            n_vars=n_vars,
            K=K,
            stride=stride,
            idx_u0=idx_u0,
            idx_imp=idx_imp,
            idx_peak=idx_peak,
            net=net,
            soc_0=soc_0,
        )

        res = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=bounds,
            method="highs",
        )
        if not res.success or res.x is None:
            return 0.0, 0.0, np.zeros(NUM_ZONES, dtype=np.float64)

        return (
            float(res.x[0]),
            float(res.x[1]),
            np.array(res.x[idx_u0 : idx_u0 + NUM_ZONES], dtype=np.float64),
        )

    def _industrial_safe_partial_limits(self) -> Tuple[str, np.float32, float]:
        """Return (mapping, max_linear_action_f32, safe_max_linear_shed_fraction)."""
        mapping = str(self.shed_mapping).strip().lower()
        max_linear_action = (
            np.float32(self.industrial_shutdown_threshold)
            if mapping in ("relu", "deadzone")
            else np.float32(2.0 * self.industrial_shutdown_threshold - 1.0)
        )
        safe_max_linear_frac = float(
            shed_action_to_fractions(
                [max_linear_action],
                shed_mapping=mapping,
                motor_min_pct=self.motor_min_pct,
                industrial_shutdown_threshold=self.industrial_shutdown_threshold,
                industrial_zone_idx=0,
            )[0]
        )
        return mapping, max_linear_action, min(1.0 - self.motor_min_pct, safe_max_linear_frac)

    def _solve_milp(
        self,
        soc_0: float,
        pv: np.ndarray,
        loads: np.ndarray,
        price_kwh: np.ndarray,
        initial_peak: float = 0.0,
        initial_tripped: bool = False,
    ) -> Tuple[float, float, np.ndarray]:
        """Solve exact HiGHS MILP enforcing piecewise industrial motor constraint.

        Variables per step t: [p_ch, p_dis, u_0..u_4, z_ind, grid_import] (+ optional y_trip, P_peak).
        Returns:
            Tuple of (p_ch_0, p_dis_0, u_zones_0)
        """
        K = len(pv)
        zone_loads = self._normalize_zone_loads(loads, K, milp_split=True)
        load_ind = zone_loads[:, _INDUSTRIAL_ZONE_IDX]
        tot_loads = zone_loads.sum(axis=1)
        net = tot_loads - pv

        has_demand = self.demand_charge_eur_per_kw_week > 0.0
        has_trip_pen = self.industrial_trip_penalty_eur > 0.0

        idx_u0 = 2
        idx_u_ind = idx_u0 + _INDUSTRIAL_ZONE_IDX
        idx_z = 2 + NUM_ZONES
        idx_imp = 3 + NUM_ZONES
        idx_trip = 4 + NUM_ZONES
        stride = 4 + NUM_ZONES + (1 if has_trip_pen else 0)
        n_vars = stride * K + (1 if has_demand else 0)
        idx_peak = stride * K if has_demand else None

        c = np.zeros(n_vars)
        integrality = np.zeros(n_vars)
        lb = np.zeros(n_vars)
        ub = np.zeros(n_vars)

        for t in range(K):
            base = stride * t
            c[base + 0] = self.w_wear * self.dt
            c[base + 1] = self.w_wear * self.dt
            c[base + idx_u0 : base + idx_u0 + NUM_ZONES] = self.zone_voll * self.dt
            c[base + idx_imp] = float(price_kwh[t]) * self.dt
            integrality[base + idx_z] = 1  # z_ind is binary

            ub[base + 0] = self.p_max
            ub[base + 1] = self.p_max
            ub[base + idx_u0 : base + idx_u0 + NUM_ZONES] = zone_loads[t]
            ub[base + idx_z] = 1.0 if load_ind[t] > 0.0 else 0.0
            ub[base + idx_imp] = self.cap

            if has_trip_pen:
                c[base + idx_trip] = self.industrial_trip_penalty_eur
                ub[base + idx_trip] = 1.0

        if idx_peak is not None:
            c[idx_peak] = self.demand_charge_eur_per_kw_week
            lb[idx_peak] = float(initial_peak)
            ub[idx_peak] = max(self.cap, float(initial_peak))

        bounds = Bounds(lb, ub)
        A_rows, rhs = self._build_common_constraints(
            n_vars=n_vars,
            K=K,
            stride=stride,
            idx_u0=idx_u0,
            idx_imp=idx_imp,
            idx_peak=idx_peak,
            net=net,
            soc_0=soc_0,
        )

        # Piecewise industrial motor operational constraints (matching float32 action feasibility):
        # u_ind - ((1 - safe_max_frac) * L_ind) * z_ind <= safe_max_frac * L_ind
        # -u_ind + L_ind * z_ind <= 0
        _, _, safe_max_frac = self._industrial_safe_partial_limits()
        for t in range(K):
            base = stride * t
            l_i = float(load_ind[t])

            row_ub = np.zeros(n_vars)
            row_ub[base + idx_u_ind] = 1.0
            row_ub[base + idx_z] = -(1.0 - safe_max_frac) * l_i
            A_rows.append(row_ub)
            rhs.append(safe_max_frac * l_i)

            row_lb = np.zeros(n_vars)
            row_lb[base + idx_u_ind] = -1.0
            row_lb[base + idx_z] = l_i
            A_rows.append(row_lb)
            rhs.append(0.0)

        # Industrial trip transition constraints: y_trip_t >= z_ind_t - z_ind_{t-1}
        if has_trip_pen:
            prev_z = 1.0 if initial_tripped else 0.0
            for t in range(K):
                base = stride * t
                row_tr = np.zeros(n_vars)
                row_tr[base + idx_z] = 1.0
                row_tr[base + idx_trip] = -1.0
                if t == 0:
                    rhs.append(prev_z)
                else:
                    row_tr[stride * (t - 1) + idx_z] = -1.0
                    rhs.append(0.0)
                A_rows.append(row_tr)

        lhs = [-np.inf] * len(rhs)
        constraints = LinearConstraint(np.array(A_rows), lhs, rhs)
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if not res.success or res.x is None:
            return 0.0, 0.0, np.zeros(NUM_ZONES, dtype=np.float64)

        return (
            float(res.x[0]),
            float(res.x[1]),
            np.array(res.x[idx_u0 : idx_u0 + NUM_ZONES], dtype=np.float64),
        )

    def _encode_zone_shed_actions(
        self,
        u_zones_0: np.ndarray,
        excess_kw: float,
        obs: np.ndarray,
    ) -> np.ndarray:
        """Convert per-zone unserved targets (with safety excess fallback) into [-1, 1]^5."""
        u_clean = np.where(u_zones_0 > self.cap_clamp_tolerance_kw, u_zones_0, 0.0)
        if np.allclose(self.zone_voll, self.zone_voll[0]):
            required_shed_kw = max(float(np.sum(u_clean)), excess_kw)
            return shed_actions_for(
                excess_kw=required_shed_kw,
                obs=obs,
                motor_min_fraction=self.motor_min_pct,
                industrial_shutdown_threshold=self.industrial_shutdown_threshold,
                shed_mapping=self.shed_mapping,
            )

        zone_demands = np.maximum(0.0, np.asarray(obs[2:7], dtype=np.float64))
        alloc = np.minimum(u_clean, zone_demands)
        additional = max(0.0, excess_kw - float(np.sum(alloc)))

        mapping, max_linear_action, safe_max_linear_frac = self._industrial_safe_partial_limits()
        max_linear_ind_shed = 1.0 - self.motor_min_pct

        if additional > 0.0:
            priority_zones = sorted(
                range(NUM_ZONES),
                key=lambda z: (float(self.zone_voll[z]), _SHED_TIE_RANK[z]),
            )
            for z in priority_zones:
                if additional <= 0.0:
                    break
                demand_z = float(zone_demands[z])
                if demand_z <= 0.0:
                    continue
                if z == _INDUSTRIAL_ZONE_IDX:
                    safe_max_kw = safe_max_linear_frac * demand_z
                    if alloc[z] + additional <= safe_max_kw:
                        alloc[z] += additional
                        additional = 0.0
                    else:
                        newly_shed = max(0.0, demand_z - alloc[z])
                        alloc[z] = demand_z
                        additional = max(0.0, additional - newly_shed)
                else:
                    take = min(additional, max(0.0, demand_z - alloc[z]))
                    alloc[z] += take
                    additional -= take

        zone_actions = np.full(NUM_ZONES, -1.0, dtype=np.float32)
        for z in range(NUM_ZONES):
            demand_z = float(zone_demands[z])
            shed_z = float(alloc[z])
            if shed_z <= 0.0 or demand_z <= 0.0:
                continue
            if z == _INDUSTRIAL_ZONE_IDX:
                safe_max_kw = safe_max_linear_frac * demand_z
                if shed_z > safe_max_kw + self.cap_clamp_tolerance_kw or max_linear_ind_shed <= 0.0:
                    zone_actions[z] = 1.0
                else:
                    linear_s = (
                        shed_z / (max_linear_ind_shed * demand_z)
                    ) * self.industrial_shutdown_threshold
                    raw_f32 = (
                        np.float32(linear_s)
                        if mapping in ("relu", "deadzone")
                        else np.float32(2.0 * linear_s - 1.0)
                    )
                    nudged = np.nextafter(raw_f32, np.float32(1.0))
                    zone_actions[z] = float(np.clip(nudged, -1.0, max_linear_action))
            else:
                if shed_z >= demand_z - self.cap_clamp_tolerance_kw:
                    zone_actions[z] = 1.0
                else:
                    frac = shed_z / demand_z
                    raw_f32 = (
                        np.float32(frac)
                        if mapping in ("relu", "deadzone")
                        else np.float32(2.0 * frac - 1.0)
                    )
                    nudged = np.nextafter(raw_f32, np.float32(1.0))
                    zone_actions[z] = float(np.clip(nudged, -1.0, 1.0))

        return zone_actions.astype(np.float32)

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

        if base_env is not None and hasattr(base_env, "_week_peak_grid_kw"):
            initial_peak = float(base_env._week_peak_grid_kw)
        elif info is not None and "week_peak_grid_kw" in info:
            initial_peak = float(info["week_peak_grid_kw"])
        else:
            initial_peak = 0.0

        if base_env is not None and hasattr(base_env, "_industrial_tripped"):
            initial_tripped = bool(base_env._industrial_tripped)
        elif info is not None and "industrial_tripped" in info:
            initial_tripped = bool(info["industrial_tripped"])
        else:
            initial_tripped = False

        pv_f, loads_f, price_f = self._get_forecast(obs, info=info, env=target_env, K=K)

        if self.use_milp:
            p_ch_0, p_dis_0, u_zones_0 = self._solve_milp(
                soc_0,
                pv_f,
                loads_f,
                price_f,
                initial_peak=initial_peak,
                initial_tripped=initial_tripped,
            )
        else:
            p_ch_0, p_dis_0, u_zones_0 = self._solve_lp(
                soc_0,
                pv_f,
                loads_f,
                price_f,
                initial_peak=initial_peak,
            )

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

        shed_acts = self._encode_zone_shed_actions(u_zones_0, excess_kw, obs)

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
