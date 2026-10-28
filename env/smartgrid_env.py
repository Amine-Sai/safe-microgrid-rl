import json
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def battery_power_kw(
    battery_command: float,
    soc: float,
    served_load_kw: float,
    pv_generation_kw: float,
    env,
    clamp: Optional[bool] = None,
    return_info: bool = False,
):
    """Post-battery power in kW: rate limit -> SoC limit -> cap clamp."""
    if clamp is None:
        clamp = getattr(env, "charge_clamp", True)
    cap_tolerance_kw = float(getattr(env, "cap_clamp_tolerance_kw", 1e-9))

    # 1. Rate limit
    bat_cmd = float(np.clip(battery_command, -1.0, 1.0))
    p = bat_cmd * env.bat_max_power_kw
    requested_charge_kw = max(0.0, p)

    # 2. SoC limit
    if p > 0:
        max_charge = (env.bat_soc_max - soc) * env.bat_capacity_kwh / (
            env.bat_eta_charge * env.dt_hours
        )
        p = max(0.0, min(p, max_charge))
    else:
        max_discharge = (soc - env.bat_soc_min) * env.bat_capacity_kwh * (
            env.bat_eta_discharge / env.dt_hours
        )
        p = min(0.0, max(p, -max_discharge))
    soc_bounded_charge_kw = max(0.0, p)

    # 3. Cap clamp (only restricts charging when clamp is True)
    is_charge_clipped_by_cap = False
    if clamp and p > 0:
        headroom = max(0.0, (env.grid_import_cap_kw - cap_tolerance_kw) + pv_generation_kw - served_load_kw)
        if p > headroom:
            p = max(0.0, headroom)
            is_charge_clipped_by_cap = True

    applied_charge_kw = max(0.0, p)
    charge_clipped_by_soc_kw = max(0.0, requested_charge_kw - soc_bounded_charge_kw)
    charge_clipped_by_cap_kw = max(0.0, soc_bounded_charge_kw - applied_charge_kw)

    if return_info:
        info = {
            "charge_requested_kw": float(requested_charge_kw),
            "charge_soc_bounded_kw": float(soc_bounded_charge_kw),
            "charge_applied_kw": float(applied_charge_kw),
            "charge_clipped_by_soc_kw": float(charge_clipped_by_soc_kw),
            "charge_clipped_by_cap_kw": float(charge_clipped_by_cap_kw),
            "is_charge_clipped_by_cap": bool(is_charge_clipped_by_cap),
        }
        return p, info
    return p


class SmartGridEnv(gym.Env):
    """Simulated microgrid environment using real time-series data."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[dict] = None):
        super().__init__()

        cfg = default_config()
        if config is not None:
            cfg.update(config)
        self.cfg = cfg

        self.zone_names = ["residential", "commercial", "stadium", "street", "industrial"]
        self.n_zones = len(self.zone_names)

        # Battery
        self.bat_capacity_kwh = cfg["battery_capacity_kwh"]
        self.bat_max_power_kw = cfg["battery_max_power_kw"]
        self.bat_eta_charge = cfg["battery_eta_charge"]
        self.bat_eta_discharge = cfg["battery_eta_discharge"]
        self.bat_soc_min = cfg["battery_soc_min"]
        self.bat_soc_max = cfg["battery_soc_max"]
        self.bat_soc_init = cfg["battery_soc_init"]

        # Reward (constant EUR/kWh penalties)
        self.cost_unserved_eur_kwh = cfg["cost_unserved_eur_kwh"]
        self.cost_wear_eur_kwh = cfg["cost_wear_eur_kwh"]

        # Constraints
        self.grid_import_cap_kw = resolve_grid_import_cap(cfg)
        self.soc_reserve_floor = cfg["soc_reserve_floor"]

        # PV
        self.pv_area_m2 = cfg["pv_area_m2"]
        self.pv_efficiency = cfg["pv_efficiency"]
        self.pv_performance_ratio = cfg["pv_performance_ratio"]

        # Episode
        self.episode_length = cfg["episode_length_hours"]
        self.dt_hours = float(cfg.get("dt_hours", 1.0))

        # industrial motor floor & shutdown threshold
        self.motor_min_pct = cfg["motor_min_pct"] / 100.0
        self.industrial_shutdown_threshold = float(cfg.get("industrial_shutdown_threshold", 0.90))

        # Charge clamp & numerical tolerances
        self.charge_clamp = bool(cfg.get("charge_clamp", True))
        self.cap_clamp_tolerance_kw = float(cfg.get("cap_clamp_tolerance_kw", 1e-9))
        self.violation_tolerance_kw = float(cfg.get("violation_tolerance_kw", 1e-4))
        self.action_saturation_threshold = float(cfg.get("action_saturation_threshold", 0.99))
        self.soc_pinned_tolerance = float(cfg.get("soc_pinned_tolerance", 0.005))
        # Shed-action mapping ("linear" or "relu"/"deadzone")
        self.shed_mapping = cfg.get("shed_mapping", "linear")

        # Forecast horizon
        self.forecast_horizon_hours = int(cfg.get("forecast_horizon_hours", 0))
        self.forecast_features_dim = 6 if self.forecast_horizon_hours > 0 else 0

        # Load data
        self._load_data(cfg["data_path"], cfg["split"])

        # Action: [battery_cmd, shed_res, shed_com, shed_stad, shed_street, shed_ind]
        self.action_dim = 1 + self.n_zones
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        # Observation dimension
        self.obs_dim = 17 + self.forecast_features_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Episode state
        self._step_idx = 0
        self._start_idx = 0
        self._soc = self.bat_soc_init
        self._rng = np.random.RandomState(cfg.get("seed", 42))

    def _load_data(self, data_path: str, split: str):
        """Load preprocessed data from parquet."""
        import pandas as pd

        path = os.path.join(data_path, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}. Run data/prepare.py first.")
        df = pd.read_parquet(path)

        # Numpy arrays for fast inner loop
        self._timestamps = df.index.values
        self._load_kw = df["load_kw"].values.astype(np.float32)
        self._price = df["price_eur_mwh"].values.astype(np.float32)
        self._irradiance = df["irradiance_wm2"].values.astype(np.float32)
        self._temperature = df["temperature_c"].values.astype(np.float32)
        self._humidity = df["humidity_pct"].values.astype(np.float32)
        self._pressure = df["pressure_msl_hpa"].values.astype(np.float32)

        # Zone loads (T, 5)
        self._zone_loads = np.stack([
            df[f"load_{z}_kw"].values.astype(np.float32)
            for z in self.zone_names
        ], axis=1)

        # Time features in Europe/Paris local time
        local_idx = df.index.tz_localize("UTC").tz_convert("Europe/Paris")
        local_hours = local_idx.hour.values + local_idx.minute.values / 60.0
        local_dow = local_idx.day_of_week.values
        self._sin_hour = np.sin(2 * np.pi * local_hours / 24).astype(np.float32)
        self._cos_hour = np.cos(2 * np.pi * local_hours / 24).astype(np.float32)
        self._sin_dow = np.sin(2 * np.pi * local_dow / 7).astype(np.float32)
        self._cos_dow = np.cos(2 * np.pi * local_dow / 7).astype(np.float32)

        # Event features: event_active and event_in_next_hour
        if "stadium_event" in df.columns:
            self._event_active = df["stadium_event"].values.astype(np.float32)
        else:
            self._event_active = np.zeros(len(df), dtype=np.float32)

        # event_in_next_hour: 1 if next timestep has an active event
        self._event_in_next_hour = np.zeros(len(df), dtype=np.float32)
        if len(self._event_active) > 1:
            self._event_in_next_hour[:-1] = self._event_active[1:]

        # Precompute series for fast step lookups
        self._pv_scale = np.float32(
            self.pv_area_m2 * self.pv_efficiency * self.pv_performance_ratio / 1000.0
        )
        self._pv_kw_series = self._irradiance * self._pv_scale
        self._net_load_kw_series = np.maximum(0.0, self._load_kw - self._pv_kw_series)

        self._n_timesteps = len(df)

        # Split report
        report_path = os.path.join(data_path, "split_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                self._split_report = json.load(f)
        else:
            self._split_report = {}

    def _compute_pv_kw(self, idx: int) -> float:
        return float(self._pv_kw_series[idx])

    def _get_obs(self, idx: int) -> np.ndarray:
        base_obs = [
            self._soc,
            self._compute_pv_kw(idx),
            self._zone_loads[idx, 0],
            self._zone_loads[idx, 1],
            self._zone_loads[idx, 2],
            self._zone_loads[idx, 3],
            self._zone_loads[idx, 4],
            self._price[idx],
            self._sin_hour[idx],
            self._cos_hour[idx],
            self._sin_dow[idx],
            self._cos_dow[idx],
            self._temperature[idx],
            self._humidity[idx],
            self._pressure[idx],
            self._event_active[idx],
            self._event_in_next_hour[idx],
        ]
        if self.forecast_horizon_hours <= 0:
            return np.array(base_obs, dtype=np.float32)

        # Lookahead window [idx + 1, idx + 1 + H] clamped to dataset end
        horizon = self.forecast_horizon_hours
        n = self._n_timesteps
        start = min(idx + 1, n - 1)
        end = min(idx + 1 + horizon, n)
        if start >= end:
            start = max(0, n - 1)
            end = n

        net_slice = self._net_load_kw_series[start:end]
        price_slice = self._price[start:end]

        forecast_obs = [
            float(self._net_load_kw_series[start]),
            float(np.max(net_slice)),
            float(np.mean(net_slice)),
            float(self._price[start]),
            float(np.min(price_slice)),
            float(np.max(price_slice)),
        ]
        return np.array(base_obs + forecast_obs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        max_start = self._n_timesteps - self.episode_length
        if max_start <= 0:
            raise ValueError(f"Data has {self._n_timesteps} steps but episode needs {self.episode_length}")
        if options and "start_idx" in options:
            self._start_idx = int(options["start_idx"])
        else:
            self._start_idx = self._rng.randint(0, max_start)
        self._step_idx = 0
        self._soc = self.bat_soc_init
        obs = self._get_obs(self._start_idx)
        return obs, {"data_idx": self._start_idx}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        data_idx = self._start_idx + self._step_idx

        # --- Parse action ---
        battery_cmd = float(action[0])
        # Shedding fractions
        if self.shed_mapping in ("relu", "deadzone"):
            shed_raw = np.maximum(0.0, action[1:])
        else:
            shed_raw = (action[1:] + 1.0) / 2.0

        # Industrial load logic
        s_ind = shed_raw[4]
        if s_ind > self.industrial_shutdown_threshold:
            ind_shed = 1.0
        else:
            ind_shed = (1.0 - self.motor_min_pct) * s_ind / self.industrial_shutdown_threshold
        shed_fractions = shed_raw.copy()
        shed_fractions[4] = ind_shed

        # --- Zone loads ---
        zone_demands = self._zone_loads[data_idx].copy()
        served = zone_demands * (1.0 - shed_fractions)
        unserved = zone_demands * shed_fractions
        total_served = float(served.sum())
        total_unserved = float(unserved.sum())

        # --- PV ---
        pv_kw = self._compute_pv_kw(data_idx)

        # --- Battery dynamics ---
        bat_power_kw, bat_info = battery_power_kw(
            battery_cmd, self._soc, total_served, pv_kw, self, return_info=True
        )

        if bat_power_kw > 0:
            delta_soc = (bat_power_kw * self.bat_eta_charge * self.dt_hours) / self.bat_capacity_kwh
        else:
            delta_soc = (bat_power_kw * self.dt_hours) / (self.bat_eta_discharge * self.bat_capacity_kwh)
        self._soc = np.clip(self._soc + delta_soc, self.bat_soc_min, self.bat_soc_max)

        # --- Power balance ---
        net_load = total_served + bat_power_kw - pv_kw
        grid_import_kw = max(0.0, net_load)
        pv_curtailed_kw = max(0.0, -net_load)

        # --- Reward (constant penalties, price clipped at 0) ---
        price_per_kwh = max(0.0, self._price[data_idx]) / 1000.0  # Clip negative prices
        energy_cost = grid_import_kw * self.dt_hours * price_per_kwh
        unserved_cost = total_unserved * self.dt_hours * self.cost_unserved_eur_kwh
        battery_wear = abs(bat_power_kw) * self.dt_hours * self.cost_wear_eur_kwh
        reward = -(energy_cost + unserved_cost + battery_wear)

        # Constraint cost: grid import exceedance in kW
        grid_exceedance_kw = max(0.0, grid_import_kw - self.grid_import_cap_kw)
        soc_deficit_kwh = max(0.0, self.soc_reserve_floor - self._soc) * self.bat_capacity_kwh
        cost = grid_exceedance_kw

        # --- Step forward ---
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_length

        if not truncated:
            obs = self._get_obs(self._start_idx + self._step_idx)
        else:
            obs = self._get_obs(data_idx)

        info = {
            "cost": float(cost),
            "grid_exceedance_kw": float(grid_exceedance_kw),
            "grid_import_kw": float(grid_import_kw),
            "grid_violation_kw": float(grid_exceedance_kw),
            "soc_violation": float(max(0.0, self.soc_reserve_floor - self._soc)),
            "soc_deficit_kwh": float(soc_deficit_kwh),
            "pv_kw": float(pv_kw),
            "pv_curtailed_kw": float(pv_curtailed_kw),
            "battery_power_kw": float(bat_power_kw),
            "soc": float(self._soc),
            "total_served_kw": float(total_served),
            "total_unserved_kw": float(total_unserved),
            "energy_cost": float(energy_cost),
            "unserved_cost": float(unserved_cost),
            "battery_wear_cost": float(battery_wear),
            "price_eur_mwh": float(self._price[data_idx]),
            "shed_fractions": shed_fractions.tolist(),
            "data_idx": int(data_idx),
            "is_violation": bool(grid_exceedance_kw > self.violation_tolerance_kw),
            "charge_requested_kw": bat_info["charge_requested_kw"],
            "charge_soc_bounded_kw": bat_info["charge_soc_bounded_kw"],
            "charge_applied_kw": bat_info["charge_applied_kw"],
            "charge_clipped_by_soc_kw": bat_info["charge_clipped_by_soc_kw"],
            "charge_clipped_by_cap_kw": bat_info["charge_clipped_by_cap_kw"],
            "is_charge_clipped_by_cap": bat_info["is_charge_clipped_by_cap"],
        }
        return obs, float(reward), terminated, truncated, info


def default_config() -> dict:
    """Default environment configuration."""
    return {
        "data_path": os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
        "split": "train",
        "seed": 42,
        "battery_capacity_kwh": 500.0,
        "battery_max_power_kw": 250.0,
        "battery_eta_charge": 0.95,
        "battery_eta_discharge": 0.95,
        "battery_soc_min": 0.10,
        "battery_soc_max": 0.90,
        "battery_soc_init": 0.50,
        "cost_unserved_eur_kwh": 1.0,
        "cost_wear_eur_kwh": 0.02,
        "grid_import_cap_kw": None,
        "soc_reserve_floor": 0.20,
        "pv_area_m2": 1471.0,
        "pv_efficiency": 0.20,
        "pv_performance_ratio": 0.85,
        "motor_min_pct": 70,
        "industrial_shutdown_threshold": 0.90,
        "episode_length_hours": 168,
        "dt_hours": 1.0,
        "charge_clamp": True,
        "cap_clamp_tolerance_kw": 1e-9,
        "violation_tolerance_kw": 1e-4,
        "action_saturation_threshold": 0.99,
        "soc_pinned_tolerance": 0.005,
        "forecast_horizon_hours": 0,
    }


def resolve_grid_import_cap(cfg: dict) -> float:
    """Resolve grid import cap: explicit config first, then split_report.json."""
    explicit = cfg.get("grid_import_cap_kw")
    if explicit is not None:
        return float(explicit)

    data_path = cfg.get("data_path") or os.path.join(
        os.path.dirname(__file__), "..", "data", "processed")
    report_path = os.path.join(data_path, "split_report.json")

    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        value = (report.get("grid_import_cap") or {}).get("value_kw")
        if value is not None:
            return float(value)

    raise ValueError(
        "grid_import_cap_kw is unset and %r has no 'grid_import_cap.value_kw'. "
        "Run `python -m data.prepare` or pass grid_import_cap_kw explicitly."
        % report_path
    )


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults = default_config()
    defaults.update(cfg)
    return defaults
