import json
import os
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
        self.grid_import_cap_kw = cfg["grid_import_cap_kw"]
        self.soc_reserve_floor = cfg["soc_reserve_floor"]

        # PV
        self.pv_area_m2 = cfg["pv_area_m2"]
        self.pv_efficiency = cfg["pv_efficiency"]
        self.pv_performance_ratio = cfg["pv_performance_ratio"]

        # Episode
        self.episode_length = cfg["episode_length_hours"]
        self.dt_hours = 1.0

        # Industrial motor floor
        self.motor_min_pct = cfg["motor_min_pct"] / 100.0
        # shed-action mapping. "linear" = s=(a+1)/2 (no dead zone).
        # "deadzone" = s=max(0,a) so a~0 means no shed (kills the a<0 dead zone).
        self.shed_mapping = cfg.get("shed_mapping", "linear")

        # Load data
        self._load_data(cfg["data_path"], cfg["split"])

        # Action: [battery_cmd, shed_res, shed_com, shed_stad, shed_street, shed_ind]
        self.action_dim = 1 + self.n_zones
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        # Observation: 17 dims
        self.obs_dim = 17
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
        local_dow = local_idx.dayofweek.values
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

        self._n_timesteps = len(df)

        # Split report
        report_path = os.path.join(data_path, "split_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                self._split_report = json.load(f)
        else:
            self._split_report = {}

    def _compute_pv_kw(self, idx: int) -> float:
        irr = self._irradiance[idx]
        return float(irr * self.pv_area_m2 * self.pv_efficiency * self.pv_performance_ratio / 1000.0)

    def _get_obs(self, idx: int) -> np.ndarray:
        return np.array([
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
        ], dtype=np.float32)

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
        # `relu` is the canonical name; `deadzone` is kept as an alias for the
        # archived S1 probe configs. Anything else -> linear (s=(a+1)/2).
        if self.shed_mapping in ("relu", "deadzone"):
            # s = max(0, a). a<=0 -> no shed; a>0 -> shed = a.
            shed_raw = np.maximum(0.0, action[1:])
        else:
            # linear (default): s = (a + 1) / 2  in [0, 1]
            shed_raw = (action[1:] + 1.0) / 2.0

        # Industrial: s = (a+1)/2. If s > 0.9 → off. Else shed = 0.3 * s / 0.9.
        s_ind = shed_raw[4]
        if s_ind > 0.9:
            ind_shed = 1.0  # Off
        else:
            ind_shed = (1.0 - self.motor_min_pct) * s_ind / 0.9  # 0.3 * s / 0.9
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
        bat_power_kw = battery_cmd * self.bat_max_power_kw
        if bat_power_kw > 0:
            max_charge = (self.bat_soc_max - self._soc) * self.bat_capacity_kwh / (
                self.bat_eta_charge * self.dt_hours)
            bat_power_kw = max(0.0, min(bat_power_kw, max_charge))
        else:
            max_discharge = (self._soc - self.bat_soc_min) * self.bat_capacity_kwh * (
                self.bat_eta_discharge / self.dt_hours)
            bat_power_kw = min(0.0, max(bat_power_kw, -max_discharge))

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

        # --- Constraint cost ---
        # The SINGLE Lagrangian constraint is grid import exceedance, in kW.
        # This matches the units in which d_main = 0.05 kW/step was calibrated.
        grid_exceedance_kw = max(0.0, grid_import_kw - self.grid_import_cap_kw)
        # SoC reserve deficit: DIAGNOSTIC ONLY, in kWh. NOT a constraint, and
        # deliberately excluded from info["cost"] so it can never enter Q_c or the dual update.
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
        }
        return obs, float(reward), terminated, truncated, info


def default_config() -> dict:
    """The env defaults (battery, costs, import cap, PV, episode length).

    Public so other modules can read a constant such as `grid_import_cap_kw`
    without instantiating an env (which would load a data split).
    """
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
        "cost_unserved_eur_kwh": 1.0,   # constant 1 EUR/kWh
        "cost_wear_eur_kwh": 0.02,      # constant 0.02 EUR/kWh throughput
        "grid_import_cap_kw": 368.15,   # Fixed P90 of train grid-only import
        "soc_reserve_floor": 0.20,
        "pv_area_m2": 1471.0,           # sized for ~250 kW peak
        "pv_efficiency": 0.20,
        "pv_performance_ratio": 0.85,
        "motor_min_pct": 70,
        "episode_length_hours": 168,
    }


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults = default_config()
    defaults.update(cfg)
    return defaults
