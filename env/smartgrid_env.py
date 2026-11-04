"""SmartGridEnv Gymnasium Environment for Safe Microgrid Reinforcement Learning.

Simulates an urban microgrid featuring rooftop solar photovoltaic (PV) generation,
a battery energy storage system (BESS), 5 heterogeneous demand zones (residential,
commercial, stadium, street lighting, industrial), and real-time electricity pricing.

Physical equations and operational constraints modeled:
- Rate limiting, SoC saturation bounds, and grid import cap clamping on battery power.
- State-of-charge updating with asymmetric charge/discharge efficiencies.
- Action space: battery dispatch and per-zone load shedding fractions.
- Non-linear piecewise industrial motor operational constraint (70% min run vs 90% shutdown).
- Microgrid power balance: grid import and PV curtailment.
- Multi-component economic reward (energy cost, unserved load penalty, battery degradation).
- Lagrangian safety constraint cost for grid import cap exceedances.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# =============================================================================
# === 1. Constants, Default Configuration Schema, and Zone Definitions ========
# =============================================================================

ZONE_NAMES: Tuple[str, ...] = (
    "residential",
    "commercial",
    "stadium",
    "street",
    "industrial",
)
NUM_ZONES: int = len(ZONE_NAMES)
BASE_OBS_DIM: int = 17
FORECAST_FEATURE_DIM: int = 6
ACTION_DIM: int = 1 + NUM_ZONES  # 1 battery command + 5 zone shedding actions


def default_config() -> Dict[str, Any]:
    """Default environment configuration dictionary.

    Returns:
        Dict mapping configuration parameter names to standard baseline values.
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


def resolve_grid_import_cap(cfg: Dict[str, Any]) -> float:
    """Resolve grid import cap in kW from explicit config or split report.

    Lookup Precedence:
        1. Explicit `grid_import_cap_kw` in configuration.
        2. `split_report.json` under `grid_import_cap.value_kw` in the data directory.

    Raises:
        ValueError: If neither source specifies a valid grid import cap.
    """
    explicit = cfg.get("grid_import_cap_kw")
    if explicit is not None:
        return float(explicit)

    data_path = cfg.get("data_path") or os.path.join(
        os.path.dirname(__file__), "..", "data", "processed"
    )
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


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration from file path and overlay onto default configuration.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Consolidated configuration dictionary.
    """
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    defaults = default_config()
    defaults.update(cfg)
    return defaults


# =============================================================================
# === 2. Standalone Physical Dynamics & Action Mapping Functions ==============
# =============================================================================


@overload
def battery_power_kw(
    battery_command: float,
    soc: float,
    served_load_kw: float,
    pv_generation_kw: float,
    env: Any,
    clamp: Optional[bool] = None,
    return_info: Literal[False] = False,
) -> float: ...


@overload
def battery_power_kw(
    battery_command: float,
    soc: float,
    served_load_kw: float,
    pv_generation_kw: float,
    env: Any,
    clamp: Optional[bool] = None,
    return_info: Literal[True] = ...,
) -> Tuple[float, Dict[str, Any]]: ...


def battery_power_kw(
    battery_command: float,
    soc: float,
    served_load_kw: float,
    pv_generation_kw: float,
    env: Any,
    clamp: Optional[bool] = None,
    return_info: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    r"""Calculate terminal battery power in kW subject to physical and grid constraints.

    Sign Convention:
        - $P_{\text{bat}} > 0$: Charging (battery absorbs power from microgrid/grid, increasing net load).
        - $P_{\text{bat}} < 0$: Discharging (battery injects power into microgrid, reducing net load).

    Clipping Pipeline (3 sequential physical stages):
        1. Inverter Rate Limiting:
           The continuous command $a_0 \in [-1, 1]$ is scaled by maximum inverter power $P_{\max}$:
           $$P_{\text{cmd}} = \text{clip}(a_0, -1.0, 1.0) \cdot P_{\max}$$
           $$P_{\text{ch, req}} = \max(0, P_{\text{cmd}})$$

        2. State-of-Charge (SoC) Bounds:
           Given battery capacity $C_{\text{bat}}$ (kWh), timestep duration $\Delta t$ (hours),
           and asymmetric efficiencies $\eta_c, \eta_d \in (0, 1]$:
           - If charging ($P > 0$):
             $$P_{\text{max-ch}} = \frac{(\text{SoC}_{\max} - \text{SoC}) C_{\text{bat}}}{\eta_c \Delta t}$$
             $$P = \max(0, \min(P, P_{\text{max-ch}}))$$
           - If discharging ($P \le 0$):
             $$P_{\text{max-dis}} = \frac{(\text{SoC} - \text{SoC}_{\min}) C_{\text{bat}} \eta_d}{\Delta t}$$
             $$P = \min(0, \max(P, -P_{\text{max-dis}}))$$

        3. Cap Clamping (Grid Headroom Limiting):
           When `clamp=True` and $P > 0$, charging power is bounded by the available grid import headroom
           to prevent battery charging from actively triggering a grid import cap exceedance:
           $$\text{Headroom} = \max\left(0, (P_{\text{cap}} - \epsilon_{\text{tol}}) + P_{\text{pv}} - P_{\text{load, served}}\right)$$
           $$P = \min(P, \text{Headroom})$$

    Args:
        battery_command: Normalized battery action in [-1.0, 1.0].
        soc: Current battery State-of-Charge fraction in [0.0, 1.0].
        served_load_kw: Total electrical load served across all zones in kW.
        pv_generation_kw: Current solar PV generation in kW.
        env: Environment or mock containing battery parameters:
             `bat_max_power_kw`, `bat_capacity_kwh`, `bat_soc_min`, `bat_soc_max`,
             `bat_eta_charge`, `bat_eta_discharge`, `dt_hours`, `grid_import_cap_kw`,
             `charge_clamp`, and `cap_clamp_tolerance_kw`.
        clamp: Whether to apply cap headroom clamping. If None, defaults to `env.charge_clamp`.
        return_info: If True, returns a tuple of (power_kw, diagnostic_dict).

    Returns:
        If return_info is False:
            Admissible terminal battery power in kW.
        If return_info is True:
            Tuple of (admissible battery power in kW, diagnostic info dict).
    """
    if clamp is None:
        clamp = getattr(env, "charge_clamp", True)
    cap_tolerance_kw = float(getattr(env, "cap_clamp_tolerance_kw", 1e-9))

    # Stage 1: Inverter rate limit
    bat_cmd = float(np.clip(battery_command, -1.0, 1.0))
    p = bat_cmd * env.bat_max_power_kw
    requested_charge_kw = max(0.0, p)

    # Stage 2: State-of-Charge (SoC) saturation limits
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

    # Stage 3: Cap clamp (restricts charging to available grid import headroom)
    is_charge_clipped_by_cap = False
    if clamp and p > 0:
        headroom = max(
            0.0,
            (env.grid_import_cap_kw - cap_tolerance_kw) + pv_generation_kw - served_load_kw,
        )
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


def shed_action_to_fractions(
    shed_actions: Union[np.ndarray, List[float]],
    shed_mapping: str = "linear",
    motor_min_pct: float = 0.70,
    industrial_shutdown_threshold: float = 0.90,
    industrial_zone_idx: int = 4,
) -> np.ndarray:
    r"""Map continuous load shedding actions in [-1.0, 1.0] to physical shedding fractions in [0.0, 1.0].

    Operational Rules:
    1. Continuous Mapping from action $a \in [-1, 1]$ to raw fraction $s_{\text{raw}} \in [0, 1]$:
       - Linear mapping (`shed_mapping == "linear"`):
         $$s_{\text{raw}} = \frac{a + 1.0}{2.0}$$
         (-1.0 -> 0% shed, 0.0 -> 50% shed, +1.0 -> 100% shed)
       - ReLU / deadzone mapping (`shed_mapping in ("relu", "deadzone")`):
         $$s_{\text{raw}} = \max(0.0, a)$$
         ($\le 0.0$ -> 0% shed, $> 0.0$ -> linear shed up to 100%)

    2. Non-linear Piecewise Industrial Motor Operational Constraint:
       Industrial motors cannot operate continuously below a minimum operating fraction
       $f_{\text{motor-min}}$ (default 0.70 = 70%), which restricts continuous partial shedding
       to at most $1.0 - f_{\text{motor-min}} = 0.30$ (30% shed). Complete emergency shutdown
       is allowed when the raw shed signal exceeds $\tau_{\text{shutdown}}$ (default 0.90):
       - If $s_{\text{raw}}[\text{ind}] > \tau_{\text{shutdown}}$:
         $$s_{\text{ind}} = 1.0 \quad \text{(100% load shed, motor completely shut down)}$$
       - If $s_{\text{raw}}[\text{ind}] \le \tau_{\text{shutdown}}$:
         $$s_{\text{ind}} = (1.0 - f_{\text{motor-min}}) \frac{s_{\text{raw}}[\text{ind}]}{\tau_{\text{shutdown}}} \in [0.0, 0.30]$$
       This guarantees that industrial load shedding is strictly within $[0.0, 0.30] \cup \{1.0\}$.

    Args:
        shed_actions: Continuous shedding action vector of shape (n_zones,) in [-1.0, 1.0].
        shed_mapping: Action mapping regime, either "linear" or "relu"/"deadzone".
        motor_min_pct: Minimum motor operating load fraction (e.g., 0.70) or percentage (70.0).
        industrial_shutdown_threshold: Action threshold in [0.0, 1.0] triggering complete shutdown.
        industrial_zone_idx: Index of the industrial zone (default: 4).

    Returns:
        np.ndarray of physical load shedding fractions in [0.0, 1.0] with shape (n_zones,).
    """
    actions_arr = np.asarray(shed_actions, dtype=np.float32)
    mapping = str(shed_mapping).strip().lower()

    # Continuous mapping
    if mapping in ("relu", "deadzone"):
        shed_raw = np.maximum(0.0, actions_arr)
    else:
        shed_raw = (actions_arr + 1.0) / 2.0

    motor_min_frac = float(motor_min_pct)
    if motor_min_frac > 1.0:
        motor_min_frac /= 100.0

    shed_fractions = np.array(shed_raw, dtype=np.float32)
    if 0 <= industrial_zone_idx < len(shed_fractions):
        s_ind = float(shed_raw[industrial_zone_idx])
        if s_ind > industrial_shutdown_threshold:
            ind_shed = 1.0
        else:
            ind_shed = (1.0 - motor_min_frac) * s_ind / industrial_shutdown_threshold
        shed_fractions[industrial_zone_idx] = float(ind_shed)

    return shed_fractions


# =============================================================================
# === 3. Gymnasium Microgrid Environment ======================================
# =============================================================================


class SmartGridEnv(gym.Env):
    r"""Simulated microgrid environment using real-world time-series data.

    The environment simulates an hourly discrete-time microgrid over multi-day or
    weekly episodes (typically 168 hours = 1 week).

    Physical Dynamics & Formulations:
    ---------------------------------
    1. Action Space:
       Continuous Box(-1.0, 1.0, shape=(6,), dtype=np.float32):
           [battery_cmd, shed_res, shed_com, shed_stad, shed_street, shed_ind]
       - action[0]: Battery dispatch command in [-1.0, 1.0] (-1.0 = discharge, +1.0 = charge).
       - action[1:6]: Raw shedding actions for the 5 zones:
             1: Residential, 2: Commercial, 3: Stadium, 4: Street lighting, 5: Industrial.

    2. Shedding Action Mapping & Industrial Motor Constraint:
       Handled by standalone `shed_action_to_fractions()`, enforcing that industrial
       load shedding is in $[0.0, 0.30] \cup \{1.0\}$.

    3. Battery Dynamics:
       At each step $t$ with duration $\Delta t$ (hours):
       - If charging ($P_{\text{bat}} > 0$):
         $$\Delta \text{SoC} = \frac{P_{\text{bat}} \eta_c \Delta t}{C_{\text{bat}}}$$
       - If discharging ($P_{\text{bat}} < 0$):
         $$\Delta \text{SoC} = \frac{P_{\text{bat}} \Delta t}{\eta_d C_{\text{bat}}}$$
       - Stored energy update:
         $$E_{\text{bat}, t+1} = E_{\text{bat}, t} + \left(\eta_c P_{\text{ch}} - \frac{1}{\eta_d} P_{\text{dis}}\right) \Delta t$$
         $$\text{SoC}_{t+1} = \text{clip}\left(\text{SoC}_t + \Delta \text{SoC}, \text{SoC}_{\min}, \text{SoC}_{\max}\right)$$

    4. Power Balance Equations:
       Total served demand and unserved load:
       $$P_{\text{load, served}} = \sum_i L_{\text{demand}, i} (1 - s_i)$$
       $$P_{\text{unserved}} = \sum_i L_{\text{demand}, i} s_i$$
       Net microgrid load:
       $$P_{\text{net}} = P_{\text{load, served}} + P_{\text{bat}} - P_{\text{pv}}$$
       $$P_{\text{grid, import}} = \max(0, P_{\text{net}})$$
       $$P_{\text{pv, curtailed}} = \max(0, -P_{\text{net}})$$

    5. Economic Cost Formulation:
       Spot electricity prices are clipped at zero to prevent artificial load dumping:
       $$\lambda_{\text{grid}, t} = \frac{\max(0, \text{price}_{\text{EUR/MWh}, t})}{1000.0} \quad [\text{EUR/kWh}]$$
       Multi-component operational step cost:
       $$C_t = \lambda_{\text{grid}, t} P_{\text{grid, import}} \Delta t + c_{\text{wear}} |P_{\text{bat}}| \Delta t + c_{\text{unserved}} P_{\text{unserved}} \Delta t$$
       Step reward:
       $$r_t = -C_t$$

    6. Safety Constraint Exceedance:
       Grid import cap exceedance:
       $$g_t = \max(0, P_{\text{grid, import}} - P_{\text{cap}}) \quad [\text{kW}]$$
       $$\text{cost}_t = g_t \quad [\text{kW}]$$
       SoC reserve floor deficit (diagnostic metric, does not enter step constraint cost):
       $$\text{soc\_deficit\_kwh} = \max(0, \text{SoC}_{\text{reserve}} - \text{SoC}) \cdot C_{\text{bat}}$$

    7. Observation Space:
       - 17-D Base observation:
             [SoC, P_pv, L_res, L_com, L_stad, L_street, L_ind, price,
              sin_hour, cos_hour, sin_dow, cos_dow, temp, humidity, pressure,
              event_active, event_in_next_hour]
       - 23-D Observation (when forecast_horizon_hours > 0):
             [... 17 base features ...,
              next_net_load, max_net_load, mean_net_load,
              next_price, min_price, max_price]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the microgrid environment with specified or default configuration."""
        super().__init__()

        cfg = default_config()
        if config is not None:
            cfg.update(config)
        self.cfg: Dict[str, Any] = cfg

        self.zone_names: List[str] = list(ZONE_NAMES)
        self.n_zones: int = len(self.zone_names)

        # Battery parameters
        self.bat_capacity_kwh: float = float(cfg["battery_capacity_kwh"])
        self.bat_max_power_kw: float = float(cfg["battery_max_power_kw"])
        self.bat_eta_charge: float = float(cfg["battery_eta_charge"])
        self.bat_eta_discharge: float = float(cfg["battery_eta_discharge"])
        self.bat_soc_min: float = float(cfg["battery_soc_min"])
        self.bat_soc_max: float = float(cfg["battery_soc_max"])
        self.bat_soc_init: float = float(cfg["battery_soc_init"])

        # Reward cost weights (constant EUR/kWh penalties)
        self.cost_unserved_eur_kwh: float = float(cfg["cost_unserved_eur_kwh"])
        self.cost_wear_eur_kwh: float = float(cfg["cost_wear_eur_kwh"])

        # Safety constraints
        self.grid_import_cap_kw: float = resolve_grid_import_cap(cfg)
        self.soc_reserve_floor: float = float(cfg["soc_reserve_floor"])

        # Photovoltaic (PV) system parameters
        self.pv_area_m2: float = float(cfg["pv_area_m2"])
        self.pv_efficiency: float = float(cfg["pv_efficiency"])
        self.pv_performance_ratio: float = float(cfg["pv_performance_ratio"])

        # Episode and time discretization
        self.episode_length: int = int(cfg["episode_length_hours"])
        self.dt_hours: float = float(cfg.get("dt_hours", 1.0))

        # Industrial motor operating floor & shutdown threshold
        raw_motor_min = float(cfg["motor_min_pct"])
        self.motor_min_pct: float = raw_motor_min / 100.0 if raw_motor_min > 1.0 else raw_motor_min
        self.industrial_shutdown_threshold: float = float(
            cfg.get("industrial_shutdown_threshold", 0.90)
        )

        # Charge clamp & numerical tolerances
        self.charge_clamp: bool = bool(cfg.get("charge_clamp", True))
        self.cap_clamp_tolerance_kw: float = float(cfg.get("cap_clamp_tolerance_kw", 1e-9))
        self.violation_tolerance_kw: float = float(cfg.get("violation_tolerance_kw", 1e-4))
        self.action_saturation_threshold: float = float(
            cfg.get("action_saturation_threshold", 0.99)
        )
        self.soc_pinned_tolerance: float = float(cfg.get("soc_pinned_tolerance", 0.005))

        # Shed-action mapping ("linear" or "relu"/"deadzone")
        self.shed_mapping: str = str(cfg.get("shed_mapping", "linear"))

        # Lookahead forecast horizon
        self.forecast_horizon_hours: int = int(cfg.get("forecast_horizon_hours", 0))
        self.forecast_features_dim: int = (
            FORECAST_FEATURE_DIM if self.forecast_horizon_hours > 0 else 0
        )

        # Validate configuration schema
        self._validate_config()

        # Load time-series data
        self._load_data(cfg["data_path"], cfg["split"])

        # Action: [battery_cmd, shed_res, shed_com, shed_stad, shed_street, shed_ind]
        self.action_dim: int = 1 + self.n_zones
        self.action_space: spaces.Box = spaces.Box(
            -1.0, 1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # Observation dimension: 17 base features + optional 6 forecast horizon features
        self.obs_dim: int = BASE_OBS_DIM + self.forecast_features_dim
        self.observation_space: spaces.Box = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Episode state variables
        self._step_idx: int = 0
        self._start_idx: int = 0
        self._soc: float = self.bat_soc_init
        self._rng: np.random.RandomState = np.random.RandomState(cfg.get("seed", 42))

    def _validate_config(self) -> None:
        """Validate environment configuration parameters."""
        if self.bat_capacity_kwh <= 0:
            raise ValueError(f"battery_capacity_kwh must be positive, got {self.bat_capacity_kwh}")
        if self.bat_max_power_kw < 0:
            raise ValueError(f"battery_max_power_kw must be non-negative, got {self.bat_max_power_kw}")
        if not (0.0 < self.bat_eta_charge <= 1.0):
            raise ValueError(f"battery_eta_charge must be in (0, 1], got {self.bat_eta_charge}")
        if not (0.0 < self.bat_eta_discharge <= 1.0):
            raise ValueError(f"battery_eta_discharge must be in (0, 1], got {self.bat_eta_discharge}")
        if not (0.0 <= self.bat_soc_min < self.bat_soc_max <= 1.0):
            raise ValueError(
                f"SoC bounds invalid: 0 <= soc_min ({self.bat_soc_min}) < "
                f"soc_max ({self.bat_soc_max}) <= 1 required"
            )
        if not (self.bat_soc_min <= self.bat_soc_init <= self.bat_soc_max):
            raise ValueError(
                f"Initial SoC {self.bat_soc_init} must be in [{self.bat_soc_min}, {self.bat_soc_max}]"
            )
        if self.episode_length <= 0:
            raise ValueError(f"episode_length_hours must be positive, got {self.episode_length}")
        if self.dt_hours <= 0:
            raise ValueError(f"dt_hours must be positive, got {self.dt_hours}")
        if self.grid_import_cap_kw < 0:
            raise ValueError(f"grid_import_cap_kw must be non-negative, got {self.grid_import_cap_kw}")
        if self.shed_mapping not in ("linear", "relu", "deadzone"):
            raise ValueError(
                f"Unsupported shed_mapping '{self.shed_mapping}'. Expected 'linear', 'relu', or 'deadzone'."
            )

    # =========================================================================
    # === Data Ingestion & Observation Building ===============================
    # =========================================================================

    def _load_data(self, data_path: str, split: str) -> None:
        """Load preprocessed time-series dataset from parquet.

        Extracts continuous time series into contiguous NumPy float32 arrays
        for high-performance inner-loop stepping, precomputes solar PV generation,
        Europe/Paris cyclic time encodings, and lookahead event flags.
        """
        import pandas as pd

        path = os.path.join(data_path, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}. Run data/prepare.py first.")
        df = pd.read_parquet(path)

        # Contiguous NumPy arrays for fast inner-loop stepping
        self._timestamps = df.index.values
        self._load_kw: np.ndarray = df["load_kw"].values.astype(np.float32)
        self._price: np.ndarray = df["price_eur_mwh"].values.astype(np.float32)
        self._irradiance: np.ndarray = df["irradiance_wm2"].values.astype(np.float32)
        self._temperature: np.ndarray = df["temperature_c"].values.astype(np.float32)
        self._humidity: np.ndarray = df["humidity_pct"].values.astype(np.float32)
        self._pressure: np.ndarray = df["pressure_msl_hpa"].values.astype(np.float32)

        # Zone loads (T, 5)
        self._zone_loads: np.ndarray = np.stack(
            [df[f"load_{z}_kw"].values.astype(np.float32) for z in self.zone_names],
            axis=1,
        )

        # Time features in Europe/Paris local time (accounting for CET/CEST daylight savings)
        local_idx = df.index.tz_localize("UTC").tz_convert("Europe/Paris")
        local_hours = local_idx.hour.values + local_idx.minute.values / 60.0
        local_dow = local_idx.day_of_week.values
        self._sin_hour: np.ndarray = np.sin(2 * np.pi * local_hours / 24).astype(np.float32)
        self._cos_hour: np.ndarray = np.cos(2 * np.pi * local_hours / 24).astype(np.float32)
        self._sin_dow: np.ndarray = np.sin(2 * np.pi * local_dow / 7).astype(np.float32)
        self._cos_dow: np.ndarray = np.cos(2 * np.pi * local_dow / 7).astype(np.float32)

        # Event features: event_active and event_in_next_hour
        if "stadium_event" in df.columns:
            self._event_active: np.ndarray = df["stadium_event"].values.astype(np.float32)
        else:
            self._event_active = np.zeros(len(df), dtype=np.float32)

        # event_in_next_hour: 1 if next timestep has an active stadium event
        self._event_in_next_hour: np.ndarray = np.zeros(len(df), dtype=np.float32)
        if len(self._event_active) > 1:
            self._event_in_next_hour[:-1] = self._event_active[1:]

        # Precompute PV and net load series for fast step lookups
        self._pv_scale: np.float32 = np.float32(
            self.pv_area_m2 * self.pv_efficiency * self.pv_performance_ratio / 1000.0
        )
        self._pv_kw_series: np.ndarray = self._irradiance * self._pv_scale
        self._net_load_kw_series: np.ndarray = np.maximum(
            0.0, self._load_kw - self._pv_kw_series
        )

        self._n_timesteps: int = len(df)

        # Metadata split report
        report_path = os.path.join(data_path, "split_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                self._split_report: Dict[str, Any] = json.load(f)
        else:
            self._split_report = {}

    def _compute_pv_kw(self, idx: int) -> float:
        """Compute solar photovoltaic generation in kW for timestep index."""
        return float(self._pv_kw_series[idx])

    def _get_obs(self, idx: int) -> np.ndarray:
        """Construct observation vector for timestep index.

        Layout:
            [0]: SoC fraction [0, 1]
            [1]: PV generation in kW
            [2..6]: Zone loads in kW (residential, commercial, stadium, street, industrial)
            [7]: Electricity spot price in EUR/MWh
            [8, 9]: Cyclical hour sin/cos
            [10, 11]: Cyclical day of week sin/cos
            [12]: Temperature in °C
            [13]: Humidity in %
            [14]: Atmospheric pressure in hPa
            [15]: Binary stadium event active flag
            [16]: Binary stadium event in next hour flag
            [17..22] (optional, when forecast_horizon_hours > 0):
                17: Next timestep net load in kW
                18: Max net load over lookahead window in kW
                19: Mean net load over lookahead window in kW
                20: Next timestep electricity price in EUR/MWh
                21: Min price over lookahead window in EUR/MWh
                22: Max price over lookahead window in EUR/MWh
        """
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

        # Lookahead window [idx + 1, idx + 1 + H] clamped to dataset boundaries
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

    # =========================================================================
    # === Step & Dynamics Execution ===========================================
    # =========================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new episode.

        Args:
            seed: Optional integer seed for reproducible episode start sampling.
            options: Optional dictionary, supporting explicit "start_idx".

        Returns:
            Tuple of (initial_observation, info_dict_with_data_idx).
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        max_start = self._n_timesteps - self.episode_length
        if max_start <= 0:
            raise ValueError(
                f"Data has {self._n_timesteps} steps but episode needs {self.episode_length}"
            )
        if options and "start_idx" in options:
            self._start_idx = int(options["start_idx"])
        else:
            self._start_idx = int(self._rng.randint(0, max_start))
        self._step_idx = 0
        self._soc = self.bat_soc_init
        obs = self._get_obs(self._start_idx)
        return obs, {"data_idx": self._start_idx}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        r"""Execute one simulation step of microgrid operations.

        Args:
            action: 6-dimensional continuous control vector in [-1.0, 1.0]:
                [battery_cmd, shed_res, shed_com, shed_stad, shed_street, shed_ind]

        Returns:
            Tuple of (obs, reward, terminated, truncated, info):
                - obs: Current observation vector (17-dim base / 23-dim forecast).
                - reward: Step economic reward (negative sum of energy, unserved, wear costs).
                - terminated: Always False (episodes terminate by horizon truncation).
                - truncated: True when episode_length steps have completed.
                - info: Detailed step telemetry, physical breakdown, and constraint metrics.
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        data_idx = self._start_idx + self._step_idx

        # --- 1. Action parsing & load shedding dispatch ---
        battery_cmd = float(action[0])
        shed_fractions = shed_action_to_fractions(
            action[1:],
            shed_mapping=self.shed_mapping,
            motor_min_pct=self.motor_min_pct,
            industrial_shutdown_threshold=self.industrial_shutdown_threshold,
            industrial_zone_idx=4,
        )

        # --- 2. Demand disaggregation: served & unserved power ---
        zone_demands = self._zone_loads[data_idx].copy()
        served = zone_demands * (1.0 - shed_fractions)
        unserved = zone_demands * shed_fractions
        total_served = float(served.sum())
        total_unserved = float(unserved.sum())

        # --- 3. Solar PV generation ---
        pv_kw = self._compute_pv_kw(data_idx)

        # --- 4. Battery dynamics & clipping ---
        bat_power_kw, bat_info = battery_power_kw(
            battery_cmd, self._soc, total_served, pv_kw, self, return_info=True
        )

        # State-of-Charge update with asymmetric efficiency:
        # E_{t+1} = E_t + (eta_c * P_ch - (1 / eta_d) * P_dis) * dt
        if bat_power_kw > 0:
            delta_soc = (bat_power_kw * self.bat_eta_charge * self.dt_hours) / self.bat_capacity_kwh
        else:
            delta_soc = (bat_power_kw * self.dt_hours) / (
                self.bat_eta_discharge * self.bat_capacity_kwh
            )
        self._soc = float(np.clip(self._soc + delta_soc, self.bat_soc_min, self.bat_soc_max))

        # --- 5. Microgrid power balance ---
        # P_net = P_load,served + P_bat - P_pv
        net_load = total_served + bat_power_kw - pv_kw
        grid_import_kw = max(0.0, net_load)
        pv_curtailed_kw = max(0.0, -net_load)

        # --- 6. Economic step cost & reward formulation ---
        # C_t = lambda_grid * P_import * dt + c_wear * |P_bat| * dt + c_unserved * P_unserved * dt
        price_per_kwh = max(0.0, self._price[data_idx]) / 1000.0  # Clip negative spot prices
        energy_cost = grid_import_kw * self.dt_hours * price_per_kwh
        unserved_cost = total_unserved * self.dt_hours * self.cost_unserved_eur_kwh
        battery_wear = abs(bat_power_kw) * self.dt_hours * self.cost_wear_eur_kwh
        reward = -(energy_cost + unserved_cost + battery_wear)

        # --- 7. Safety constraint metrics ---
        # g_t = max(0, P_grid,import - P_cap)
        grid_exceedance_kw = max(0.0, grid_import_kw - self.grid_import_cap_kw)
        soc_deficit_kwh = max(0.0, self.soc_reserve_floor - self._soc) * self.bat_capacity_kwh
        cost = grid_exceedance_kw

        # --- 8. Step advancement & truncation ---
        self._step_idx += 1
        terminated = False
        truncated = bool(self._step_idx >= self.episode_length)

        if not truncated:
            obs = self._get_obs(self._start_idx + self._step_idx)
        else:
            obs = self._get_obs(data_idx)

        # --- 9. Diagnostic info dictionary construction ---
        info: Dict[str, Any] = {
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
            "charge_requested_kw": float(bat_info["charge_requested_kw"]),
            "charge_soc_bounded_kw": float(bat_info["charge_soc_bounded_kw"]),
            "charge_applied_kw": float(bat_info["charge_applied_kw"]),
            "charge_clipped_by_soc_kw": float(bat_info["charge_clipped_by_soc_kw"]),
            "charge_clipped_by_cap_kw": float(bat_info["charge_clipped_by_cap_kw"]),
            "is_charge_clipped_by_cap": bool(bat_info["is_charge_clipped_by_cap"]),
        }
        return obs, float(reward), terminated, truncated, info
