from typing import Optional
import numpy as np
from env.smartgrid_env import battery_power_kw

# (obs index of zone load, action dim of its shed, industrial flag)
# Order: street -> stadium -> commercial -> residential -> industrial (PSM').
SHED_ORDER = (
    {"zone": "street", "obs_idx": 5, "act_dim": 4, "industrial": False},
    {"zone": "stadium", "obs_idx": 4, "act_dim": 3, "industrial": False},
    {"zone": "commercial", "obs_idx": 3, "act_dim": 2, "industrial": False},
    {"zone": "residential", "obs_idx": 2, "act_dim": 1, "industrial": False},
    {"zone": "industrial", "obs_idx": 6, "act_dim": 5, "industrial": True},
)


def predict_battery_kw(
    batt_cmd: float,
    soc: float,
    env,
    served_kw: Optional[float] = None,
    pv_kw: float = 0.0,
    clamp: Optional[bool] = None,
) -> float:
    """Post-battery power in kW, mirroring the env's clipping without mutating it.

    Positive means charging (raises grid import), negative means discharging.
    Delegates to the shared battery_power_kw in env.smartgrid_env.
    """
    if served_kw is None:
        clamp = False if clamp is None else clamp
        served_kw = 0.0
    return battery_power_kw(batt_cmd, soc, served_kw, pv_kw, env, clamp=clamp)


def shed_actions_for(
    excess_kw: float,
    obs: np.ndarray,
    motor_min_fraction: float = 0.70,
    industrial_shutdown_threshold: float = 0.90,
) -> np.ndarray:
    """Minimum-excess shed actions (5 dims, linear mapping a = 2s-1), PSM' order."""
    zone_shed_kw = np.zeros(5, dtype=np.float64)
    remaining_excess_kw = max(0.0, float(excess_kw))
    max_linear_industrial_shed_fraction = 1.0 - motor_min_fraction

    for entry in SHED_ORDER:
        if remaining_excess_kw <= 0.0:
            break
        zone_demand_kw = max(0.0, float(obs[entry["obs_idx"]]))
        if zone_demand_kw <= 0.0:
            continue
        zone_index = entry["obs_idx"] - 2  # zone position 0..4 == shed dim 0..4

        if entry["industrial"]:
            # partial cover within the linear motor band, otherwise fully off
            max_linear_kw = max_linear_industrial_shed_fraction * zone_demand_kw
            if remaining_excess_kw <= max_linear_kw:
                zone_shed_kw[zone_index] = remaining_excess_kw
                remaining_excess_kw = 0.0
            else:
                zone_shed_kw[zone_index] = zone_demand_kw
                remaining_excess_kw -= zone_demand_kw
        else:
            allocated_shed_kw = min(remaining_excess_kw, zone_demand_kw)
            zone_shed_kw[zone_index] = allocated_shed_kw
            remaining_excess_kw -= allocated_shed_kw

    # Invert to linear-mapping actions: s = (a + 1) / 2 -> a = 2s - 1.
    # Nudge active continuous actions up by 1 float32 ULP so float32 round-trip
    # in SmartGridEnv.step never under-sheds due to floating-point truncation.
    zone_shed_actions = np.full(5, -1.0, dtype=np.float32)  # s = 0 default
    for entry in SHED_ORDER:
        zone_index = entry["obs_idx"] - 2
        zone_demand_kw = max(0.0, float(obs[entry["obs_idx"]]))
        allocated_shed_kw = zone_shed_kw[zone_index]
        if allocated_shed_kw <= 0.0 or zone_demand_kw <= 0.0:
            continue

        action_dim_idx = entry["act_dim"] - 1
        if entry["industrial"]:
            if allocated_shed_kw >= zone_demand_kw - 1e-9:
                zone_shed_actions[action_dim_idx] = 1.0  # fully off
            else:
                linear_s = (
                    (allocated_shed_kw / (max_linear_industrial_shed_fraction * zone_demand_kw))
                    * industrial_shutdown_threshold
                )
                raw_action_f32 = np.float32(2.0 * linear_s - 1.0)
                nudged_action = np.nextafter(raw_action_f32, np.float32(1.0))
                max_linear_action = float(2.0 * industrial_shutdown_threshold - 1.0)
                zone_shed_actions[action_dim_idx] = float(
                    np.clip(nudged_action, -1.0, max_linear_action)
                )
        else:
            raw_action_f32 = np.float32(2.0 * (allocated_shed_kw / zone_demand_kw) - 1.0)
            nudged_action = np.nextafter(raw_action_f32, np.float32(1.0))
            zone_shed_actions[action_dim_idx] = float(np.clip(nudged_action, -1.0, 1.0))

    return zone_shed_actions.astype(np.float32)


class CapShield:
    """Wraps a SmartGridEnv. step(battery_1d) -> (obs, r, term, trunc, info).

    The obs returned is the inner env's obs unchanged. The info dict is the inner
    env's, plus `shield_shed_kw` (total kW shed by the shield this step).
    """

    def __init__(self, env, cap_kw: float = None):
        self.env = env
        # Default to the wrapped env's cap so the shield can never shave to a
        # different threshold than the one the env is judged against.
        self.cap_kw = env.grid_import_cap_kw if cap_kw is None else float(cap_kw)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def compose_action(self, battery_command_1d, obs: np.ndarray) -> np.ndarray:
        """1-D battery command + obs -> full 6-D env action with minimum-excess shed."""
        battery_command = float(np.asarray(battery_command_1d, dtype=np.float32).ravel()[0])
        soc = float(obs[0])
        pv_kw = float(obs[1])
        total_demand_kw = float(sum(float(obs[i]) for i in (2, 3, 4, 5, 6)))

        clamp_enabled = getattr(self.env, "charge_clamp", True)
        motor_min_pct = getattr(self.env, "motor_min_pct", 0.70)
        ind_off_thresh = getattr(self.env, "industrial_shutdown_threshold", 0.90)

        if not clamp_enabled:
            batt_kw = predict_battery_kw(battery_command, soc, self.env)
            predicted_import_kw = max(0.0, total_demand_kw - pv_kw + batt_kw)
            excess_kw = predicted_import_kw - self.cap_kw
            shed_actions = shed_actions_for(
                excess_kw, obs, motor_min_fraction=motor_min_pct, industrial_shutdown_threshold=ind_off_thresh
            )
            full_action = np.concatenate([[np.clip(battery_command, -1.0, 1.0)], shed_actions]).astype(np.float32)
            return full_action

        # Charge-clamped composition:
        # Pre-clamp battery power against total un-shed demand. When demand - pv >= cap,
        # available import headroom is zero so positive charge is clamped to zero,
        # preventing unnecessary load shedding to accommodate charging.
        batt_kw = battery_power_kw(battery_command, soc, total_demand_kw, pv_kw, self.env, clamp=True)
        predicted_import_kw = max(0.0, total_demand_kw - pv_kw + batt_kw)
        excess_kw = predicted_import_kw - self.cap_kw

        shed_actions = shed_actions_for(
            excess_kw, obs, motor_min_fraction=motor_min_pct, industrial_shutdown_threshold=ind_off_thresh
        )
        full_action = np.concatenate([[np.clip(battery_command, -1.0, 1.0)], shed_actions]).astype(np.float32)
        return full_action

    def step(self, battery_command_1d):
        """Drop-in env step with a 1-D battery action (uses the obs from reset/last step)."""
        if not hasattr(self, "_last_obs"):
            raise RuntimeError("CapShield.step called before reset")
        return self.step_with_obs(battery_command_1d, self._last_obs)

    def step_with_obs(self, battery_command_1d, obs: np.ndarray):
        full_action = self.compose_action(battery_command_1d, obs)
        next_obs, reward, terminated, truncated, info = self.env.step(full_action)
        self._last_obs = next_obs.copy()
        info = dict(info)
        info["shield_shed_kw"] = float(info.get("total_unserved_kw", 0.0))
        return next_obs, reward, terminated, truncated, info
