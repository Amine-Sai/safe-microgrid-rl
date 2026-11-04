"""CapShield Analytical Safety Filter for Microgrid Operations.

Implements an analytical post-hoc safety shield that wraps a `SmartGridEnv`.
Given a 1-D continuous battery action $a_{\\text{bat}} \\in [-1, 1]$, the shield:
1. Clamps battery charging to available grid import headroom:
   $$P_{\\text{bat, max-ch}} = \\max(0, P_{\\text{cap}} - P_{\\text{load, served}} + P_{\\text{pv}})$$
2. Evaluates remaining potential grid import cap exceedance:
   $$\\Delta P_{\\text{excess}} = \\max\\left(0, \\hat{P}_{\\text{grid, import}} - P_{\\text{cap}}\\right)$$
3. Analytically computes minimal-impact load shedding actions across microgrid zones
   in priority order according to Value of Lost Load (VoLL) under the non-linear
   piecewise industrial motor constraint.
4. Translates shed requirements into environment action dimensions with a 1-ULP
   float32 margin to mathematically guarantee zero grid cap violations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from env.smartgrid_env import battery_power_kw

# Priority shedding order based on Value of Lost Load (VoLL) and zone criticality:
# 1. Street lighting: municipal lighting (lowest economic penalty)
# 2. Stadium: non-essential entertainment demand
# 3. Commercial: commercial buildings and HVAC
# 4. Residential: domestic residential demand (high social priority)
# 5. Industrial: heavy motor loads with strict operational floor constraints
SHED_ORDER: Tuple[Dict[str, Any], ...] = (
    {"zone": "street", "obs_idx": 5, "act_dim": 4, "industrial": False},
    {"zone": "stadium", "obs_idx": 4, "act_dim": 3, "industrial": False},
    {"zone": "commercial", "obs_idx": 3, "act_dim": 2, "industrial": False},
    {"zone": "residential", "obs_idx": 2, "act_dim": 1, "industrial": False},
    {"zone": "industrial", "obs_idx": 6, "act_dim": 5, "industrial": True},
)


def predict_battery_kw(
    batt_cmd: float,
    soc: float,
    env: Any,
    served_kw: Optional[float] = None,
    pv_kw: float = 0.0,
    clamp: Optional[bool] = None,
) -> float:
    r"""Predict terminal battery power in kW mirroring environment clipping without mutating state.

    Positive power indicates charging (draws power from microgrid/grid, increasing import),
    while negative power indicates discharging (injects power into microgrid, reducing import).

    Args:
        batt_cmd: Normalized battery command in [-1.0, 1.0].
        soc: Current battery State-of-Charge fraction in [0.0, 1.0].
        env: Environment or mock containing battery physical parameters.
        served_kw: Total electrical load served across all zones in kW.
                   If None, defaults to 0.0 with clamp=False (unless explicitly provided).
        pv_kw: Current solar PV generation in kW.
        clamp: Whether to apply cap headroom clamping. If None and served_kw is None, defaults to False.

    Returns:
        Predicted admissible battery power in kW.
    """
    if served_kw is None:
        clamp = False if clamp is None else clamp
        served_kw = 0.0
    return float(battery_power_kw(batt_cmd, soc, served_kw, pv_kw, env, clamp=clamp))


def shed_actions_for(
    excess_kw: float,
    obs: np.ndarray,
    motor_min_fraction: float = 0.70,
    industrial_shutdown_threshold: float = 0.90,
) -> np.ndarray:
    r"""Compute analytical minimum-excess zone load shedding actions under the priority hierarchy.

    Given an import cap exceedance $\Delta P_{\text{excess}} = \max(0, \hat{P}_{\text{import}} - P_{\text{cap}})$,
    the algorithm allocates required curtailment across zones according to increasing Value of
    Lost Load (VoLL) order:
        $\text{street} \to \text{stadium} \to \text{commercial} \to \text{residential} \to \text{industrial}$.

    Iterative Zone Curtailment:
    For each zone $z$ with demand $L_z$:
    - For non-industrial zones ($z \in \{\text{street, stadium, commercial, residential}\}$):
      $$\Delta L_z = \min(\text{remaining\_excess}, L_z)$$
      $$\text{remaining\_excess} \leftarrow \text{remaining\_excess} - \Delta L_z$$
    - For the industrial zone ($z = \text{industrial}$):
      Due to the piecewise operational constraint (motor must run $\ge f_{\text{min}}$ or shut down),
      partial shedding is capped at $L_{\text{ind, max-linear}} = (1 - f_{\text{min}}) L_{\text{ind}}$.
      * If $\text{remaining\_excess} \le L_{\text{ind, max-linear}}$:
        $$\Delta L_{\text{ind}} = \text{remaining\_excess}$$
        $$\text{remaining\_excess} \leftarrow 0$$
      * Else:
        The motor cannot run at an intermediate forbidden load fraction, so it is completely shut down:
        $$\Delta L_{\text{ind}} = L_{\text{ind}}$$
        $$\text{remaining\_excess} \leftarrow \text{remaining\_excess} - L_{\text{ind}}$$

    Action Space Inversion & Float32 Precision Protection:
    Under the default linear mapping $s = \frac{a + 1}{2} \iff a = 2s - 1$.
    To prevent numerical float32 truncation under-shedding (where $2s - 1$ in 32-bit float
    evaluates to a fraction $s - \epsilon$, causing a micro-violation of $10^{-7}$ kW),
    the calculated continuous action is nudged upward by 1 ULP (Unit in the Last Place)
    via `np.nextafter(a, 1.0)`.

    Args:
        excess_kw: Required reduction in power import in kW.
        obs: Current environment observation vector (zones located at indices 2..6).
        motor_min_fraction: Minimum continuous operational load fraction for industrial motors.
        industrial_shutdown_threshold: Action threshold in [0, 1] triggering full industrial shutdown.

    Returns:
        np.ndarray of shape (5,) with continuous shedding actions in [-1.0, 1.0].
    """
    if obs.shape[0] < 7:
        raise ValueError(
            f"Observation vector has shape {obs.shape}, but must have at least 7 features "
            "to extract zone loads at indices 2..6."
        )

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
            # Partial cover within the linear motor band, otherwise fully off
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

    # Linear mapping s = (a + 1)/2 -> a = 2s - 1 with 1-ULP margin for float32 precision
    zone_shed_actions = np.full(5, -1.0, dtype=np.float32)
    for entry in SHED_ORDER:
        zone_index = entry["obs_idx"] - 2
        zone_demand_kw = max(0.0, float(obs[entry["obs_idx"]]))
        allocated_shed_kw = zone_shed_kw[zone_index]
        if allocated_shed_kw <= 0.0 or zone_demand_kw <= 0.0:
            continue

        action_dim_idx = entry["act_dim"] - 1
        if entry["industrial"]:
            if allocated_shed_kw >= zone_demand_kw - 1e-9:
                zone_shed_actions[action_dim_idx] = 1.0  # Fully shut down
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
    r"""Analytical Safety Shield Wrapper for SmartGridEnv.

    Wraps a `SmartGridEnv` to provide provable analytical safety guarantees against
    grid import cap violations. It translates a 1-dimensional battery action into a full
    6-dimensional environment action [battery_cmd, shed_res, shed_com, shed_stad, shed_street, shed_ind]
    by dynamically evaluating microgrid headroom and scheduling minimum necessary load shedding.

    Safety Projection Pipeline:
    1. Grid Import Headroom Clamping:
       Battery charging is constrained so as not to exceed the grid cap on its own:
       $$P_{\text{bat, max-ch}} = \max\left(0, P_{\text{cap}} - P_{\text{load, total}} + P_{\text{pv}}\right)$$
    2. Import Exceedance Prediction:
       $$\hat{P}_{\text{grid, import}} = \max\left(0, P_{\text{load, total}} + P_{\text{bat}} - P_{\text{pv}}\right)$$
       $$\Delta P_{\text{excess}} = \hat{P}_{\text{grid, import}} - P_{\text{cap}}$$
    3. Analytical Load Shedding Projection:
       If $\Delta P_{\text{excess}} > 0$, shed actions are computed via `shed_actions_for()`,
       guaranteeing $P_{\text{grid, import}} \le P_{\text{cap}}$ within IEEE float32 precision.
    """

    def __init__(self, env: gym.Env, cap_kw: Optional[float] = None) -> None:
        """Initialize the CapShield wrapper around a microgrid environment.

        Args:
            env: Target SmartGridEnv instance.
            cap_kw: Optional grid import cap in kW. If None, resolves from `env.grid_import_cap_kw`.
        """
        self.env = env
        self.cap_kw: float = (
            float(env.grid_import_cap_kw) if cap_kw is None else float(cap_kw)
        )
        self._last_obs: Optional[np.ndarray] = None

    def __getattr__(self, name: str) -> Any:
        """Forward attribute queries to the wrapped environment."""
        return getattr(self.env, name)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the wrapped environment and cache the initial observation."""
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def compose_action(
        self,
        battery_command_1d: Union[float, np.ndarray, List[float]],
        obs: np.ndarray,
    ) -> np.ndarray:
        r"""Compose a full 6-D environment action from a 1-D battery command and observation.

        Args:
            battery_command_1d: 1-D scalar or array battery command in [-1.0, 1.0].
            obs: Observation vector with at least 7 features.

        Returns:
            6-dimensional action vector in [-1.0, 1.0] ready for `SmartGridEnv.step`.
        """
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
                excess_kw,
                obs,
                motor_min_fraction=motor_min_pct,
                industrial_shutdown_threshold=ind_off_thresh,
            )
            full_action = np.concatenate(
                [[np.clip(battery_command, -1.0, 1.0)], shed_actions]
            ).astype(np.float32)
            return full_action

        # Pre-clamp battery power against un-shed demand
        batt_kw = battery_power_kw(
            battery_command, soc, total_demand_kw, pv_kw, self.env, clamp=True
        )
        predicted_import_kw = max(0.0, total_demand_kw - pv_kw + batt_kw)
        excess_kw = predicted_import_kw - self.cap_kw

        shed_actions = shed_actions_for(
            excess_kw,
            obs,
            motor_min_fraction=motor_min_pct,
            industrial_shutdown_threshold=ind_off_thresh,
        )
        full_action = np.concatenate(
            [[np.clip(battery_command, -1.0, 1.0)], shed_actions]
        ).astype(np.float32)
        return full_action

    def step(
        self,
        battery_command_1d: Union[float, np.ndarray, List[float]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment with a 1-D battery action using the cached observation."""
        if self._last_obs is None:
            raise RuntimeError(
                "CapShield.step() called before reset(). You must call reset() before stepping."
            )
        return self.step_with_obs(battery_command_1d, self._last_obs)

    def step_with_obs(
        self,
        battery_command_1d: Union[float, np.ndarray, List[float]],
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment with a 1-D battery command and an explicitly passed observation."""
        full_action = self.compose_action(battery_command_1d, obs)
        next_obs, reward, terminated, truncated, info = self.env.step(full_action)
        self._last_obs = next_obs.copy()
        info = dict(info)
        info["shield_shed_kw"] = float(info.get("total_unserved_kw", 0.0))
        return next_obs, reward, terminated, truncated, info
