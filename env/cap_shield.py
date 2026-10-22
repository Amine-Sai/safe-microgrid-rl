import numpy as np

CAP_KW = 368.15

# (obs index of zone load, action dim of its shed, industrial flag)
# Order: street -> stadium -> commercial -> residential -> industrial (PSM').
SHED_ORDER = (
    {"zone": "street", "obs_idx": 5, "act_dim": 4, "industrial": False},
    {"zone": "stadium", "obs_idx": 4, "act_dim": 3, "industrial": False},
    {"zone": "commercial", "obs_idx": 3, "act_dim": 2, "industrial": False},
    {"zone": "residential", "obs_idx": 2, "act_dim": 1, "industrial": False},
    {"zone": "industrial", "obs_idx": 6, "act_dim": 5, "industrial": True},
)


def predict_battery_kw(batt_cmd: float, soc: float, env) -> float:
    """Post-battery power in kW, replicating the env's clipping (read-only).

    Positive = charging (raises grid import), negative = discharging. Uses only the
    env's public constants; the env's own step applies the true physics.
    """
    p = float(np.clip(batt_cmd, -1.0, 1.0)) * env.bat_max_power_kw
    if p > 0:
        max_charge = ((env.bat_soc_max - soc) * env.bat_capacity_kwh
                      / (env.bat_eta_charge * env.dt_hours))
        p = max(0.0, min(p, max_charge))
    else:
        max_discharge = ((soc - env.bat_soc_min) * env.bat_capacity_kwh
                         * (env.bat_eta_discharge / env.dt_hours))
        p = min(0.0, max(p, -max_discharge))
    return p


def shed_actions_for(excess_kw: float, obs: np.ndarray) -> np.ndarray:
    """Minimum-excess shed actions (5 dims, linear mapping a = 2s-1), PSM' order."""
    shed_kw = np.zeros(5)
    rem = max(0.0, float(excess_kw))
    for entry in SHED_ORDER:
        if rem <= 0:
            break
        d = max(0.0, float(obs[entry["obs_idx"]]))
        if d <= 0:
            continue
        k = entry["obs_idx"] - 2  # zone position 0..4 == shed dim 0..4
        if entry["industrial"]:
            # partial cover within the 30% band, else full OFF
            if rem <= 0.30 * d:
                shed_kw[k] = rem
                rem = 0.0
            else:
                shed_kw[k] = d
                rem -= d
        else:
            take = min(rem, d)
            shed_kw[k] = take
            rem -= take
    # invert to linear-mapping actions
    act = np.full(5, -1.0, dtype=np.float32)  # s = 0
    for entry in SHED_ORDER:
        k = entry["obs_idx"] - 2
        d = max(0.0, float(obs[entry["obs_idx"]]))
        if shed_kw[k] <= 0 or d <= 0:
            continue
        if entry["industrial"]:
            if shed_kw[k] >= d - 1e-9:
                act[entry["act_dim"] - 1] = 1.0  # OFF
            else:
                s = shed_kw[k] * 0.9 / (0.30 * d)
                act[entry["act_dim"] - 1] = float(np.clip(2.0 * s - 1.0, -1.0, 0.80))
        else:
            act[entry["act_dim"] - 1] = float(2.0 * (shed_kw[k] / d) - 1.0)
    return np.clip(act, -1.0, 1.0).astype(np.float32)


class CapShield:
    """Wraps a SmartGridEnv. step(battery_1d) -> (obs, r, term, trunc, info).

    The obs returned is the inner env's obs unchanged. The info dict is the inner
    env's, plus `shield_shed_kw` (total kW shed by the shield this step).
    """

    def __init__(self, env, cap_kw: float = CAP_KW):
        self.env = env
        self.cap_kw = cap_kw

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        return obs, info

    def compose_action(self, batt_cmd_1d, obs: np.ndarray) -> np.ndarray:
        """1-D battery command + obs -> full 6-D env action with minimum-excess shed."""
        b = float(np.asarray(batt_cmd_1d, dtype=np.float32).ravel()[0])
        soc = float(obs[0])
        pv = float(obs[1])
        demand = float(sum(float(obs[i]) for i in (2, 3, 4, 5, 6)))
        batt_kw = predict_battery_kw(b, soc, self.env)
        predicted_import = max(0.0, demand - pv + batt_kw)
        excess = predicted_import - self.cap_kw
        shed = shed_actions_for(excess, obs)
        full = np.concatenate([[np.clip(b, -1.0, 1.0)], shed]).astype(np.float32)
        return full

    def step(self, batt_cmd_1d):
        """Drop-in env step with a 1-D battery action (uses the obs from reset/last step)."""
        if not hasattr(self, "_last_obs"):
            raise RuntimeError("CapShield.step called before reset")
        return self.step_with_obs(batt_cmd_1d, self._last_obs)

    def step_with_obs(self, batt_cmd_1d, obs: np.ndarray):
        full = self.compose_action(batt_cmd_1d, obs)
        obs2, r, term, trunc, info = self.env.step(full)
        self._last_obs = obs2.copy()
        info = dict(info)
        info["shield_shed_kw"] = float(np.sum([
            max(0.0, float(obs[i])) * max(0.0, (float(full[d]) + 1.0) / 2.0)
            for i, d in ((2, 1), (3, 2), (4, 3), (5, 4))
        ]))
        return obs2, r, term, trunc, info
