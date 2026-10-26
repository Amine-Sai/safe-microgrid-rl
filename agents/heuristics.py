import os
from typing import Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class GridOnlyPolicy:
    """No battery, 100% served load across all zones."""

    name = "grid_only"

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}

    def reset(self):
        pass

    def get_action(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        # bat_cmd = 0.0, all zone shed actions = -1.0 (0% shed, 100% served)
        return np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def pct_to_action(pct_res, pct_com, pct_stad, pct_street, pct_ind, bat_cmd=0.0):
    """Map firmware zone percentages (0-100) to a continuous action in [-1, 1].

    Zones 0..3 (res, com, stad, street): shed = 1 - pct/100, s = (a+1)/2
    => a = 1 - 2*(pct/100).
    Industrial: pct <= 0 => motor OFF (a = 1.0); else pct in [70, 100],
    shed in [0, 0.3], s = 3*shed, a = 2*s - 1.
    """
    actions = []
    for p in (pct_res, pct_com, pct_stad, pct_street):
        p_c = max(0.0, min(100.0, float(p)))
        actions.append(float(np.clip(1.0 - 2.0 * (p_c / 100.0), -1.0, 1.0)))

    if pct_ind <= 0:
        a_ind = 1.0
    else:
        p_ind = max(70.0, min(100.0, float(pct_ind)))
        shed_ind = 1.0 - p_ind / 100.0
        a_ind = 2.0 * (shed_ind * 3.0) - 1.0
        a_ind = float(np.clip(a_ind, -1.0, 0.80))
    actions.append(a_ind)

    return np.array([float(np.clip(bat_cmd, -1.0, 1.0))] + actions, dtype=np.float32)


def _train_price_quantile(q: float) -> float:
    """The price quantile that matches `q`, read from the Train split."""
    import pandas as pd

    path = os.path.join(REPO_ROOT, "data", "processed", "train.parquet")
    df = pd.read_parquet(path)
    return float(df["price_eur_mwh"].quantile(q))


class PeakShaveMinPolicy:
    """PSM': battery peak shaving + cheap-price charging + minimal load shedding.

    The last resort sheds only the excess import, in fixed priority order:
    street -> stadium -> commercial -> residential -> industrial (max 30 % shed).
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = {
            "theta_hi": 1.00,
            "q": 0.60,
            "price_threshold": None,   # filled in below when not given
            "charge_power_frac": 0.50,
            "cap_kw": None,            # filled in below (the env's import cap)
            "max_power_kw": 250.0,
            "battery_capacity_kwh": 500.0,
            "soc_floor": 0.10,
            "soc_max": 0.90,
            "eta_charge": 0.95,
            "eta_discharge": 0.95,
            "dt_hours": 1.0,
        }
        if params:
            self.params.update(params)

        p = self.params
        if p["price_threshold"] is None:
            p["price_threshold"] = _train_price_quantile(float(p["q"]))
        if p["cap_kw"] is None:
            p["cap_kw"] = _env_cap_kw()

        self.theta_hi = float(p["theta_hi"])
        self.price_thresh = float(p["price_threshold"])
        self.charge_frac = float(p["charge_power_frac"])
        self.cap = float(p["cap_kw"])
        self.p_max = float(p["max_power_kw"])
        self.c_bat = float(p["battery_capacity_kwh"])
        self.soc_floor = float(p["soc_floor"])
        self.soc_max = float(p["soc_max"])
        self.eta_c = float(p["eta_charge"])
        self.eta_d = float(p["eta_discharge"])
        self.dt = float(p["dt_hours"])

    def reset(self):
        pass

    def get_action(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        soc = float(obs[0])
        pv_kw = float(obs[1])
        l_res, l_com, l_stad, l_street, l_ind = (float(obs[i]) for i in range(2, 7))
        total_load = l_res + l_com + l_stad + l_street + l_ind
        price_mwh = float(obs[7])

        net_import_raw = max(0.0, total_load - pv_kw)
        thresh_kw = self.theta_hi * self.cap
        bat_cmd = 0.0

        if net_import_raw > thresh_kw:
            deficit = net_import_raw - thresh_kw
            max_dis_soc = max(0.0, (soc - self.soc_floor) * self.c_bat * self.eta_d / self.dt)
            p_dis = min(self.p_max, deficit, max_dis_soc)
            if p_dis > 0:
                bat_cmd = -p_dis / self.p_max
        elif price_mwh <= self.price_thresh:
            headroom = max(0.0, self.cap - net_import_raw)
            max_ch_soc = max(0.0, (self.soc_max - soc) * self.c_bat / (self.eta_c * self.dt))
            p_ch = min(self.charge_frac * self.p_max, headroom, max_ch_soc)
            if p_ch > 0:
                bat_cmd = p_ch / self.p_max
        elif pv_kw > total_load:
            surplus = pv_kw - total_load
            max_ch_soc = max(0.0, (self.soc_max - soc) * self.c_bat / (self.eta_c * self.dt))
            p_ch = min(self.p_max, surplus, max_ch_soc)
            if p_ch > 0:
                bat_cmd = p_ch / self.p_max

        if bat_cmd > 0:
            p_ch_est, p_dis_est = bat_cmd * self.p_max, 0.0
        else:
            p_ch_est, p_dis_est = 0.0, -bat_cmd * self.p_max
        import_est = max(0.0, total_load - pv_kw + p_ch_est - p_dis_est)

        pct_res = pct_com = pct_stad = pct_street = pct_ind = 100.0
        target_cap = self.cap - 0.05
        if import_est > target_cap:
            rem = import_est - target_cap
            if rem > 0 and l_street > 0:
                shed = min(rem, l_street)
                pct_street = (1.0 - shed / l_street) * 100.0
                rem -= shed
            if rem > 0 and l_stad > 0:
                shed = min(rem, l_stad)
                pct_stad = (1.0 - shed / l_stad) * 100.0
                rem -= shed
            if rem > 0 and l_com > 0:
                shed = min(rem, l_com)
                pct_com = (1.0 - shed / l_com) * 100.0
                rem -= shed
            if rem > 0 and l_res > 0:
                shed = min(rem, l_res)
                pct_res = (1.0 - shed / l_res) * 100.0
                rem -= shed
            if rem > 0 and l_ind > 0:
                if rem <= 0.30 * l_ind:
                    pct_ind = (1.0 - rem / l_ind) * 100.0
                    rem = 0.0
                else:
                    pct_ind = 0.0
                    rem -= l_ind

        return pct_to_action(pct_res, pct_com, pct_stad, pct_street, pct_ind, bat_cmd=bat_cmd)


def _env_cap_kw() -> float:
    """The env's grid import cap, resolved the same way the env resolves it."""
    from env.smartgrid_env import default_config, resolve_grid_import_cap

    return resolve_grid_import_cap(default_config())


def psm_prime_policy() -> PeakShaveMinPolicy:
    """The residual base policy used by the `ppo_residual` agent."""
    return PeakShaveMinPolicy()
