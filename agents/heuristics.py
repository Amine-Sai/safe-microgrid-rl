"""Microgrid heuristic policies and action mapping utilities.

Includes:
    - GridOnlyPolicy: Baseline policy that idles battery and serves 100% load across all zones.
    - PeakShaveMinPolicy (PSM'): Priority-ordered peak shaving and price-arbitrage heuristic.
    - pct_to_action: Conversion from zone service percentages to continuous action vectors.
"""

import os
from typing import Any, Dict, Optional
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class GridOnlyPolicy:
    r"""Grid-only baseline policy: zero battery dispatch and 100% served demand.

    Produces constant action vector:
    $$a = [0.0, -1.0, -1.0, -1.0, -1.0, -1.0]^\top$$
    representing idle battery ($a_0 = 0$) and zero load shedding across all 5 zones.
    """

    name: str = "grid_only"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialise grid-only policy.

        Args:
            params: Optional configuration dictionary (unused; for interface consistency).
        """
        self.params: Dict[str, Any] = params or {}

    def reset(self) -> None:
        """Reset policy state at episode boundary."""
        pass

    def get_action(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Return constant action: idle battery and 0% load shedding.

        Args:
            obs: Observation vector from environment.
            info: Optional environment auxiliary info dictionary.

        Returns:
            Continuous action array of shape (6,).
        """
        return np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def pct_to_action(
    pct_res: float,
    pct_com: float,
    pct_stad: float,
    pct_street: float,
    pct_ind: float,
    bat_cmd: float = 0.0,
    motor_min_pct: float = 70.0,
    industrial_shutdown_threshold: float = 0.90,
    shed_mapping: str = "linear",
) -> np.ndarray:
    r"""Map firmware zone service percentages (0-100%) to continuous action vector $\in [-1, 1]^6$.

    Action vector specification:
        - action[0]: Battery power command in [-1, 1] (-1 = discharge, +1 = charge).
        - action[1..5]: Zone shed actions for Residential, Commercial, Stadium, Street, Industrial.

    Mapping rules:
        - linear: shed_frac $\in [0, 1] \implies a = 2 \cdot \text{shed\_frac} - 1 \in [-1, 1]$.
        - relu/deadzone: shed_frac $\in [0, 1] \implies a = \text{shed\_frac}$ if shed_frac $> 0$ else $-1$.

    Industrial zone non-convexity:
        Motors trip offline if served power falls below `motor_min_pct` (default 70%).
        If requested shed exceeds `industrial_shutdown_threshold`, complete shutdown is commanded.

    Args:
        pct_res: Residential served percentage [0, 100].
        pct_com: Commercial served percentage [0, 100].
        pct_stad: Stadium served percentage [0, 100].
        pct_street: Street lighting served percentage [0, 100].
        pct_ind: Industrial served percentage [0, 100].
        bat_cmd: Battery command in [-1, 1].
        motor_min_pct: Minimum continuous operating percentage for industrial motors.
        industrial_shutdown_threshold: Shedding fraction threshold triggering full shutdown.
        shed_mapping: Mapping rule, either 'linear' or 'relu'/'deadzone'.

    Returns:
        Continuous action array with shape (6,) and dtype np.float32.
    """
    def shed_fraction_to_action(shed_frac: float) -> float:
        if shed_mapping in ("relu", "deadzone"):
            return float(np.clip(shed_frac, 0.0, 1.0)) if shed_frac > 0.0 else -1.0
        return float(np.clip(2.0 * shed_frac - 1.0, -1.0, 1.0))

    actions = []
    for p in (pct_res, pct_com, pct_stad, pct_street):
        p_c = max(0.0, min(100.0, float(p)))
        shed_frac = 1.0 - (p_c / 100.0)
        actions.append(shed_fraction_to_action(shed_frac))

    max_linear_shed = max(1e-9, 1.0 - (float(motor_min_pct) / 100.0))
    if pct_ind <= 0.0:
        a_ind = 1.0
    else:
        p_ind = max(float(motor_min_pct), min(100.0, float(pct_ind)))
        shed_ind = 1.0 - (p_ind / 100.0)
        s_ind = (shed_ind / max_linear_shed) * float(industrial_shutdown_threshold)
        max_linear_action = shed_fraction_to_action(float(industrial_shutdown_threshold))
        a_ind = min(shed_fraction_to_action(s_ind), float(max_linear_action))
    actions.append(float(a_ind))

    return np.array([float(np.clip(bat_cmd, -1.0, 1.0))] + actions, dtype=np.float32)


def compute_train_price_quantile(q: float) -> float:
    r"""Compute electricity price quantile $q \in [0, 1]$ on historical training split.

    Args:
        q: Quantile level, e.g. 0.60 for the 60th percentile.

    Returns:
        Price threshold in EUR/MWh.
    """
    import pandas as pd

    path = os.path.join(REPO_ROOT, "data", "processed", "train.parquet")
    df = pd.read_parquet(path)
    return float(df["price_eur_mwh"].quantile(q))


_train_price_quantile = compute_train_price_quantile


def resolve_default_cap_kw() -> float:
    """Resolve grid import capacity limit from environment default configuration.

    Returns:
        Grid import capacity threshold in kW.
    """
    from env.smartgrid_env import default_config, resolve_grid_import_cap

    return float(resolve_grid_import_cap(default_config()))


_env_cap_kw = resolve_default_cap_kw


class PeakShaveMinPolicy:
    r"""Peak shaving + low-price charging + minimal priority shedding heuristic (PSM').

    Heuristic dispatch algorithm:
    1. Battery Discharge (Peak Shaving):
       If estimated net grid import exceeds $\theta_{\text{hi}} \cdot P_{\text{cap}}$,
       discharge battery up to $P_{\text{max}}$ and available SoC above $\text{SoC}_{\text{min}}$.

    2. Battery Charge (Price Arbitrage / Renewable Absorption):
       If electricity price $\le \text{price\_threshold}$ (e.g. 60th percentile) or
       PV generation exceeds total demand, charge battery up to available headroom
       below $P_{\text{cap}}$ and below $\text{SoC}_{\text{max}}$.

    3. Priority Load Shedding:
       If estimated net grid import after battery action still exceeds
       $(P_{\text{cap}} - \text{margin})$, shed loads in reverse critical priority order:
       $$\text{Street Lights} \to \text{Stadium} \to \text{Commercial} \to \text{Residential} \to \text{Industrial}$$
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PSM' heuristic policy with microgrid parameters.

        Args:
            params: Optional parameter dictionary overriding environment defaults.
        """
        from env.smartgrid_env import default_config
        env_defaults = default_config()

        self.params: Dict[str, Any] = {
            "theta_hi": 1.00,
            "q": 0.60,
            "price_threshold": None,
            "charge_power_frac": 0.50,
            "cap_kw": None,
            "max_power_kw": env_defaults["battery_max_power_kw"],
            "battery_capacity_kwh": env_defaults["battery_capacity_kwh"],
            "soc_floor": env_defaults["battery_soc_min"],
            "soc_max": env_defaults["battery_soc_max"],
            "eta_charge": env_defaults["battery_eta_charge"],
            "eta_discharge": env_defaults["battery_eta_discharge"],
            "dt_hours": env_defaults["dt_hours"],
            "motor_min_pct": env_defaults["motor_min_pct"],
            "industrial_shutdown_threshold": env_defaults.get("industrial_shutdown_threshold", 0.90),
            "shed_mapping": env_defaults.get("shed_mapping", "linear"),
            "safety_margin_kw": 0.05,
        }
        if params:
            self.params.update(params)

        p = self.params
        if p["price_threshold"] is None:
            p["price_threshold"] = compute_train_price_quantile(float(p["q"]))
        if p["cap_kw"] is None:
            p["cap_kw"] = resolve_default_cap_kw()

        self.theta_hi: float = float(p["theta_hi"])
        self.price_thresh: float = float(p["price_threshold"])
        self.charge_frac: float = float(p["charge_power_frac"])
        self.cap: float = float(p["cap_kw"])
        self.p_max: float = float(p["max_power_kw"])
        self.c_bat: float = float(p["battery_capacity_kwh"])
        self.soc_floor: float = float(p["soc_floor"])
        self.soc_max: float = float(p["soc_max"])
        self.eta_c: float = float(p["eta_charge"])
        self.eta_d: float = float(p["eta_discharge"])
        self.dt: float = float(p["dt_hours"])
        self.motor_min_pct: float = float(p["motor_min_pct"])
        self.industrial_shutdown_threshold: float = float(p["industrial_shutdown_threshold"])
        self.shed_mapping: str = str(p["shed_mapping"])
        self.safety_margin_kw: float = float(p["safety_margin_kw"])

    def reset(self) -> None:
        """Reset policy internal state at episode boundary."""
        pass

    def get_action(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        r"""Compute heuristic dispatch action from environment observation.

        Args:
            obs: Observation array containing [SoC, PV, Loads (5), Price, Hour, Day, Month].
            info: Optional auxiliary info dictionary.

        Returns:
            Continuous action array $\in [-1, 1]^6$.
        """
        soc = float(obs[0])
        pv_kw = float(obs[1])
        (
            residential_load_kw,
            commercial_load_kw,
            stadium_load_kw,
            street_load_kw,
            industrial_load_kw,
        ) = (float(obs[i]) for i in range(2, 7))
        total_load = (
            residential_load_kw
            + commercial_load_kw
            + stadium_load_kw
            + street_load_kw
            + industrial_load_kw
        )
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
        target_cap = self.cap - self.safety_margin_kw
        max_linear_ind_frac = 1.0 - (self.motor_min_pct / 100.0)

        if import_est > target_cap:
            rem = import_est - target_cap
            if rem > 0 and street_load_kw > 0:
                shed = min(rem, street_load_kw)
                pct_street = (1.0 - shed / street_load_kw) * 100.0
                rem -= shed
            if rem > 0 and stadium_load_kw > 0:
                shed = min(rem, stadium_load_kw)
                pct_stad = (1.0 - shed / stadium_load_kw) * 100.0
                rem -= shed
            if rem > 0 and commercial_load_kw > 0:
                shed = min(rem, commercial_load_kw)
                pct_com = (1.0 - shed / commercial_load_kw) * 100.0
                rem -= shed
            if rem > 0 and residential_load_kw > 0:
                shed = min(rem, residential_load_kw)
                pct_res = (1.0 - shed / residential_load_kw) * 100.0
                rem -= shed
            if rem > 0 and industrial_load_kw > 0:
                if rem <= max_linear_ind_frac * industrial_load_kw:
                    pct_ind = (1.0 - rem / industrial_load_kw) * 100.0
                    rem = 0.0
                else:
                    pct_ind = 0.0
                    rem -= industrial_load_kw

        return pct_to_action(
            pct_res,
            pct_com,
            pct_stad,
            pct_street,
            pct_ind,
            bat_cmd=bat_cmd,
            motor_min_pct=self.motor_min_pct,
            industrial_shutdown_threshold=self.industrial_shutdown_threshold,
            shed_mapping=self.shed_mapping,
        )


def psm_prime_policy() -> PeakShaveMinPolicy:
    """Construct a default instance of the PSM' heuristic baseline policy.

    Returns:
        Configured PeakShaveMinPolicy instance.
    """
    return PeakShaveMinPolicy()
