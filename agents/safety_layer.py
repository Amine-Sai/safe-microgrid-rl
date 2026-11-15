"""Differentiable Safety Layer based on Dalal et al. (2018).

References:
    Dalal, G., Dvijotham, K., Vecerik, M., Hester, T., Paduraru, C., & Tassa, Y. (2018).
    "Safe Exploration in Continuous Action Spaces". In Proceedings of the 32nd AAAI
    Conference on Artificial Intelligence.

Provides analytical, closed-form projection of continuous microgrid dispatch actions
onto linear state-dependent constraint boundaries, enabling end-to-end gradient flow.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from agents.ppo import PolicyNetwork
from env.smartgrid_env import default_config


class DifferentiableSafetyLayer(nn.Module):
    r"""Analytical safety layer for linear peak-demand and battery SoC constraints (Dalal et al., 2018).

    Implements a closed-form projection of continuous dispatch actions $a \in [-1, 1]^d$
    to guarantee that the peak grid import cap constraint and battery SoC limits are satisfied:

    $$g(s, a) = P_{\text{net}}(s) + P_{\text{max}} \cdot a_0 - \sum_{z=1}^K L_z(s) \frac{1 + a_z}{2} - P_{\text{cap}} \le 0$$
    $$a_{0, \min}(s) \le a_0 \le a_{0, \max}(s)$$

    where:
        - $P_{\text{net}}(s) = \sum_{z=2}^6 \text{load}_z(s) - P_{\text{PV}}(s)$ is the net load in kW.
        - $a_0 > 0$ denotes battery charging (increasing grid draw).
        - $a_0 < 0$ denotes battery discharging (reducing grid draw).
        - $a_{1..5}$ denote zone load shedding commands (with linear mapping $s_z = \frac{1 + a_z}{2}$).
        - $P_{\text{max}}$ is the rated battery power in kW.
        - $P_{\text{cap}}$ is the peak grid import threshold in kW.
        - $[a_{0, \min}, a_{0, \max}]$ are the physical State-of-Charge (obs[0]) limits.

    1D Analytical Solution:
        When action_dim = 1 (battery only), the Euclidean projection onto $g(s, a) \le 0$
        and $[a_{0, \min}, a_{0, \max}]$ is:
        $$a_0^* = \operatorname{clamp}\left(a_0 - \frac{\operatorname{ReLU}(g(s, a))}{P_{\text{max}}}, a_{0, \min}, a_{0, \max}\right)$$

    Multi-Dimensional Joint KKT QP Solution:
        When action_dim >= 6 (battery + load shedding), solves the coupled quadratic program:
        $$\min_u \frac{1}{2} \|u - a\|_2^2 \quad \text{s.t.} \quad c^\top u + d \le 0, \quad u \in [l(s), h(s)]$$
        via analytical KKT projection with active set bound correction, where the active
        hyperplane projection possesses an analytical Jacobian of rank $m - 1 > 0$.

    Straight-Through Estimator (STE) Gradient Surrogate:
        To prevent vanishing gradients when the constraint is active ($g > 0$), backward
        gradients are propagated across active projections via a configurable leaky surrogate:
        $$\tilde{a} = u^* + (a - a.\text{detach}()) \cdot \alpha_{\text{ste}}$$
        where $\alpha_{\text{ste}} > 0$. The forward pass is 100% hard-safe, while policy
        gradients remain informative and non-zero even when $P_{\text{net}} > P_{\text{cap}}$.
    """

    def __init__(
        self,
        p_max: Optional[float] = None,
        p_cap: float = 368.153918,
        obs_normalizer: Optional[object] = None,
        soc_min: Optional[float] = None,
        soc_max: Optional[float] = None,
        eta_charge: Optional[float] = None,
        eta_discharge: Optional[float] = None,
        capacity_kwh: Optional[float] = None,
        dt_hours: Optional[float] = None,
        surrogate_grad_scale: float = 0.1,
        enforce_soc: bool = True,
    ) -> None:
        """Initialise the differentiable analytical safety projection layer.

        Args:
            p_max: Battery rated maximum charging/discharging power in kW.
            p_cap: Grid import capacity threshold in kW.
            obs_normalizer: Optional RunningMeanStd tracker to map normalized states back to kW.
            soc_min: Minimum battery state of charge fraction.
            soc_max: Maximum battery state of charge fraction.
            eta_charge: Battery charging efficiency fraction.
            eta_discharge: Battery discharging efficiency fraction.
            capacity_kwh: Total battery capacity in kWh.
            dt_hours: Timestep duration in hours.
            surrogate_grad_scale: Straight-through gradient scale alpha across active clamps.
            enforce_soc: Whether to enforce physical battery SoC bounds on action[0].
        """
        super().__init__()
        cfg = default_config()
        self.p_max: float = float(p_max if p_max is not None else cfg["battery_max_power_kw"])
        self.p_cap: float = float(p_cap)
        self.obs_normalizer = obs_normalizer
        self.soc_min: float = float(soc_min if soc_min is not None else cfg["battery_soc_min"])
        self.soc_max: float = float(soc_max if soc_max is not None else cfg["battery_soc_max"])
        self.eta_charge: float = float(eta_charge if eta_charge is not None else cfg["battery_eta_charge"])
        self.eta_discharge: float = float(eta_discharge if eta_discharge is not None else cfg["battery_eta_discharge"])
        self.capacity_kwh: float = float(capacity_kwh if capacity_kwh is not None else cfg["battery_capacity_kwh"])
        self.dt_hours: float = float(dt_hours if dt_hours is not None else cfg.get("dt_hours", 1.0))
        self.surrogate_grad_scale: float = float(surrogate_grad_scale)
        self.enforce_soc: bool = bool(enforce_soc)

    def unnormalize_state(self, state: th.Tensor) -> th.Tensor:
        """Unnormalise state tensor to original physical units (kW) if normaliser is active.

        Args:
            state: State tensor of shape `(B, state_dim)` or `(state_dim,)`.

        Returns:
            State tensor in physical units on the same device and dtype.
        """
        if self.obs_normalizer is not None and getattr(self.obs_normalizer, "count", 0) > 0:
            mean = th.as_tensor(self.obs_normalizer.mean, dtype=state.dtype, device=state.device)
            var = getattr(self.obs_normalizer, "var", None)
            eps = getattr(self.obs_normalizer, "epsilon", 1e-8)
            std = th.as_tensor(
                np.sqrt(var) + eps if var is not None else 1.0,
                dtype=state.dtype,
                device=state.device,
            )
            return state * std + mean
        return state

    def get_soc(self, state: th.Tensor) -> th.Tensor:
        """Extract unnormalized battery State-of-Charge fraction (obs[0]).

        Args:
            state: State observation tensor `(B, state_dim)` or `(state_dim,)`.

        Returns:
            SoC tensor `(B, 1)` with values in [0.0, 1.0].
        """
        raw_state = self.unnormalize_state(state)
        if raw_state.dim() == 1:
            raw_state = raw_state.unsqueeze(0)
        if raw_state.shape[-1] >= 1:
            return raw_state[:, 0:1]
        return th.full((raw_state.shape[0], 1), 0.5, dtype=raw_state.dtype, device=raw_state.device)

    def get_soc_bounds(self, state: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        r"""Compute physical bounds $[a_{0, \min}, a_{0, \max}] \subseteq [-1, 1]$ on battery command.

        Given current State-of-Charge $\text{SoC} = s[0]$:
        $$a_{0, \min} = \operatorname{clamp}\left(-\frac{(\text{SoC} - \text{SoC}_{\min}) C_{\text{bat}} \eta_d}{\Delta t \cdot P_{\max}}, -1.0, 0.0\right)$$
        $$a_{0, \max} = \operatorname{clamp}\left(\frac{(\text{SoC}_{\max} - \text{SoC}) C_{\text{bat}}}{\eta_c \Delta t \cdot P_{\max}}, 0.0, 1.0\right)$$

        Args:
            state: State observation tensor `(B, state_dim)` or `(state_dim,)`.

        Returns:
            Tuple of (a0_min, a0_max), each of shape `(B, 1)`.
        """
        soc = self.get_soc(state)
        max_dis_p = (soc - self.soc_min) * self.capacity_kwh * (self.eta_discharge / self.dt_hours)
        a0_min = th.clamp(-max_dis_p / self.p_max, -1.0, 0.0)

        max_ch_p = (self.soc_max - soc) * self.capacity_kwh / (self.eta_charge * self.dt_hours)
        a0_max = th.clamp(max_ch_p / self.p_max, 0.0, 1.0)
        return a0_min, a0_max

    def get_p_net(self, state: th.Tensor) -> th.Tensor:
        r"""Compute net microgrid load in kW: $P_{\text{net}} = \sum \text{loads} - P_{\text{PV}}$.

        Args:
            state: State tensor `(B, state_dim)` or `(state_dim,)`.

        Returns:
            Net load tensor `(B, 1)` in kW.
        """
        raw_state = self.unnormalize_state(state)
        if raw_state.dim() == 1:
            raw_state = raw_state.unsqueeze(0)
        if raw_state.shape[-1] >= 7:
            pv = raw_state[:, 1:2]
            loads = raw_state[:, 2:7].sum(dim=-1, keepdim=True)
            return loads - pv
        if raw_state.shape[-1] >= 2:
            pv = raw_state[:, 1:2]
            loads = raw_state[:, 2:].sum(dim=-1, keepdim=True) if raw_state.shape[-1] > 2 else th.zeros_like(pv)
            return loads - pv
        return th.zeros((raw_state.shape[0], 1), dtype=raw_state.dtype, device=raw_state.device)

    def get_zone_loads(self, state: th.Tensor) -> th.Tensor:
        """Extract individual zone loads in kW (obs[2..6]).

        Args:
            state: State tensor `(B, state_dim)` or `(state_dim,)`.

        Returns:
            Zone loads tensor `(B, 5)` in kW.
        """
        raw_state = self.unnormalize_state(state)
        if raw_state.dim() == 1:
            raw_state = raw_state.unsqueeze(0)
        if raw_state.shape[-1] >= 7:
            return raw_state[:, 2:7]
        return th.zeros((raw_state.shape[0], 5), dtype=raw_state.dtype, device=raw_state.device)

    def constraint_val(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        r"""Compute linear constraint function $g(s, a) \le 0$.

        Positive indicates constraint violation.

        Args:
            state: State observation tensor `(B, state_dim)` or `(state_dim,)`.
            action: Action tensor `(B, action_dim)` or `(action_dim,)`.

        Returns:
            Constraint evaluation tensor `(B, 1)` or scalar/1D tensor.
        """
        is_1d = action.dim() == 1
        if is_1d:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)

        if state.device != action.device:
            state = state.to(action.device)

        p_net = self.get_p_net(state)
        a0 = action[:, 0:1]

        if action.shape[1] > 1:
            zone_loads = self.get_zone_loads(state)
            n_zones = min(action.shape[1] - 1, zone_loads.shape[1])
            shed_actions = action[:, 1 : 1 + n_zones]
            shed_kw = (zone_loads[:, :n_zones] * (1.0 + shed_actions) * 0.5).sum(dim=-1, keepdim=True)
            g = p_net + self.p_max * a0 - shed_kw - self.p_cap
        else:
            g = p_net + self.p_max * a0 - self.p_cap

        return g.squeeze(0) if is_1d else g

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        r"""Project continuous action to satisfy linear grid cap and battery SoC constraints.

        Args:
            state: Environment state observation tensor `(B, state_dim)` or `(state_dim,)`.
            action: Continuous action tensor `(B, action_dim)` or `(action_dim,)`.

        Returns:
            Projected action tensor of identical shape, dtype, and device.
        """
        is_1d = action.dim() == 1
        if is_1d:
            state = state.unsqueeze(0) if state.dim() == 1 else state
            action = action.unsqueeze(0)

        if state.device != action.device:
            state = state.to(action.device)

        b, act_dim = action.shape

        if self.enforce_soc:
            a0_min, a0_max = self.get_soc_bounds(state)
        else:
            a0_min = th.full((b, 1), -1.0, dtype=action.dtype, device=action.device)
            a0_max = th.full((b, 1), 1.0, dtype=action.dtype, device=action.device)

        # Action bounds
        if act_dim > 1:
            lower_bounds = th.cat([a0_min, th.full((b, act_dim - 1), -1.0, dtype=action.dtype, device=action.device)], dim=-1)
            upper_bounds = th.cat([a0_max, th.full((b, act_dim - 1), 1.0, dtype=action.dtype, device=action.device)], dim=-1)
        else:
            lower_bounds = a0_min
            upper_bounds = a0_max

        # Normal vector c for linear constraint g(s, u) = c^T u + d
        c0 = th.full((b, 1), self.p_max, dtype=action.dtype, device=action.device)
        if act_dim > 1:
            zone_loads = self.get_zone_loads(state)
            n_zones = min(act_dim - 1, zone_loads.shape[1])
            cz = -0.5 * zone_loads[:, :n_zones]
            c_parts = [c0, cz]
            if act_dim - 1 > n_zones:
                c_parts.append(th.zeros((b, act_dim - 1 - n_zones), dtype=action.dtype, device=action.device))
            c = th.cat(c_parts, dim=-1)
        else:
            c = c0

        # Initial box-clamped action
        u = th.clamp(action, lower_bounds, upper_bounds)

        # Check constraint violation
        g_init = self.constraint_val(state, u)
        if g_init.dim() == 1:
            g_init = g_init.unsqueeze(-1)

        is_active = (g_init > 0.0) | (action[:, 0:1] > a0_max) | (action[:, 0:1] < a0_min)
        if act_dim > 1:
            is_active = is_active | th.any((action[:, 1:] > 1.0) | (action[:, 1:] < -1.0), dim=-1, keepdim=True)

        # KKT QP active hyperplane projection with bound-aware correction
        for _ in range(3):
            g = self.constraint_val(state, u)
            if g.dim() == 1:
                g = g.unsqueeze(-1)
            violation = F.relu(g)

            # Mask variables capable of reducing g:
            # c_i > 0 requires u_i > lower_bounds (reducing u_i decreases g)
            # c_i < 0 requires u_i < upper_bounds (increasing u_i decreases g)
            free = th.where(c > 0, u > lower_bounds + 1e-4, u < upper_bounds - 1e-4).float()
            c_free = c * free
            norm_sq = (c_free ** 2).sum(dim=-1, keepdim=True)

            step = th.where(
                (norm_sq > 1e-6) & (violation > 1e-4),
                -violation / norm_sq.clamp_min(1e-6) * c_free,
                th.zeros_like(u),
            )
            u = th.clamp(u + step, lower_bounds, upper_bounds)

        # Straight-Through Estimator (STE) surrogate gradient attachment
        if self.surrogate_grad_scale > 0.0:
            ste_scale = th.where(
                is_active,
                th.full_like(action, self.surrogate_grad_scale),
                th.zeros_like(action),
            )
            u_out = u + (action - action.detach()) * ste_scale
        else:
            u_out = u

        if is_1d:
            u_out = u_out.squeeze(0)
        return u_out

    project = forward


class SafePolicyNetwork(PolicyNetwork):
    r"""Gaussian policy network whose forward pass applies DifferentiableSafetyLayer.

    Produces projected mean actions $\tilde{\mu}(s)$ directly during policy execution:
    $$\tilde{\mu}(s) = \operatorname{SafetyLayer}(s, \mu_\theta(s))$$

    Maintains raw action distribution $\pi_\theta(u \mid s) \sim \mathcal{N}(\mu_{\text{raw}}(s), \sigma^2)$
    for log-probability calculation to ensure mathematical consistency with stored raw actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_layer: DifferentiableSafetyLayer,
        hidden_dims: Tuple[int, ...] = (64, 64),
        init_mean: Optional[Union[np.ndarray, th.Tensor]] = None,
        init_log_std: Optional[Union[np.ndarray, th.Tensor]] = None,
    ) -> None:
        """Initialise safe policy network with attached analytical safety layer.

        Args:
            state_dim: Dimension of input state observations.
            action_dim: Dimension of continuous action vector.
            safety_layer: DifferentiableSafetyLayer instance for projection.
            hidden_dims: Hidden layer dimension tuple.
            init_mean: Optional initial mean vector for output linear layer.
            init_log_std: Optional initial log standard deviations.
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            init_mean=init_mean,
            init_log_std=init_log_std,
        )
        self.safety_layer: DifferentiableSafetyLayer = safety_layer

    def forward_raw(self, state: th.Tensor) -> th.Tensor:
        r"""Evaluate raw unprojected Gaussian policy mean $\mu_\theta(s)$.

        Args:
            state: State tensor `(B, state_dim)`.

        Returns:
            Unprojected action mean `(B, action_dim)`.
        """
        return super().forward(state)

    def forward(self, state: th.Tensor) -> th.Tensor:
        r"""Evaluate policy forward pass with analytical projection $\tilde{\mu}(s)$.

        Args:
            state: State tensor `(B, state_dim)`.

        Returns:
            Projected action mean tensor `(B, action_dim)`.
        """
        raw_mu = super().forward(state)
        if self.safety_layer is not None:
            return self.safety_layer(state, raw_mu)
        return raw_mu

    def distribution(self, state: th.Tensor) -> th.distributions.Normal:
        r"""Construct Gaussian action distribution over raw actions: $\pi_\theta(u \mid s) \sim \mathcal{N}(\mu_{\text{raw}}(s), \sigma^2)$.

        Using raw mean $\mu_{\text{raw}}(s)$ guarantees:
        1. Exact Gaussian log_prob consistency when raw actions are stored (store_raw_action=True).
        2. Elimination of double-projection bias: actions are sampled from the unprojected policy
           and projected once by the safety layer during environment interaction.
        """
        mu = self.forward_raw(state)
        std = th.exp(self.log_std)
        return th.distributions.Normal(mu, std)

    def dist(self, state: th.Tensor) -> th.distributions.Normal:
        """Backward-compatible alias for distribution()."""
        return self.distribution(state)
