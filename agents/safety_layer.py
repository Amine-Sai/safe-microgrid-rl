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


class DifferentiableSafetyLayer(nn.Module):
    r"""Analytical safety layer for linear peak-demand constraint (Dalal et al., 2018).

    Implements a closed-form Euclidean projection of the continuous battery dispatch action
    $a_0 \in [-1, 1]$ to guarantee that the peak grid import cap constraint is satisfied:

    $$g(s, a) = P_{\text{net}}(s) + P_{\text{max}} \cdot a_0 - P_{\text{cap}} \le 0$$

    where:
        - $P_{\text{net}}(s) = \sum_{z=2}^6 \text{load}_z(s) - P_{\text{PV}}(s)$ is the net load in kW.
        - $a_0 > 0$ denotes charging (increasing grid draw).
        - $a_0 < 0$ denotes discharging (reducing grid draw).
        - $P_{\text{max}}$ is the rated battery power in kW.
        - $P_{\text{cap}}$ is the peak grid import threshold in kW.

    The minimal Euclidean projection problem is formulated as:
    $$\min_{\tilde{a}} \frac{1}{2} \|\tilde{a} - a\|_2^2 \quad \text{s.t.} \quad g(s, \tilde{a}) \le 0, \quad -1 \le \tilde{a}_0 \le 1$$

    Since only $a_0$ affects the linear constraint $g(s, a)$ with constant Jacobian
    $\nabla_a g = [P_{\text{max}}, 0, \dots, 0]^\top$, the analytical closed-form solution is:

    $$\tilde{a}_0 = \operatorname{clip}\left(a_0 - \frac{\operatorname{ReLU}(g(s, a))}{P_{\text{max}}}, -1.0, 1.0\right)$$

    Gradients $\frac{\partial \tilde{a}}{\partial a}$ flow smoothly through standard PyTorch
    operations (`F.relu`, `th.clamp`), providing direct policy gradient corrections:
    $$\nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \tilde{a}} \cdot \frac{\partial \tilde{a}}{\partial a} \cdot \frac{\partial a}{\partial \theta}$$
    """

    def __init__(
        self,
        p_max: float = 250.0,
        p_cap: float = 368.153918,
        obs_normalizer: Optional[object] = None,
    ) -> None:
        """Initialise the differentiable analytical safety projection layer.

        Args:
            p_max: Battery rated maximum charging/discharging power in kW.
            p_cap: Grid import capacity threshold in kW.
            obs_normalizer: Optional RunningMeanStd tracker to map normalized states back to kW.
        """
        super().__init__()
        self.p_max: float = float(p_max)
        self.p_cap: float = float(p_cap)
        self.obs_normalizer = obs_normalizer

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
        # obs[1] is PV generation, obs[2..6] are zone loads
        pv = raw_state[:, 1:2]
        loads = raw_state[:, 2:7].sum(dim=-1, keepdim=True)
        return loads - pv

    def constraint_val(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        r"""Compute linear constraint function $g(s, a) = P_{\text{net}} + P_{\text{max}} a_0 - P_{\text{cap}}$.

        Args:
            state: State observation tensor `(B, state_dim)` or `(state_dim,)`.
            action: Action tensor `(B, action_dim)` or `(action_dim,)`.

        Returns:
            Constraint evaluation tensor; positive indicates constraint violation.
        """
        is_1d = action.dim() == 1
        if is_1d:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)

        p_net = self.get_p_net(state)
        a0 = action[:, 0:1]
        g = p_net + self.p_max * a0 - self.p_cap
        return g.squeeze(0) if is_1d else g

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        r"""Project continuous action to satisfy the linear grid cap constraint.

        $$\tilde{a}_0 = \operatorname{clip}\left(a_0 - \frac{\operatorname{ReLU}(g(s, a))}{P_{\text{max}}}, -1.0, 1.0\right)$$

        Args:
            state: Environment state observation tensor `(B, state_dim)` or `(state_dim,)`.
            action: Continuous action tensor `(B, action_dim)` or `(action_dim,)`.

        Returns:
            Projected action tensor of identical shape and device.
        """
        is_1d = action.dim() == 1
        if is_1d:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)

        # Align devices if necessary
        if state.device != action.device:
            state = state.to(action.device)

        p_net = self.get_p_net(state)
        a0 = action[:, 0:1]
        g = p_net + self.p_max * a0 - self.p_cap

        # Dalal et al. (2018) analytical correction:
        correction = F.relu(g) / self.p_max
        a0_tilde = th.clamp(a0 - correction, -1.0, 1.0)

        if action.shape[1] > 1:
            a_tilde = th.cat([a0_tilde, action[:, 1:]], dim=-1)
        else:
            a_tilde = a0_tilde

        if is_1d:
            a_tilde = a_tilde.squeeze(0)
        return a_tilde

    project = forward


class SafePolicyNetwork(PolicyNetwork):
    r"""Gaussian policy network whose forward pass applies DifferentiableSafetyLayer.

    Produces projected mean actions $\tilde{\mu}(s)$ directly during policy execution:
    $$\tilde{\mu}(s) = \operatorname{SafetyLayer}(s, \mu_\theta(s))$$
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
