r"""Proximal Policy Optimization (PPO) and Safe RL Variants for Microgrid Control.

This module provides a unified, modular implementation of PPO and its safe / constrained
extensions for continuous microgrid energy management:

1. PolicyNetwork & ValueNetwork:
   Parametric Gaussian actor with state-independent log-std and scalar state-value critic:
   $$\pi_\theta(a | s) = \mathcal{N}(\mu_\theta(s), \operatorname{diag}(\sigma^2))$$
   $$V_\phi(s) \approx \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \mid s_0 = s \right]$$

2. Generalized Advantage Estimation (GAE):
   Truncation-aware temporal difference advantage estimation:
   $$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$
   $$\hat{A}_t = \delta_t + (\gamma \lambda) (1 - d_t) \hat{A}_{t+1}$$

3. Clipped Surrogate Policy Objective:
   $$\mathcal{L}_{\text{CLIP}}(\theta) = -\hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \,
   \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
   where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$.

4. Constrained RL via Lagrangian Dual Relaxation (PPOLagrangianAgent):
   $$\min_\theta \max_{\lambda \ge 0} \mathcal{L}(\theta, \lambda) = J_R(\pi_\theta) - \lambda (J_C(\pi_\theta) - d)$$
   $$\hat{A}_{\text{net}} = \frac{\hat{A}_R - \lambda \hat{A}_C}{1 + \lambda}$$
   $$\lambda_{k+1} = \max\left(0, \lambda_k + \alpha_\lambda (\bar{J}_C - d)\right)$$

5. Residual Policy Learning (PPOResidualAgent):
   $$a = \operatorname{clip}\left(a_{\text{base}}(s) + \alpha_{\text{res}} u, -1.0, 1.0\right), \quad u \sim \pi_\theta(s)$$

6. Auxiliary Shield Imitation Loss (PPOAuxShieldLossAgent):
   $$\mathcal{L}_{\text{aux}} = \mathbb{E}\left[ \mathbb{I}(\text{shield\_shed} > 0) \cdot (\mu_\theta(s)_0 - (-1.0))^2 \right]$$

7. Differentiable Analytical Safety Layer (PPODiffSafetyLayerAgent):
   Analytical QP projection layer embedded directly in the PyTorch execution graph.

8. Potential-Based Reward Shaping (PPOPotentialShapingAgent):
   $$F(s, s') = \gamma \Phi(s') - \Phi(s), \quad \Phi(s) = \kappa \cdot \text{SoC}(s)$$
"""

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from agents.running_stat import RunningMeanStd


# ============================================================================
# Section 1: Utility Functions & GAE
# ============================================================================

def initial_grid_only_action_mean(shed_mapping: str, action_dim: int = 6) -> np.ndarray:
    r"""Compute the deterministic continuous action reproducing Grid-Only baseline dispatch.

    In the microgrid environment:
        - action[0] is the battery command: 0.0 represents idle battery.
        - action[1..5] are zone shedding commands:
          * relu/deadzone mapping: $\text{shed} = \max(0, a) \implies a \le 0$ commands 0% shed.
          * linear mapping: $\text{shed} = \frac{a + 1}{2} \implies a = -1$ commands 0% shed.

    Args:
        shed_mapping: Zone shedding mapping rule ('linear', 'relu', or 'deadzone').
        action_dim: Dimensionality of action vector (default 6).

    Returns:
        Action array of shape (action_dim,) reproducing idle battery and 100% served load.
    """
    if shed_mapping in ("relu", "deadzone"):
        no_shed = 0.0
    else:
        no_shed = -1.0
    a = np.full(action_dim, no_shed, dtype=np.float32)
    a[0] = 0.0
    return a


# Backward-compatible alias
grid_only_action = initial_grid_only_action_mean


def compute_gae(
    rewards: Union[np.ndarray, List[float]],
    values: Union[np.ndarray, List[float]],
    dones: Union[np.ndarray, List[bool]],
    truncated: Union[np.ndarray, List[bool]],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute Generalized Advantage Estimation (GAE) with truncation-aware bootstrapping.

    Given trajectory transitions of length $T$:
    $$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$
    $$\hat{A}_t = \delta_t + (\gamma \lambda) (1 - d_t) \hat{A}_{t+1}$$
    $$R_t = \hat{A}_t + V(s_t)$$

    Truncation Handling:
        `dones` marks true terminal transitions only. Episode truncations (fixed horizon
        cuts at 168 hours) still bootstrap from $V(s_T) = \text{last\_value}$ rather than 0.

    Args:
        rewards: Sequence of step rewards of length T.
        values: Sequence of critic state-value estimates V(s_t) of length T.
        dones: Terminal flag sequence of length T (True if absorbing transition).
        truncated: Truncation flag sequence of length T (True if time-limit cut).
        last_value: Critic value estimate for the bootstrap state V(s_T).
        gamma: Temporal discount factor $\gamma \in (0, 1]$.
        lam: GAE exponential decay factor $\lambda \in [0, 1]$.

    Returns:
        Tuple of (advantages, returns), both float64 numpy arrays of length T.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        not_terminal = 0.0 if dones[t] else 1.0
        next_value = last_value if t == T - 1 else values[t + 1]
        if truncated[t] and not (t == T - 1):
            next_value = values[t + 1] if not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value * not_terminal - values[t]
        gae = delta + gamma * lam * not_terminal * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float64)
    return adv, returns


# ============================================================================
# Section 2: Rollout Transition Storage
# ============================================================================

class RolloutBuffer:
    """In-memory transition buffer for on-policy PPO trajectory rollouts."""

    def __init__(self) -> None:
        """Initialise empty rollout buffer."""
        self.reset()

    def reset(self) -> None:
        """Clear all stored transition trajectories."""
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.costs: List[float] = []
        self.values: List[float] = []
        self.cost_values: List[float] = []
        self.log_probs: List[float] = []
        self.terminated: List[bool] = []
        self.truncated: List[bool] = []
        self.rollout_costs: List[float] = []
        self.shield_shed_kws: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        value: float,
        cost_value: float,
        log_prob: float,
        terminated: bool,
        truncated: bool,
        shield_shed_kw: float = 0.0,
    ) -> None:
        """Append one transition tuple to the rollout buffer."""
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.costs.append(float(cost))
        self.values.append(float(value))
        self.cost_values.append(float(cost_value))
        self.log_probs.append(float(log_prob))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.rollout_costs.append(float(cost))
        self.shield_shed_kws.append(float(shield_shed_kw))

    def __len__(self) -> int:
        """Return number of transitions currently buffered."""
        return len(self.rewards)


# ============================================================================
# Section 3: Neural Network Architectures
# ============================================================================

class PolicyNetwork(nn.Module):
    r"""Gaussian actor network with state-independent log standard deviations.

    Maps environment state $s \in \mathbb{R}^{d_s}$ to Gaussian action distribution parameters:
    $$\mu_\theta(s) = \operatorname{Linear}(\operatorname{ReLU}(\operatorname{Linear}(\operatorname{ReLU}(\operatorname{Linear}(s)))))$$
    $$\sigma = \exp(\log \sigma)$$
    $$\pi_\theta(a \mid s) \sim \mathcal{N}(\mu_\theta(s), \operatorname{diag}(\sigma^2))$$
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        init_mean: Optional[Union[np.ndarray, th.Tensor]] = None,
        init_log_std: Optional[Union[np.ndarray, th.Tensor, float]] = None,
    ) -> None:
        """Initialise Gaussian policy MLP.

        Args:
            state_dim: Dimension of environment observation vector.
            action_dim: Dimension of continuous action vector.
            hidden_dims: Tuple of hidden layer units.
            init_mean: Optional mean vector for zero-weight initialization of output layer.
            init_log_std: Optional initial log standard deviation values.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mu = nn.Linear(hidden_dims[1], action_dim)

        ls = th.full((action_dim,), -1.0, dtype=th.float32)
        if init_log_std is not None:
            ls = th.as_tensor(init_log_std, dtype=th.float32).clone()
        self.log_std = nn.Parameter(ls)

        if init_mean is not None:
            self.mu.weight.data.zero_()
            self.mu.bias.data.copy_(th.as_tensor(init_mean, dtype=th.float32))

    def forward(self, state: th.Tensor) -> th.Tensor:
        r"""Compute the deterministic action mean $\mu_\theta(s)$."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.mu(x)

    def distribution(self, state: th.Tensor) -> th.distributions.Normal:
        r"""Construct diagonal Gaussian action distribution $\pi_\theta(\cdot \mid s)$."""
        mu = self.forward(state)
        std = th.exp(self.log_std)
        return th.distributions.Normal(mu, std)

    def dist(self, state: th.Tensor) -> th.distributions.Normal:
        """Backward-compatible alias for distribution()."""
        return self.distribution(state)


_PolicyNet = PolicyNetwork


class ValueNetwork(nn.Module):
    r"""Critic MLP estimating scalar state-value $V(s)$.

    Architecture:
    $$V_\phi(s) = \operatorname{Linear}(\operatorname{ReLU}(\operatorname{Linear}(\operatorname{ReLU}(\operatorname{Linear}(s)))))$$
    """

    def __init__(self, state_dim: int, hidden_dims: Tuple[int, ...] = (64, 64)) -> None:
        """Initialise value network MLP.

        Args:
            state_dim: Dimension of environment observation vector.
            hidden_dims: Tuple of hidden layer units.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.v = nn.Linear(hidden_dims[1], 1)

    def forward(self, state: th.Tensor) -> th.Tensor:
        r"""Compute scalar state value estimate $V_\phi(s)$."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)


_ValueNet = ValueNetwork


# ============================================================================
# Section 4: Base Proximal Policy Optimization (PPO) Agent
# ============================================================================

class PPOAgent:
    """Base Clipped Proximal Policy Optimization (PPO) agent (Schulman et al., 2017).

    Implements:
    - Clipped surrogate policy objective with advantage normalization.
    - Value function regression with empirical return normalization.
    - Welford running observation normalization.
    - KL divergence monitoring and early stopping.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "PPOAgent":
        """Polymorphic instantiation routing to appropriate subclass based on kwargs."""
        if cls is PPOAgent:
            method = str(kwargs.get("method", ""))
            if method == "ppo_shield_aux":
                return super().__new__(PPOAuxShieldLossAgent)
            if method == "ppo_diff_shield":
                return super().__new__(PPODiffSafetyLayerAgent)

            is_residual = (
                kwargs.get("residual_policy") is not None
                or kwargs.get("residual_on_heuristic", False)
            )
            is_lag = kwargs.get("lagrangian", False)
            is_pbrs = kwargs.get("pbrs", False) or kwargs.get("potential_shaping", False)
            is_shielded = kwargs.get("shielded", False) or kwargs.get("action_dim") == 1

            if is_residual:
                return super().__new__(PPOResidualAgent)
            if is_shielded:
                return super().__new__(PPOShieldedAgent)
            if is_lag and is_pbrs:
                return super().__new__(PPOLagrangianPotentialShapingAgent)
            if is_pbrs:
                return super().__new__(PPOPotentialShapingAgent)
            if is_lag:
                return super().__new__(PPOLagrangianAgent)
        return super().__new__(cls)

    def __init__(
        self,
        state_dim: int = 17,
        action_dim: int = 6,
        shed_mapping: str = "linear",
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.0,
        target_kl: float = 0.05,
        n_epochs: int = 10,
        batch_size: int = 64,
        reward_scale: float = 1.0,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        use_minibatches: bool = False,
        value_epochs: int = 1,
        store_raw_action: bool = True,
        init_log_std: float = -1.0,
        init_log_std_shed: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise PPO agent parameters, neural networks, optimizers, and accumulators."""
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.shed_mapping: str = shed_mapping
        self.gamma: float = float(gamma)
        self.gae_lambda: float = float(gae_lambda)
        self.clip_ratio: float = float(clip_ratio)
        self.entropy_coef: float = float(entropy_coef)
        self.target_kl: float = float(target_kl)
        self.n_epochs: int = int(n_epochs)
        self.batch_size: int = int(batch_size)
        self.reward_scale: float = float(reward_scale)
        self.max_grad_norm: float = float(max_grad_norm)
        self.normalize_advantage: bool = bool(normalize_advantage)
        self.use_minibatches: bool = bool(use_minibatches)
        self.value_epochs: int = int(value_epochs)
        self.store_raw_action: bool = bool(store_raw_action)
        self.last_raw_action: Optional[np.ndarray] = None
        self.rng: np.random.RandomState = np.random.RandomState(0)
        self.device: th.device = th.device(device or "cpu")

        # Compatibility flags for unconstrained base PPO
        self.lagrangian: bool = False
        self.pbrs: bool = False
        self.residual_policy: Optional[Any] = None
        self.residual_coef: float = 0.0
        self.cost_limit: float = float(kwargs.get("cost_limit", 0.05))
        self.lambda_: float = 0.0
        self.cost_net: Optional[ValueNetwork] = None

        ls = np.full(action_dim, init_log_std, dtype=np.float32)
        if init_log_std_shed is not None and action_dim > 1:
            ls[1:] = init_log_std_shed

        init_mean = self.get_initial_policy_mean()
        self.policy: PolicyNetwork = PolicyNetwork(
            state_dim, action_dim, init_mean=init_mean, init_log_std=ls
        ).to(self.device)
        self.value_net: ValueNetwork = ValueNetwork(state_dim).to(self.device)

        self.policy_optimizer: th.optim.Optimizer = th.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer: th.optim.Optimizer = th.optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.obs_normalizer: RunningMeanStd = RunningMeanStd(shape=(state_dim,))
        self.ret_normalizer: RunningMeanStd = RunningMeanStd(shape=())
        self.cost_normalizer: RunningMeanStd = RunningMeanStd(shape=())

        self.buffer: RolloutBuffer = RolloutBuffer()

    def get_initial_policy_mean(self) -> np.ndarray:
        """Compute initial policy mean reproducing Grid-Only baseline."""
        return initial_grid_only_action_mean(self.shed_mapping, self.action_dim)

    # ---------------------------------------------------------------- buffer
    def reset_buffer(self) -> None:
        """Reset the internal rollout buffer."""
        self.buffer.reset()

    _reset_buffer = reset_buffer

    @property
    def buf(self) -> Dict[str, List[Any]]:
        """Backward-compatibility mapping for legacy dict-based buffer access."""
        return {
            "s": self.buffer.states,
            "a": self.buffer.actions,
            "r": self.buffer.rewards,
            "c": self.buffer.costs,
            "v": self.buffer.values,
            "cv": self.buffer.cost_values,
            "logp": self.buffer.log_probs,
            "term": self.buffer.terminated,
            "trunc": self.buffer.truncated,
            "shield_shed_kw": getattr(self.buffer, "shield_shed_kws", []),
        }

    @property
    def rollout_costs(self) -> List[float]:
        """Rollout constraint costs accumulator."""
        return self.buffer.rollout_costs

    @rollout_costs.setter
    def rollout_costs(self, value: Any) -> None:
        self.buffer.rollout_costs = list(value)

    @property
    def _last_u(self) -> Optional[np.ndarray]:
        return self.last_raw_action

    @_last_u.setter
    def _last_u(self, value: Optional[np.ndarray]) -> None:
        self.last_raw_action = value

    @property
    def _rng(self) -> np.random.RandomState:
        return self.rng

    @_rng.setter
    def _rng(self, value: np.random.RandomState) -> None:
        self.rng = value

    @property
    def _shed_mapping(self) -> str:
        return self.shed_mapping

    @_shed_mapping.setter
    def _shed_mapping(self, value: str) -> None:
        self.shed_mapping = value

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        value: float,
        cost_value: float,
        log_prob: float,
        terminated: bool,
        truncated: bool,
        shield_shed_kw: float = 0.0,
    ) -> None:
        """Store a transition tuple, preserving raw unclipped action for on-policy consistency."""
        raw = self.last_raw_action if (self.store_raw_action and self.last_raw_action is not None) else action
        self.buffer.add(
            state=state,
            action=raw,
            reward=reward,
            cost=cost,
            value=value,
            cost_value=cost_value,
            log_prob=log_prob,
            terminated=terminated,
            truncated=truncated,
            shield_shed_kw=shield_shed_kw,
        )

    # ---------------------------------------------------------------- acting
    def to_tensor(self, x: Any) -> th.Tensor:
        """Convert array-like input to float32 PyTorch tensor on agent device."""
        return th.as_tensor(x, dtype=th.float32, device=self.device)

    _t = to_tensor

    def denormalize_value(self, v: th.Tensor) -> th.Tensor:
        """Denormalise value prediction from standard normal space to true return units."""
        std = float(np.sqrt(self.ret_normalizer.var) + self.ret_normalizer.epsilon)
        return v * std + float(self.ret_normalizer.mean)

    _denorm_v = denormalize_value

    def denormalize_cost_value(self, v: th.Tensor) -> th.Tensor:
        """Denormalise cost value prediction to true discounted cost units."""
        std = float(np.sqrt(self.cost_normalizer.var) + self.cost_normalizer.epsilon)
        return v * std + float(self.cost_normalizer.mean)

    _denorm_c = denormalize_cost_value

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> Tuple[np.ndarray, float, float, float]:
        r"""Select continuous action $a \in [-1, 1]^d$ and query critics.

        Args:
            state: Raw environment observation vector of shape (state_dim,).
            evaluate: If True, executes deterministically using policy mean.

        Returns:
            Tuple of (clamped_action, log_prob, value_estimate, cost_value_estimate).
        """
        if not evaluate:
            self.obs_normalizer.update(state[None, :])
        s = self.to_tensor(self.obs_normalizer.normalize(state[None, :]))
        with th.no_grad():
            if evaluate:
                u = self.policy.forward(s)
                logp = th.zeros((s.shape[0],), device=self.device)
            else:
                dist = self.policy.distribution(s)
                u = dist.sample()
                logp = dist.log_prob(u).sum(dim=-1)

            a = th.clamp(u, -1.0, 1.0)
            v = self.denormalize_value(self.value_net(s))
            cv_raw = self.cost_net(s) if self.cost_net is not None else th.zeros_like(v)
            cv = self.denormalize_cost_value(cv_raw) if self.cost_net is not None else cv_raw

        if not evaluate:
            self.last_raw_action = u.squeeze(0).cpu().numpy().astype(np.float32)

        return (
            a.squeeze(0).cpu().numpy().astype(np.float32),
            float(logp),
            float(v),
            float(cv),
        )

    # ---------------------------------------------------------------- reward scaling
    def scale_reward(
        self,
        reward: float,
        state: Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> float:
        r"""Scale reward by multiplier $c = 1 / \mathbb{E}[|r|]$."""
        return float(reward) * self.reward_scale

    shape = scale_reward

    def compute_potential(self, state: np.ndarray) -> float:
        r"""Compute potential function $\Phi(s)$ (zero for unshaped base PPO)."""
        return 0.0

    _phi = compute_potential

    # ---------------------------------------------------------------- learning
    def finish_path(
        self,
        last_value: float,
        last_cost_value: float = 0.0,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Compute GAE advantages across buffered rollout and return batch dictionary."""
        b = self.buffer
        T = len(b.rewards)
        if T == 0:
            return None
        adv_r, ret_r = compute_gae(
            b.rewards, b.values, b.terminated, b.truncated, last_value, self.gamma, self.gae_lambda
        )
        shed_kws = getattr(b, "shield_shed_kws", None)
        if shed_kws is None or len(shed_kws) != T:
            shed_arr = np.zeros(T, dtype=np.float32)
        else:
            shed_arr = np.asarray(shed_kws, dtype=np.float32)
        out: Dict[str, np.ndarray] = {
            "states": np.asarray(b.states, dtype=np.float32),
            "actions": np.asarray(b.actions, dtype=np.float32),
            "logp": np.asarray(b.log_probs, dtype=np.float32),
            "adv_r": adv_r,
            "ret_r": ret_r,
            "shield_shed_kw": shed_arr,
        }
        if self.lagrangian and self.cost_net is not None:
            adv_c, ret_c = compute_gae(
                b.costs, b.cost_values, b.terminated, b.truncated, last_cost_value, self.gamma, self.gae_lambda
            )
            out.update(adv_c=adv_c, ret_c=ret_c)
        return out

    def update_lambda_from_source(self) -> float:
        r"""Update dual multiplier $\lambda$ (no-op in unconstrained PPO)."""
        return self.lambda_

    def _get_batch_indices(self, n: int) -> Iterator[Union[slice, th.Tensor]]:
        """Yield mini-batch indices or full-batch slice, preserving seed determinism."""
        if self.use_minibatches and self.batch_size < n:
            perm = th.as_tensor(self.rng.permutation(n), device=self.device)
            for st in range(0, n, self.batch_size):
                yield perm[st : st + self.batch_size]
        else:
            yield slice(None)

    def compute_value_loss(self, state: th.Tensor, target_return: th.Tensor) -> th.Tensor:
        r"""Compute MSE loss for reward critic: $\mathbb{E}[(V_\phi(s) - R)^2]$."""
        return F.mse_loss(self.value_net(state), target_return)

    def compute_cost_value_loss(self, state: th.Tensor, target_cost: th.Tensor) -> th.Tensor:
        r"""Compute MSE loss for cost critic: $\mathbb{E}[(V_C(s) - R_C)^2]$."""
        if self.cost_net is None:
            raise RuntimeError("Cost value network is not initialized.")
        return F.mse_loss(self.cost_net(state), target_cost)

    def fit_critics(
        self,
        s_t: th.Tensor,
        ret_norm: th.Tensor,
        retc_norm: Optional[th.Tensor] = None,
    ) -> Tuple[float, int]:
        """Fit critic networks via gradient descent on normalized returns."""
        n = s_t.shape[0]
        steps = 0
        v_loss_val = 0.0

        for _ in range(self.value_epochs):
            for idx in self._get_batch_indices(n):
                s_b = s_t[idx]
                ret_b = ret_norm[idx]

                v_loss = self.compute_value_loss(s_b, ret_b)
                self.value_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                if retc_norm is not None and self.cost_net is not None:
                    c_loss = self.compute_cost_value_loss(s_b, retc_norm[idx])
                    self.cost_optimizer.zero_grad()
                    c_loss.backward()
                    nn.utils.clip_grad_norm_(self.cost_net.parameters(), self.max_grad_norm)
                    self.cost_optimizer.step()

                steps += 1
                v_loss_val = float(v_loss.detach())

        return v_loss_val, steps

    _fit_critics_mb = fit_critics

    def compute_surrogate_loss(
        self,
        logp: th.Tensor,
        old_logp: th.Tensor,
        adv: th.Tensor,
    ) -> th.Tensor:
        r"""Compute clipped surrogate policy objective:

        $$\mathcal{L}_{\text{CLIP}}(\theta) = -\hat{\mathbb{E}} \left[ \min\left( r_t(\theta) \hat{A}_t, \,
        \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
        """
        ratio = th.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        return -th.min(surr1, surr2).mean()

    def compute_policy_loss(
        self,
        s_b: th.Tensor,
        a_b: th.Tensor,
        old_lp_b: th.Tensor,
        adv_b: th.Tensor,
        dist: th.distributions.Normal,
        weights_b: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        r"""Compute scalar policy loss: $\mathcal{L}_{\text{CLIP}} - c_{\text{ent}} \mathcal{H}(\pi)$."""
        logp = dist.log_prob(a_b).sum(dim=-1, keepdim=True)
        surr_loss = self.compute_surrogate_loss(logp, old_lp_b, adv_b)
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        return surr_loss + entropy_loss

    def update_policy(
        self,
        s_t: th.Tensor,
        a_t: th.Tensor,
        old_logp: th.Tensor,
        adv_t: th.Tensor,
        shield_shed_weights: Optional[th.Tensor] = None,
    ) -> Tuple[float, float, float, float, int]:
        """Perform PPO policy optimization epochs with KL early stopping."""
        n = s_t.shape[0]
        steps = 0
        loss_v = 0.0
        ent = kl = clipf = 0.0

        for _ in range(self.n_epochs):
            for idx in self._get_batch_indices(n):
                s_b = s_t[idx]
                a_b = a_t[idx]
                old_lp_b = old_logp[idx]
                adv_b = adv_t[idx]
                w_b = shield_shed_weights[idx] if shield_shed_weights is not None else None

                d = self.policy.distribution(s_b)
                loss = self.compute_policy_loss(s_b, a_b, old_lp_b, adv_b, d, weights_b=w_b)

                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                steps += 1
                loss_v = float(loss.detach())

            with th.no_grad():
                d = self.policy.distribution(s_t)
                lp = d.log_prob(a_t).sum(dim=-1, keepdim=True)
                kl = float((old_logp - lp).mean())
                clipf = float((th.abs(th.exp(lp - old_logp) - 1.0) > self.clip_ratio).float().mean())
                ent = float(d.entropy().mean())

            if self.target_kl and kl > 1.5 * self.target_kl:
                break

        return loss_v, ent, kl, clipf, steps

    _policy_epochs_mb = update_policy

    def combine_advantages(
        self,
        batch: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Combine reward and cost advantages (overridden in PPOLagrangianAgent)."""
        adv = batch["adv_r"].copy()
        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, None

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Execute one complete PPO training update over the buffered rollout."""
        states = self.obs_normalizer.normalize(batch["states"])
        s_t = self.to_tensor(states)
        a_t = self.to_tensor(batch["actions"])
        old_logp = self.to_tensor(batch["logp"]).view(-1, 1)

        with th.no_grad():
            lp0 = self.policy.distribution(s_t).log_prob(a_t).sum(dim=-1, keepdim=True)
            ratio_off_at_start = float((th.abs(th.exp(lp0 - old_logp) - 1.0) > self.clip_ratio).float().mean())

        adv, norm_ratio = self.combine_advantages(batch)
        adv_t = self.to_tensor(adv).view(-1, 1)
        ret_t = self.to_tensor(batch["ret_r"]).view(-1, 1)

        self.ret_normalizer.update(np.asarray(batch["ret_r"], dtype=np.float64))
        v_std = float(np.sqrt(self.ret_normalizer.var) + self.ret_normalizer.epsilon)
        v_mean = float(self.ret_normalizer.mean)
        ret_norm = (ret_t - v_mean) / v_std

        if self.lagrangian and "ret_c" in batch and self.cost_net is not None:
            retc_t = self.to_tensor(batch["ret_c"]).view(-1, 1)
            self.cost_normalizer.update(np.asarray(batch["ret_c"], dtype=np.float64))
            c_std = float(np.sqrt(self.cost_normalizer.var) + self.cost_normalizer.epsilon)
            c_mean = float(self.cost_normalizer.mean)
            retc_norm = (retc_t - c_mean) / c_std
        else:
            retc_norm = None

        v_loss_val, n_val_steps = self.fit_critics(s_t, ret_norm, retc_norm)
        shed_weights = getattr(self, "_current_shield_shed_weights", None)
        pg_loss_val, ent_val, kl_val, clip_frac_val, n_pol_steps = self.update_policy(
            s_t, a_t, old_logp, adv_t, shield_shed_weights=shed_weights
        )

        frac_sat = float(np.mean(np.abs(batch["actions"]) >= 0.99))
        with th.no_grad():
            ev_val = 1.0 - float(F.mse_loss(self.denormalize_value(self.value_net(s_t)), ret_t) / (ret_t.var() + 1e-8))

        mean_abs_adv = float(np.abs(batch["adv_r"]).mean())

        return {
            "pg_loss": pg_loss_val,
            "value_loss": v_loss_val,
            "entropy": ent_val,
            "approx_kl": kl_val,
            "frac_saturated": frac_sat,
            "lag_adv_norm_ratio": norm_ratio,
            "clip_frac": clip_frac_val,
            "explained_variance": ev_val,
            "mean_abs_adv": mean_abs_adv,
            "log_std": self.policy.log_std.detach().cpu().numpy().tolist(),
            "n_policy_grad_steps": n_pol_steps,
            "n_value_grad_steps": n_val_steps,
            "ratio_off_at_start": ratio_off_at_start,
        }

    def end_episode(self, terminated: bool) -> None:
        """Record the finished episode mean constraint cost; clear rollout costs."""
        if not hasattr(self, "cost_history"):
            self.cost_history: List[float] = []
        if self.buffer.rollout_costs:
            self.cost_history.append(float(np.mean(self.buffer.rollout_costs)))
        self.buffer.rollout_costs = []

    # ---------------------------------------------------------------- io
    def save_models(self, path: str) -> None:
        """Serialise agent model weights, normalisers, and hyperparameters to checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload: Dict[str, Any] = {
            "agent_class": self.__class__.__name__,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "obs_mean": self.obs_normalizer.mean.tolist(),
            "obs_var": self.obs_normalizer.var.tolist(),
            "obs_count": float(self.obs_normalizer.count),
            "ret_mean": float(self.ret_normalizer.mean),
            "ret_var": float(self.ret_normalizer.var),
            "ret_count": float(self.ret_normalizer.count),
            "cost_mean": float(self.cost_normalizer.mean),
            "cost_var": float(self.cost_normalizer.var),
            "cost_count": float(self.cost_normalizer.count),
            "lambda": float(self.lambda_),
            "lagrangian": bool(self.lagrangian),
            "pbrs": bool(self.pbrs),
            "residual": bool(self.residual_policy is not None),
            "residual_coef": float(self.residual_coef),
            "shed_mapping": getattr(self, "shed_mapping", "linear"),
            "aux_shield_coef": float(getattr(self, "aux_shield_coef", 0.5)),
            "p_max": float(getattr(self, "p_max", 250.0)),
            "p_cap": float(getattr(self, "p_cap", 368.153918)),
        }
        if self.cost_net is not None:
            payload["cost_net"] = self.cost_net.state_dict()
        th.save(payload, path)

    def load_models(self, path: str) -> None:
        """Restore neural network parameters and normaliser statistics from checkpoint."""
        payload = th.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["policy"])
        self.value_net.load_state_dict(payload["value_net"])
        self.obs_normalizer.mean = np.array(payload["obs_mean"], dtype=np.float32)
        self.obs_normalizer.var = np.array(payload["obs_var"], dtype=np.float32)
        self.obs_normalizer.count = float(payload["obs_count"])
        for prefix, norm in (("ret_", self.ret_normalizer), ("cost_", self.cost_normalizer)):
            if prefix + "mean" in payload:
                norm.mean = np.array(payload[prefix + "mean"], dtype=np.float64)
                norm.var = np.array(payload[prefix + "var"], dtype=np.float64)
                norm.count = float(payload[prefix + "count"])
        self.lambda_ = float(payload.get("lambda", 0.0))
        if self.cost_net is not None and "cost_net" in payload:
            self.cost_net.load_state_dict(payload["cost_net"])

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[str] = None) -> "PPOAgent":
        """Reconstruct matching agent subclass and restore state from saved checkpoint."""
        payload = th.load(path, map_location="cpu")
        state_dim = int(payload["policy"]["fc1.weight"].shape[1])
        action_dim = int(payload["policy"]["mu.bias"].shape[0])
        shed_mapping = payload.get("shed_mapping", "linear")
        residual = bool(payload.get("residual", False))
        lagrangian = bool(payload.get("lagrangian", False))
        pbrs = bool(payload.get("pbrs", False))
        residual_coef = float(payload.get("residual_coef", 0.25))
        agent_cls = payload.get("agent_class")

        if agent_cls == "PPOAuxShieldLossAgent":
            agent: PPOAgent = PPOAuxShieldLossAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                shed_mapping=shed_mapping,
                aux_shield_coef=payload.get("aux_shield_coef", 0.5),
                device=device,
            )
        elif agent_cls == "PPODiffSafetyLayerAgent":
            agent = PPODiffSafetyLayerAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                shed_mapping=shed_mapping,
                p_max=payload.get("p_max", 250.0),
                p_cap=payload.get("p_cap", 368.153918),
                device=device,
            )
        elif residual:
            from agents.heuristics import psm_prime_policy
            agent = PPOResidualAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                shed_mapping=shed_mapping,
                residual_policy=psm_prime_policy(),
                residual_coef=residual_coef,
                device=device,
            )
        elif action_dim == 1:
            agent = PPOShieldedAgent(
                state_dim=state_dim,
                shed_mapping=shed_mapping,
                device=device,
            )
        elif lagrangian and pbrs:
            agent = PPOLagrangianPotentialShapingAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                shed_mapping=shed_mapping,
                device=device,
            )
        elif pbrs:
            agent = PPOPotentialShapingAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                shed_mapping=shed_mapping,
                device=device,
            )
        elif lagrangian:
            agent = PPOLagrangianAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                shed_mapping=shed_mapping,
                device=device,
            )
        else:
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                shed_mapping=shed_mapping,
                device=device,
            )

        agent.load_models(path)
        return agent


# ============================================================================
# Section 5: Constrained PPO with Lagrangian Dual Ascent
# ============================================================================

class PPOLagrangianAgent(PPOAgent):
    r"""PPO with Lagrangian dual relaxation for Constrained Markov Decision Processes (CMDPs).

    Formulation:
    $$\min_\theta J_R(\pi_\theta) \quad \text{s.t.} \quad J_C(\pi_\theta) \le d$$

    Unconstrained Lagrangian saddle-point problem:
    $$\max_{\lambda \ge 0} \min_\theta \mathcal{L}(\theta, \lambda) = J_R(\pi_\theta) - \lambda (J_C(\pi_\theta) - d)$$

    Advantage combination:
    $$\hat{A}_{\text{net}} = \frac{\hat{A}_R - \lambda \hat{A}_C}{1 + \lambda}$$

    Dual variable update (projected subgradient ascent):
    $$\lambda_{k+1} = \max\left(0, \lambda_k + \alpha_\lambda (\bar{J}_C - d)\right)$$
    """

    def __init__(
        self,
        *args: Any,
        cost_limit: float = 0.05,
        lr_dual: float = 0.01,
        lambda_init: float = 0.0,
        lambda_lr_mult: float = 0.1,
        lag_normalize_by_1p_lam: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise PPOLagrangianAgent with cost critic and dual multiplier."""
        super().__init__(*args, **kwargs)
        self.lagrangian: bool = True
        self.cost_limit: float = float(cost_limit)
        self.lag_normalize_by_1p_lam: bool = bool(lag_normalize_by_1p_lam)
        self.cost_net: ValueNetwork = ValueNetwork(self.state_dim).to(self.device)
        lr_val = kwargs.get("lr_value", 1e-3)
        self.cost_optimizer: th.optim.Optimizer = th.optim.Adam(self.cost_net.parameters(), lr=lr_val)

        self.lambda_: float = float(max(0.0, lambda_init))
        self.lambda_lr: float = float(lr_dual * lambda_lr_mult)
        self.lambda_history: List[float] = []
        self.cost_history: List[float] = []

    def combine_advantages(
        self,
        batch: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Optional[float]]:
        r"""Combine reward and constraint cost advantages via dual multiplier $\lambda$."""
        adv_r = batch["adv_r"]
        adv_c = batch["adv_c"]
        adv = adv_r - self.lambda_ * adv_c
        if self.lag_normalize_by_1p_lam and (1.0 + self.lambda_) > 1e-8:
            adv = adv / (1.0 + self.lambda_)

        denom = float(np.abs(adv_r).mean()) + 1e-12
        norm_ratio = float(np.abs(self.lambda_ * adv_c).mean()) / denom

        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, norm_ratio

    def update_lambda_from_source(self) -> float:
        r"""Update dual multiplier $\lambda \leftarrow \max(0, \lambda + \alpha_\lambda (\bar{J}_C - d))$."""
        recent = self.cost_history[-10:] if self.cost_history else self.buffer.rollout_costs
        j_c = float(np.mean(recent)) if recent else 0.0
        self.lambda_ = max(0.0, self.lambda_ + self.lambda_lr * (j_c - self.cost_limit))
        self.lambda_history.append(self.lambda_)
        return self.lambda_


# ============================================================================
# Section 6: Residual Policy Agent
# ============================================================================

class PPOResidualAgent(PPOLagrangianAgent):
    r"""PPO agent learning a continuous residual perturbation on top of a fixed base policy.

    The composite action executed in the environment is:
    $$a(s) = \operatorname{clip}\left(a_{\text{base}}(s) + \alpha_{\text{res}} u, -1.0, 1.0\right), \quad u \sim \pi_\theta(s)$$

    The policy network mean is initialized to 0.0 so that training begins exactly at the
    performance of the base heuristic.
    """

    def __init__(
        self,
        *args: Any,
        residual_policy: Optional[Any] = None,
        residual_coef: float = 0.25,
        **kwargs: Any,
    ) -> None:
        """Initialise residual agent with base heuristic policy and scaling coefficient."""
        super().__init__(*args, **kwargs)
        if residual_policy is None:
            from agents.heuristics import psm_prime_policy
            residual_policy = psm_prime_policy()
        self.residual_policy = residual_policy
        self.residual_coef: float = float(residual_coef)

    def get_initial_policy_mean(self) -> np.ndarray:
        """Return zero initial mean, representing zero initial deviation from base policy."""
        return np.zeros(self.action_dim, dtype=np.float32)

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Sample residual correction and add to base heuristic action."""
        if not evaluate:
            self.obs_normalizer.update(state[None, :])
        s = self.to_tensor(self.obs_normalizer.normalize(state[None, :]))
        with th.no_grad():
            if evaluate:
                u = self.policy.forward(s)
                logp = th.zeros((s.shape[0],), device=self.device)
            else:
                dist = self.policy.distribution(s)
                u = dist.sample()
                logp = dist.log_prob(u).sum(dim=-1)

            base = self.residual_policy.get_action(state)
            a = th.clamp(self.to_tensor(base) + self.residual_coef * u, -1.0, 1.0)
            v = self.denormalize_value(self.value_net(s))
            cv_raw = self.cost_net(s) if self.cost_net is not None else th.zeros_like(v)
            cv = self.denormalize_cost_value(cv_raw) if self.cost_net is not None else cv_raw

        if not evaluate:
            self.last_raw_action = u.squeeze(0).cpu().numpy().astype(np.float32)

        return (
            a.squeeze(0).cpu().numpy().astype(np.float32),
            float(logp),
            float(v),
            float(cv),
        )


# ============================================================================
# Section 7: Shielded Action Agent
# ============================================================================

class PPOShieldedAgent(PPOAgent):
    """PPO agent with 1-D battery action paired with an external CapShield safety filter."""

    def __init__(self, *args: Any, action_dim: int = 1, **kwargs: Any) -> None:
        """Initialise 1-D battery dispatch agent."""
        super().__init__(*args, action_dim=action_dim, **kwargs)


# ============================================================================
# Section 8: Auxiliary Shield Imitation Loss Agent
# ============================================================================

class PPOAuxShieldLossAgent(PPOShieldedAgent):
    r"""PPO shielded agent with auxiliary loss penalizing battery inaction during shield interventions.

    When the analytical shield sheds load ($\text{shield\_shed\_kw} > 0$), the battery was
    insufficiently discharging to shave the peak. The auxiliary loss penalizes deviation
    from full discharge ($a_{\text{target}} = -1.0$):

    $$\mathcal{L}_{\text{aux}} = \mathbb{E}\left[ w \cdot (\mu_\theta(s)_0 - (-1.0))^2 \right]$$
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CLIP}} - c_{\text{ent}} \mathcal{H}(\pi) + c_{\text{aux}} \mathcal{L}_{\text{aux}}$$
    """

    def __init__(
        self,
        *args: Any,
        aux_shield_coef: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Initialise PPOAuxShieldLossAgent with auxiliary loss penalty coefficient."""
        super().__init__(*args, **kwargs)
        self.aux_shield_coef: float = float(aux_shield_coef)
        self._current_shield_shed_weights: Optional[th.Tensor] = None

    def compute_aux_loss(self, s_t: th.Tensor, shield_shed_weights: th.Tensor) -> th.Tensor:
        r"""Compute auxiliary imitation penalty: $\mathbb{E}[w \cdot (\mu_\theta(s)_0 - (-1.0))^2]$."""
        action_mean = self.policy.forward(s_t)[:, :1]
        target_action = -1.0
        weights = shield_shed_weights if shield_shed_weights.dtype.is_floating_point else shield_shed_weights.float()
        if weights.dim() == 1:
            weights = weights.view(-1, 1)
        return th.mean(weights * (action_mean - target_action) ** 2)

    def compute_policy_loss(
        self,
        s_b: th.Tensor,
        a_b: th.Tensor,
        old_lp_b: th.Tensor,
        adv_b: th.Tensor,
        dist: th.distributions.Normal,
        weights_b: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute policy loss augmented by the auxiliary shield imitation penalty."""
        base_loss = super().compute_policy_loss(s_b, a_b, old_lp_b, adv_b, dist)
        if weights_b is None:
            weights_b = self._current_shield_shed_weights
        if weights_b is not None:
            if weights_b.shape[0] != s_b.shape[0]:
                weights_b = weights_b[: s_b.shape[0]]
            aux_loss = self.compute_aux_loss(s_b, weights_b)
            return base_loss + self.aux_shield_coef * aux_loss
        return base_loss

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract shield shed weights from batch and execute PPO training update."""
        shed_kw = batch.get("shield_shed_kw")
        if shed_kw is not None:
            shed_kw_t = self.to_tensor(shed_kw).view(-1, 1)
            self._current_shield_shed_weights = (shed_kw_t > 0.0).float()
        else:
            self._current_shield_shed_weights = th.zeros((len(batch["states"]), 1), device=self.device)

        try:
            metrics = super().update(batch)
        finally:
            self._current_shield_shed_weights = None
        return metrics


# ============================================================================
# Section 9: Differentiable Analytical Safety Layer Agent
# ============================================================================

class PPODiffSafetyLayerAgent(PPOAgent):
    r"""PPO agent with an analytical differentiable safety projection layer (Dalal et al., 2018).

    Embeds closed-form QP projection directly into the policy network computation graph:
    $$\tilde{a}_0 = \operatorname{clip}\left(a_0 - \frac{\operatorname{ReLU}(g(s, a))}{P_{\text{max}}}, -1.0, 1.0\right)$$
    permitting end-to-end analytical gradient propagation from safety projection into policy weights.
    """

    def __init__(
        self,
        *args: Any,
        p_max: float = 250.0,
        p_cap: float = 368.153918,
        action_dim: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialise PPODiffSafetyLayerAgent with analytical projection layer."""
        super().__init__(*args, action_dim=action_dim, **kwargs)
        self.p_max: float = float(p_max)
        self.p_cap: float = float(p_cap)

        from agents.safety_layer import DifferentiableSafetyLayer, SafePolicyNetwork
        self.safety_layer: DifferentiableSafetyLayer = DifferentiableSafetyLayer(
            p_max=self.p_max,
            p_cap=self.p_cap,
            obs_normalizer=self.obs_normalizer,
        ).to(self.device)

        init_mean = self.get_initial_policy_mean()
        ls = np.full(self.action_dim, kwargs.get("init_log_std", -1.0), dtype=np.float32)
        self.policy: SafePolicyNetwork = SafePolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            safety_layer=self.safety_layer,
            init_mean=init_mean,
            init_log_std=ls,
        ).to(self.device)
        lr_policy = kwargs.get("lr_policy", 3e-4)
        self.policy_optimizer: th.optim.Optimizer = th.optim.Adam(self.policy.parameters(), lr=lr_policy)

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Select action projected analytically to satisfy peak import constraint."""
        if not evaluate:
            self.obs_normalizer.update(state[None, :])
        s = self.to_tensor(self.obs_normalizer.normalize(state[None, :]))
        with th.no_grad():
            if evaluate:
                u = self.policy.forward(s)
                logp = th.zeros((s.shape[0],), device=self.device)
                a = th.clamp(u, -1.0, 1.0)
            else:
                dist = self.policy.distribution(s)
                u = dist.sample()
                logp = dist.log_prob(u).sum(dim=-1)
                a = self.safety_layer(s, u)
                a = th.clamp(a, -1.0, 1.0)

            v = self.denormalize_value(self.value_net(s))
            cv_raw = self.cost_net(s) if self.cost_net is not None else th.zeros_like(v)
            cv = self.denormalize_cost_value(cv_raw) if self.cost_net is not None else cv_raw

        if not evaluate:
            self.last_raw_action = u.squeeze(0).cpu().numpy().astype(np.float32)

        return (
            a.squeeze(0).cpu().numpy().astype(np.float32),
            float(logp),
            float(v),
            float(cv),
        )


# ============================================================================
# Section 10: Potential-Based Reward Shaping Variants
# ============================================================================

class PPOPotentialShapingAgent(PPOAgent):
    r"""PPO with Potential-Based Reward Shaping (PBRS, Ng et al., 1999).

    Transforms step reward according to:
    $$r'(s, a, s') = c \cdot r(s, a, s') + c \cdot \kappa (\gamma \Phi(s') - \Phi(s))$$
    where $\Phi(s) = \text{SoC}(s)$.
    """

    def __init__(
        self,
        *args: Any,
        potential_shaping_kappa: float = 105.0,
        **kwargs: Any,
    ) -> None:
        r"""Initialise PPOPotentialShapingAgent with potential scale multiplier $\kappa$."""
        super().__init__(*args, **kwargs)
        self.pbrs: bool = True
        self.kappa: float = float(potential_shaping_kappa)
        from agents.shaping import PotentialBasedShaping
        self.shaping: PotentialBasedShaping = PotentialBasedShaping(gamma=self.gamma, kappa=1.0)

    def compute_potential(self, state: np.ndarray) -> float:
        r"""Compute state potential $\Phi(s) = \text{SoC}$."""
        return float(self.shaping.phi(np.asarray(state, dtype=np.float32)))

    _phi = compute_potential

    def scale_reward(
        self,
        reward: float,
        state: Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> float:
        """Scale reward and apply potential-difference shaping term."""
        scaled = float(reward) * self.reward_scale
        if state is None or next_state is None:
            return scaled
        phi_s = self.compute_potential(state)
        phi_ns = 0.0 if terminated else self.compute_potential(next_state)
        return scaled + self.reward_scale * self.kappa * (self.gamma * phi_ns - phi_s)

    shape = scale_reward


class PPOLagrangianPotentialShapingAgent(PPOLagrangianAgent, PPOPotentialShapingAgent):
    """PPO agent combining both Lagrangian dual ascent and potential-based reward shaping."""

    def __init__(self, *args: Any, potential_shaping_kappa: float = 105.0, **kwargs: Any) -> None:
        """Initialise joint Lagrangian and PBRS agent."""
        super().__init__(*args, **kwargs)
        self.pbrs: bool = True
        self.kappa: float = float(potential_shaping_kappa)
        from agents.shaping import PotentialBasedShaping
        self.shaping: PotentialBasedShaping = PotentialBasedShaping(gamma=self.gamma, kappa=1.0)

    compute_potential = PPOPotentialShapingAgent.compute_potential
    _phi = compute_potential
    scale_reward = PPOPotentialShapingAgent.scale_reward
    shape = scale_reward
