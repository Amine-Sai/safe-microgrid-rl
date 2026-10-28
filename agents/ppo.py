import os
from typing import Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from agents.running_stat import RunningMeanStd


def initial_grid_only_action_mean(shed_mapping: str, action_dim: int = 6) -> np.ndarray:
    """The action that reproduces Grid-Only, derived from the env's mapping rule.

    env/smartgrid_env.py::step parses the action as:
        relu/deadzone : shed = max(0, a)          -> a <= 0 sheds nothing
        linear        : shed = (a + 1) / 2        -> a = -1   sheds nothing
    and battery_cmd = action[0], so 0.0 means the battery is idle. Grid-Only is the
    battery idle plus no shedding, i.e. [0.0, a_noshed, ...].
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


def compute_gae(rewards, values, dones, truncated, last_value, gamma=0.99, lam=0.95):
    """GAE(lambda) with truncation-aware bootstrapping.

    `dones` must mark terminal transitions only. A truncation (time limit) still
    bootstraps from `last_value`, which is why it is passed in separately and never
    zeroed.

    Returns (advantages, returns), both plain numpy arrays of length T.
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


class RolloutBuffer:
    """Rollout buffer for on-policy PPO trajectories."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.values = []
        self.cost_values = []
        self.log_probs = []
        self.terminated = []
        self.truncated = []
        self.rollout_costs = []

    def add(
        self,
        state,
        action,
        reward,
        cost,
        value,
        cost_value,
        log_prob,
        terminated,
        truncated,
    ):
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

    def __len__(self):
        return len(self.rewards)


class PolicyNetwork(nn.Module):
    """Gaussian policy: state-independent log-std, continuous actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=(64, 64),
        init_mean=None,
        init_log_std=None,
    ):
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
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.mu(x)

    def distribution(self, state: th.Tensor) -> th.distributions.Normal:
        mu = self.forward(state)
        std = th.exp(self.log_std)
        return th.distributions.Normal(mu, std)

    def dist(self, state: th.Tensor) -> th.distributions.Normal:
        """Alias for distribution()."""
        return self.distribution(state)


_PolicyNet = PolicyNetwork


class ValueNetwork(nn.Module):
    """State-value function critic."""

    def __init__(self, state_dim: int, hidden_dims=(64, 64)):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.v = nn.Linear(hidden_dims[1], 1)

    def forward(self, state: th.Tensor) -> th.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)


_ValueNet = ValueNetwork


class PPOAgent:
    """Base Proximal Policy Optimization (PPO) agent."""

    def __new__(cls, *args, **kwargs):
        if cls is PPOAgent:
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
        init_log_std_shed=None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shed_mapping = shed_mapping
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.use_minibatches = use_minibatches
        self.value_epochs = int(value_epochs)
        self.store_raw_action = store_raw_action
        self.last_raw_action = None
        self.rng = np.random.RandomState(0)
        self.device = th.device(device or "cpu")

        # Compatibility flags for unconstrained base PPO
        self.lagrangian = False
        self.pbrs = False
        self.residual_policy = None
        self.residual_coef = 0.0
        self.cost_limit = float(kwargs.get("cost_limit", 0.05))
        self.lambda_ = 0.0
        self.cost_net = None

        ls = np.full(action_dim, init_log_std, dtype=np.float32)
        if init_log_std_shed is not None and action_dim > 1:
            ls[1:] = init_log_std_shed

        init_mean = self.get_initial_policy_mean()
        self.policy = PolicyNetwork(
            state_dim, action_dim, init_mean=init_mean, init_log_std=ls
        ).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        self.policy_optimizer = th.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = th.optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.obs_normalizer = RunningMeanStd(shape=(state_dim,))
        self.ret_normalizer = RunningMeanStd(shape=())
        self.cost_normalizer = RunningMeanStd(shape=())

        self.buffer = RolloutBuffer()

    def get_initial_policy_mean(self) -> np.ndarray:
        """Initial policy mean. Vanilla PPO starts at Grid-Only (idle battery, 0% shed)."""
        return initial_grid_only_action_mean(self.shed_mapping, self.action_dim)

    # ---------------------------------------------------------------- buffer
    def reset_buffer(self):
        self.buffer.reset()

    _reset_buffer = reset_buffer

    @property
    def buf(self):
        """Backward-compatibility mapping for self.buf."""
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
        }

    @property
    def rollout_costs(self):
        return self.buffer.rollout_costs

    @rollout_costs.setter
    def rollout_costs(self, value):
        self.buffer.rollout_costs = list(value)

    @property
    def _last_u(self):
        return self.last_raw_action

    @_last_u.setter
    def _last_u(self, value):
        self.last_raw_action = value

    @property
    def _rng(self):
        return self.rng

    @_rng.setter
    def _rng(self, value):
        self.rng = value

    @property
    def _shed_mapping(self):
        return self.shed_mapping

    @_shed_mapping.setter
    def _shed_mapping(self, value):
        self.shed_mapping = value

    def store_transition(
        self,
        state,
        action,
        reward,
        cost,
        value,
        cost_value,
        log_prob,
        terminated,
        truncated,
    ):
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
        )

    # ---------------------------------------------------------------- acting
    def to_tensor(self, x):
        return th.as_tensor(x, dtype=th.float32, device=self.device)

    _t = to_tensor

    def denormalize_value(self, v):
        """Denormalise a value-net output into true reward units."""
        std = float(np.sqrt(self.ret_normalizer.var) + self.ret_normalizer.epsilon)
        return v * std + float(self.ret_normalizer.mean)

    _denorm_v = denormalize_value

    def denormalize_cost_value(self, v):
        std = float(np.sqrt(self.cost_normalizer.var) + self.cost_normalizer.epsilon)
        return v * std + float(self.cost_normalizer.mean)

    _denorm_c = denormalize_cost_value

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """Returns (action, log_prob, value, cost_value)."""
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
    def scale_reward(self, reward, state=None, next_state=None, terminated=False, truncated=False):
        """Apply explicit reward scale c = 1/mean|r|."""
        return float(reward) * self.reward_scale

    shape = scale_reward

    def compute_potential(self, state):
        return 0.0

    _phi = compute_potential

    # ---------------------------------------------------------------- learning
    def finish_path(self, last_value, last_cost_value=0.0):
        """Compute GAE for the buffered rollout and return the training batch."""
        b = self.buffer
        T = len(b.rewards)
        if T == 0:
            return None
        adv_r, ret_r = compute_gae(
            b.rewards, b.values, b.terminated, b.truncated, last_value, self.gamma, self.gae_lambda
        )
        out = {
            "states": np.asarray(b.states, dtype=np.float32),
            "actions": np.asarray(b.actions, dtype=np.float32),
            "logp": np.asarray(b.log_probs, dtype=np.float32),
            "adv_r": adv_r,
            "ret_r": ret_r,
        }
        if self.lagrangian and self.cost_net is not None:
            adv_c, ret_c = compute_gae(
                b.costs, b.cost_values, b.terminated, b.truncated, last_cost_value, self.gamma, self.gae_lambda
            )
            out.update(adv_c=adv_c, ret_c=ret_c)
        return out

    def update_lambda_from_source(self):
        return self.lambda_

    def fit_critics(self, s_t, ret_norm, retc_norm=None):
        """Fit critic(s). Unifies full-batch and minibatch modes."""
        n = s_t.shape[0]
        steps = 0
        v_loss = None
        for _ in range(self.value_epochs):
            if self.use_minibatches and self.batch_size < n:
                perm = th.as_tensor(self.rng.permutation(n), device=self.device)
                for st in range(0, n, self.batch_size):
                    i = perm[st : st + self.batch_size]
                    v_loss = F.mse_loss(self.value_net(s_t[i]), ret_norm[i])
                    self.value_optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    self.value_optimizer.step()
                    if retc_norm is not None and self.cost_net is not None:
                        c_loss = F.mse_loss(self.cost_net(s_t[i]), retc_norm[i])
                        self.cost_optimizer.zero_grad()
                        c_loss.backward()
                        nn.utils.clip_grad_norm_(self.cost_net.parameters(), self.max_grad_norm)
                        self.cost_optimizer.step()
                    steps += 1
            else:
                v_loss = F.mse_loss(self.value_net(s_t), ret_norm)
                self.value_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                if retc_norm is not None and self.cost_net is not None:
                    c_loss = F.mse_loss(self.cost_net(s_t), retc_norm)
                    self.cost_optimizer.zero_grad()
                    c_loss.backward()
                    nn.utils.clip_grad_norm_(self.cost_net.parameters(), self.max_grad_norm)
                    self.cost_optimizer.step()
                steps += 1
        return float(v_loss.detach()) if v_loss is not None else 0.0, steps

    _fit_critics_mb = fit_critics

    def update_policy(self, s_t, a_t, old_logp, adv_t):
        """Policy optimization epochs with KL early stopping."""
        n = s_t.shape[0]
        steps = 0
        loss_v = 0.0
        ent = kl = clipf = 0.0
        for _ in range(self.n_epochs):
            if self.use_minibatches and self.batch_size < n:
                perm = th.as_tensor(self.rng.permutation(n), device=self.device)
                for st in range(0, n, self.batch_size):
                    i = perm[st : st + self.batch_size]
                    d = self.policy.distribution(s_t[i])
                    logp = d.log_prob(a_t[i]).sum(dim=-1, keepdim=True)
                    ratio = th.exp(logp - old_logp[i])
                    ai = adv_t[i]
                    surr = th.min(
                        ratio * ai,
                        th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * ai,
                    )
                    loss = -surr.mean() - self.entropy_coef * d.entropy().mean()
                    self.policy_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy_optimizer.step()
                    steps += 1
                    loss_v = float(loss.detach())
            else:
                d = self.policy.distribution(s_t)
                logp = d.log_prob(a_t).sum(dim=-1, keepdim=True)
                ratio = th.exp(logp - old_logp)
                surr = th.min(
                    ratio * adv_t,
                    th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t,
                )
                loss = -surr.mean() - self.entropy_coef * d.entropy().mean()
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

    def combine_advantages(self, batch):
        """Combine reward and cost advantages. Overridden in PPOLagrangianAgent."""
        adv = batch["adv_r"].copy()
        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, None

    def update(self, batch):
        """One PPO update over the rollout batch. Returns diagnostic metrics."""
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
        pg_loss_val, ent_val, kl_val, clip_frac_val, n_pol_steps = self.update_policy(s_t, a_t, old_logp, adv_t)

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

    def end_episode(self, terminated):
        """Record the episode's mean cost; clear rollout costs."""
        if not hasattr(self, "cost_history"):
            self.cost_history = []
        if self.buffer.rollout_costs:
            self.cost_history.append(float(np.mean(self.buffer.rollout_costs)))
        self.buffer.rollout_costs = []

    # ---------------------------------------------------------------- io
    def save_models(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
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
        }
        if self.cost_net is not None:
            payload["cost_net"] = self.cost_net.state_dict()
        th.save(payload, path)

    def load_models(self, path: str):
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
        """Reconstruct the matching agent subclass from a saved checkpoint."""
        payload = th.load(path, map_location="cpu")
        state_dim = int(payload["policy"]["fc1.weight"].shape[1])
        action_dim = int(payload["policy"]["mu.bias"].shape[0])
        shed_mapping = payload.get("shed_mapping", "linear")
        residual = bool(payload.get("residual", False))
        lagrangian = bool(payload.get("lagrangian", False))
        pbrs = bool(payload.get("pbrs", False))
        residual_coef = float(payload.get("residual_coef", 0.25))

        if residual:
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


class PPOLagrangianAgent(PPOAgent):
    """PPO with Lagrangian dual ascent for constraint satisfaction."""

    def __init__(
        self,
        *args,
        cost_limit: float = 0.05,
        lr_dual: float = 0.01,
        lambda_init: float = 0.0,
        lambda_lr_mult: float = 0.1,
        lag_normalize_by_1p_lam: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lagrangian = True
        self.cost_limit = float(cost_limit)
        self.lag_normalize_by_1p_lam = bool(lag_normalize_by_1p_lam)
        self.cost_net = ValueNetwork(self.state_dim).to(self.device)
        lr_val = kwargs.get("lr_value", 1e-3)
        self.cost_optimizer = th.optim.Adam(self.cost_net.parameters(), lr=lr_val)

        self.lambda_ = float(max(0.0, lambda_init))
        self.lambda_lr = float(lr_dual * lambda_lr_mult)
        self.lambda_history = []
        self.cost_history = []

    def combine_advantages(self, batch):
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

    def update_lambda_from_source(self):
        recent = self.cost_history[-10:] if self.cost_history else self.buffer.rollout_costs
        j_c = float(np.mean(recent)) if recent else 0.0
        self.lambda_ = max(0.0, self.lambda_ + self.lambda_lr * (j_c - self.cost_limit))
        self.lambda_history.append(self.lambda_)
        return self.lambda_


class PPOResidualAgent(PPOLagrangianAgent):
    """PPO that learns a residual correction on top of a fixed base policy."""

    def __init__(
        self,
        *args,
        residual_policy=None,
        residual_coef: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if residual_policy is None:
            from agents.heuristics import psm_prime_policy
            residual_policy = psm_prime_policy()
        self.residual_policy = residual_policy
        self.residual_coef = float(residual_coef)

    def get_initial_policy_mean(self) -> np.ndarray:
        """Residual policy mean starts at 0.0 (zero correction over the base policy)."""
        return np.zeros(self.action_dim, dtype=np.float32)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
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


class PPOShieldedAgent(PPOAgent):
    """PPO with 1-D battery action paired with CapShield for analytical safety."""

    def __init__(self, *args, action_dim: int = 1, **kwargs):
        super().__init__(*args, action_dim=action_dim, **kwargs)


class PPOPotentialShapingAgent(PPOAgent):
    """PPO + Potential-Based Reward Shaping (PBRS)."""

    def __init__(
        self,
        *args,
        potential_shaping_kappa: float = 105.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pbrs = True
        self.kappa = float(potential_shaping_kappa)
        from agents.shaping import PotentialBasedShaping
        self.shaping = PotentialBasedShaping(gamma=self.gamma, kappa=1.0)

    def compute_potential(self, state):
        return float(self.shaping.phi(np.asarray(state, dtype=np.float32)))

    _phi = compute_potential

    def scale_reward(self, reward, state=None, next_state=None, terminated=False, truncated=False):
        scaled = float(reward) * self.reward_scale
        if state is None or next_state is None:
            return scaled
        phi_s = self.compute_potential(state)
        phi_ns = 0.0 if terminated else self.compute_potential(next_state)
        return scaled + self.reward_scale * self.kappa * (self.gamma * phi_ns - phi_s)

    shape = scale_reward


class PPOLagrangianPotentialShapingAgent(PPOLagrangianAgent, PPOPotentialShapingAgent):
    """PPO with both Lagrangian dual ascent and potential-based reward shaping."""

    def __init__(self, *args, potential_shaping_kappa: float = 105.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbrs = True
        self.kappa = float(potential_shaping_kappa)
        from agents.shaping import PotentialBasedShaping
        self.shaping = PotentialBasedShaping(gamma=self.gamma, kappa=1.0)

    compute_potential = PPOPotentialShapingAgent.compute_potential
    _phi = compute_potential
    scale_reward = PPOPotentialShapingAgent.scale_reward
    shape = scale_reward
