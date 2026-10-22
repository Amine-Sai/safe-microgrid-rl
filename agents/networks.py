import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class CriticNetwork(nn.Module):
    """Twin Q-network for state-action value estimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256), lr: float = 3e-4):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q = nn.Linear(hidden_dims[1], 1)

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: th.Tensor, action: th.Tensor) -> th.Tensor:
        x = th.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class ActorNetwork(nn.Module):
    """Gaussian policy network with tanh squashing for continuous actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
        lr: float = 3e-4,
        max_action: float = 1.0,
    ):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mu = nn.Linear(hidden_dims[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[1], action_dim)

        self.optimizer = th.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: th.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = th.clamp(self.log_std_layer(x), min=-20, max=2)
        std = th.exp(log_std)
        return mu, std

    def sample_normal(self, state: th.Tensor, reparameterize: bool = True):
        mu, std = self.forward(state)
        dist = Normal(mu, std)

        u = dist.rsample() if reparameterize else dist.sample()
        action = th.tanh(u) * self.max_action

        # Enforcing Action Bounds (Haarnoja et al., 2018)
        # log pi(a|s) = log mu(u|s) - sum_i log(1 - tanh^2(u_i))
        log_prob = dist.log_prob(u)
        # Numerically stable adjustment:
        # log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u))
        log_prob -= 2.0 * (th.log(th.tensor(2.0, device=state.device)) - u - F.softplus(-2.0 * u))
        if self.max_action != 1.0:
            log_prob -= th.log(th.tensor(self.max_action, device=state.device))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state: th.Tensor) -> th.Tensor:
        mu, _ = self.forward(state)
        return th.tanh(mu) * self.max_action
