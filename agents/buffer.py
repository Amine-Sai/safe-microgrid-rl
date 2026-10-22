import numpy as np


class ReplayBuffer:
    """Replay buffer with truncation/termination separation and cost storage."""

    def __init__(self, max_size: int, input_shape: tuple, n_actions: int):
        self.mem_size = int(max_size)
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.cost_memory = np.zeros(self.mem_size, dtype=np.float32)

        # FIX #1: Separate terminated and truncated flags
        # terminated: the episode reached a true terminal state (bootstrap = 0)
        # truncated: the episode hit a time limit (bootstrap continues)
        # In our environment, terminated is always False.
        self.terminated_memory = np.zeros(self.mem_size, dtype=bool)
        self.truncated_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        cost: float = 0.0,
    ):
        """Store a transition with separate termination flags and cost."""
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.cost_memory[index] = cost
        self.terminated_memory[index] = terminated
        self.truncated_memory[index] = truncated

        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        """Sample a random batch. Returns terminated (for bootstrap zeroing)
        and truncated flags separately."""
        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.randint(0, max_mem, size=batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        costs = self.cost_memory[batch]
        terminated = self.terminated_memory[batch]
        truncated = self.truncated_memory[batch]

        return states, actions, rewards, next_states, terminated, truncated, costs

    def __len__(self):
        return min(self.mem_size, self.mem_cntr)

    @property
    def is_ready(self):
        """Whether the buffer has enough samples for a batch."""
        return self.mem_cntr >= 1
