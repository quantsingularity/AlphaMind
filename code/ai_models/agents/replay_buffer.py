"""
Experience replay and exploration noise utilities.
"""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Tuple, Union

import numpy as np
import torch

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)
State = Union[dict, np.ndarray]


class ReplayBuffer:
    """
    Fixed-capacity circular experience-replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: State,
        action: np.ndarray,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer."""
        self.buffer.append(
            Experience(state, action, float(reward), next_state, bool(done))
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random mini-batch of transitions.

        Returns five tensors: states, actions, rewards, next_states, dones.
        If fewer than *batch_size* transitions are stored, all are returned.
        """
        experiences = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
        to_flat = self._flatten
        states = torch.FloatTensor([to_flat(e.state) for e in experiences])
        actions = torch.FloatTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(-1)
        next_states = torch.FloatTensor([to_flat(e.next_state) for e in experiences])
        dones = torch.FloatTensor([float(e.done) for e in experiences]).unsqueeze(-1)
        return states, actions, rewards, next_states, dones

    @staticmethod
    def _flatten(state: State) -> np.ndarray:
        if isinstance(state, dict):
            return np.concatenate([np.asarray(v).flatten() for v in state.values()])
        return np.asarray(state).flatten()

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}, capacity={self.buffer.maxlen})"


class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Parameters
    ----------
    size  : Dimensionality of the action space.
    mu    : Long-run mean of the process.
    theta : Mean-reversion rate.
    sigma : Diffusion coefficient.
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.20,
    ) -> None:
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state: np.ndarray = np.zeros(size)
        self.reset()

    def reset(self) -> None:
        """Reset the noise process to its long-run mean."""
        self.state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Sample a single noise vector."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.size
        )
        self.state = self.state + dx
        return self.state.copy()

    def __repr__(self) -> str:
        return f"OUNoise(size={self.size}, theta={self.theta}, sigma={self.sigma})"
