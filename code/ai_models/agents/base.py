"""Abstract base class shared by all AlphaMind RL agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class BaseAgent(ABC):
    """
    Abstract RL agent interface.

    All concrete agents must implement ``select_action``, ``update``,
    ``train``, ``evaluate``, ``save_model``, and ``load_model``.
    """

    @abstractmethod
    def select_action(
        self,
        state: Union[Dict, np.ndarray],
        add_noise: bool = True,
    ) -> np.ndarray:
        """Return an action for the given state."""

    @abstractmethod
    def update(self) -> Optional[Tuple[float, ...]]:
        """Perform one gradient-update step. Returns losses or None."""

    @abstractmethod
    def train(self, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        """Full training loop. Returns a history dict."""

    @abstractmethod
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate without exploration. Returns metric dict."""

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Persist weights to *path*."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Restore weights from *path*."""
