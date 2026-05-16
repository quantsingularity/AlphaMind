"""Abstract base class for AlphaMind trading environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseTradingEnv(ABC):
    """
    Minimal interface that all AlphaMind environments must satisfy.
    Concrete environments inherit from this *and* from ``gym.Env`` /
    ``gymnasium.Env`` as appropriate.
    """

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Any, Dict]:
        """Reset to initial state."""

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute one step."""

    @abstractmethod
    def _get_obs(self) -> Any:
        """Return current observation."""
