"""
PPO PortfolioGymEnv
-------------------
Gymnasium environment for multi-asset portfolio management.
Reward is a rolling Sharpe-ratio approximation minus transaction costs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from ai_models.config import PortfolioEnvConfig
from ai_models.environments.base import BaseTradingEnv
from gymnasium import spaces

logger = logging.getLogger(__name__)


class PortfolioGymEnv(BaseTradingEnv, gym.Env):
    """
    Gymnasium portfolio environment compatible with Stable-Baselines3.

    Parameters
    ----------
    universe         : List of asset ticker symbols.
    config           : ``PortfolioEnvConfig`` instance.
    """

    metadata: Dict = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        universe: List[str],
        config: Optional[PortfolioEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.universe = universe
        self.n_assets = len(universe)
        cfg = config or PortfolioEnvConfig()
        self.cfg = cfg

        self.action_space = spaces.Box(-1.0, 1.0, (self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "prices": spaces.Box(
                    -np.inf, np.inf, (self.n_assets, cfg.window), dtype=np.float32
                ),
                "volumes": spaces.Box(0.0, np.inf, (self.n_assets,), dtype=np.float32),
                "macro": spaces.Box(-np.inf, np.inf, (cfg.n_macro,), dtype=np.float32),
            }
        )

        self.current_step = 0
        self.max_steps = cfg.max_steps
        self.returns = np.zeros((cfg.max_steps, self.n_assets), dtype=np.float32)
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.returns = np.random.normal(
            0.0005, 0.01, (self.max_steps, self.n_assets)
        ).astype(np.float32)
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).flatten()
        new_weights = self._normalize(action)
        cost = self._tc(new_weights)
        reward = float(self._sharpe_reward(new_weights) - cost)
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        return self._get_obs(), reward, done, False, {}

    def _sharpe_reward(self, weights: np.ndarray) -> float:
        if self.current_step < 2:
            return 0.0
        start = max(0, self.current_step - 20)
        window = self.returns[start : self.current_step]
        port_r = np.dot(window, weights)
        rf = 0.02 / 252
        std = float(np.std(port_r))
        if std < 1e-9 or np.isnan(std):
            return -1.0
        return float((np.mean(port_r) - rf) / std)

    def _normalize(self, action: np.ndarray) -> np.ndarray:
        w = np.tanh(action)
        denom = np.abs(w).sum()
        if denom < 1e-8:
            return np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return (w / denom).astype(np.float32)

    def _tc(self, new_weights: np.ndarray) -> float:
        to = float(np.abs(new_weights - self.current_weights).sum())
        self.current_weights = new_weights.copy()
        return to * self.cfg.transaction_cost

    def _get_obs(self) -> Dict[str, np.ndarray]:
        start = max(0, self.current_step - self.cfg.window)
        pw = self.returns[start : self.current_step + 1]
        if len(pw) < self.cfg.window:
            pad = np.zeros((self.cfg.window - len(pw), self.n_assets), dtype=np.float32)
            pw = np.vstack([pad, pw])
        return {
            "prices": pw[-self.cfg.window :].T.astype(np.float32),
            "volumes": np.abs(np.random.normal(1000.0, 500.0, (self.n_assets,))).astype(
                np.float32
            ),
            "macro": np.random.normal(0.0, 1.0, (self.cfg.n_macro,)).astype(np.float32),
        }
