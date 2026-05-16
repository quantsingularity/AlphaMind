"""
DDPG TradingEnvironment
-----------------------
A discrete-step gym.Env where the agent submits a continuous weight vector
and receives a portfolio return minus transaction costs as reward.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from ai_models.config import TradingEnvConfig
from ai_models.environments.base import BaseTradingEnv
from gymnasium import spaces


class TradingEnvironment(BaseTradingEnv, gym.Env):
    """
    Single-step portfolio trading environment compatible with OpenAI Gym.

    Observation
    -----------
    Dict with keys ``prices`` (n_assets x window), ``volumes`` (n_assets,),
    and ``macro`` (n_macro,).

    Action
    ------
    Continuous vector in [-1, 1]^n_assets, normalised to portfolio weights.

    Reward
    ------
    ``portfolio_return - transaction_cost``
    """

    metadata: Dict = {"render_modes": ["human"]}

    def __init__(self, config: Optional[TradingEnvConfig] = None, **kwargs) -> None:
        super().__init__()
        cfg = config or TradingEnvConfig(
            **{
                k: v
                for k, v in kwargs.items()
                if k in TradingEnvConfig.__dataclass_fields__
            }
        )
        self.cfg = cfg
        self.n_assets = cfg.n_assets
        self.window = cfg.window
        self.n_macro = cfg.n_macro
        self.transaction_cost = cfg.transaction_cost
        self.max_steps = cfg.max_steps
        self.current_step = 0
        self.returns = np.zeros((cfg.max_steps, cfg.n_assets), dtype=np.float32)
        self.current_weights = np.ones(cfg.n_assets, dtype=np.float32) / cfg.n_assets

        self.action_space = spaces.Box(-1.0, 1.0, (cfg.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "prices": spaces.Box(
                    -np.inf, np.inf, (cfg.n_assets, cfg.window), dtype=np.float32
                ),
                "volumes": spaces.Box(0.0, np.inf, (cfg.n_assets,), dtype=np.float32),
                "macro": spaces.Box(-np.inf, np.inf, (cfg.n_macro,), dtype=np.float32),
            }
        )

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
        reward = float(self._port_return(new_weights) - cost)
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        return self._get_obs(), reward, done, False, {}

    def _port_return(self, weights: np.ndarray) -> float:
        if self.current_step >= len(self.returns):
            return 0.0
        return float(np.dot(self.returns[self.current_step], weights))

    def _normalize(self, action: np.ndarray) -> np.ndarray:
        w = np.tanh(action)
        denom = np.abs(w).sum()
        if denom < 1e-8:
            return np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        return (w / denom).astype(np.float32)

    def _tc(self, new_weights: np.ndarray) -> float:
        turnover = float(np.abs(new_weights - self.current_weights).sum())
        self.current_weights = new_weights.copy()
        return turnover * self.transaction_cost

    def _get_obs(self) -> Dict[str, np.ndarray]:
        start = max(0, self.current_step - self.window)
        pw = self.returns[start : self.current_step + 1]
        if len(pw) < self.window:
            pad = np.zeros((self.window - len(pw), self.n_assets), dtype=np.float32)
            pw = np.vstack([pad, pw])
        return {
            "prices": pw[-self.window :].T.astype(np.float32),
            "volumes": np.abs(np.random.normal(1000, 500, (self.n_assets,))).astype(
                np.float32
            ),
            "macro": np.random.normal(0.0, 1.0, (self.n_macro,)).astype(np.float32),
        }
