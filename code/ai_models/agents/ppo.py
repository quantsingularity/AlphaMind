"""
Proximal Policy Optimisation (PPO) agent via Stable-Baselines3.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from ai_models.agents.base import BaseAgent
from ai_models.config import PPOConfig

logger = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """
    PPO agent for multi-asset portfolio management.

    Wraps ``stable_baselines3.PPO`` with a consistent AlphaMind interface.

    Parameters
    ----------
    env    : A Gymnasium-compatible environment.
    config : ``PPOConfig`` instance.
    """

    def __init__(self, env, config: Optional[PPOConfig] = None) -> None:
        from stable_baselines3 import PPO

        self.env = env
        cfg = config or PPOConfig()
        self.cfg = cfg

        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            verbose=0,
        )

    def select_action(self, state, add_noise: bool = False) -> np.ndarray:
        action, _ = self.model.predict(state, deterministic=not add_noise)
        return action

    def update(self):
        """Not used directly; training is handled by SB3 internally."""
        return None

    def train(self, num_episodes: int = 0, max_steps: int = 0) -> Dict:
        """
        Train via ``model.learn``.

        Parameters
        ----------
        num_episodes : Ignored (use *total_timesteps* via :meth:`learn`).
        max_steps    : Total environment timesteps when called directly.
        """
        total = max_steps or 1_000_000
        logger.info("PPO training: %d timesteps...", total)
        self.model.learn(total_timesteps=total)
        logger.info("PPO training complete.")
        return {}

    def learn(self, total_timesteps: int = 1_000_000) -> None:
        """Convenience wrapper around ``model.learn``."""
        logger.info("PPO learning: %d timesteps...", total_timesteps)
        self.model.learn(total_timesteps=total_timesteps)

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        rewards, lengths = [], []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            ep_r, steps, done = 0.0, 0, False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = self.env.step(action)
                ep_r += float(r)
                steps += 1
                done = done or truncated
            rewards.append(ep_r)
            lengths.append(steps)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_episode_length": float(np.mean(lengths)),
        }

    def save_model(self, path: str) -> None:
        self.model.save(path)
        logger.info("PPO model saved to %s", path)

    def load_model(self, path: str) -> None:
        from stable_baselines3 import PPO

        self.model = PPO.load(path, env=self.env)
        logger.info("PPO model loaded from %s", path)

    @classmethod
    def from_pretrained(cls, path: str, env) -> "PPOAgent":
        """Load a previously saved PPO model."""
        agent = cls.__new__(cls)
        from stable_baselines3 import PPO

        agent.model = PPO.load(path, env=env)
        agent.env = env
        return agent
