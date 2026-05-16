"""
Deep Deterministic Policy Gradient (DDPG) trading agent.

Architecture:   Actor-Critic with target networks and experience replay.
Exploration:    Ornstein-Uhlenbeck process.
Optimisation:   Adam with gradient clipping on both networks.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ai_models.agents.base import BaseAgent
from ai_models.agents.replay_buffer import OUNoise, ReplayBuffer
from ai_models.config import DDPGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    """
    Policy network: maps observations to deterministic actions in [-1, 1].

    Parameters
    ----------
    state_dim   : Flattened observation dimensionality.
    action_dim  : Number of continuous actions.
    hidden_dims : Widths of the hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(state)


class Critic(nn.Module):
    """
    Value network: estimates Q(s, a) for a given (state, action) pair.

    Parameters
    ----------
    state_dim   : Flattened observation dimensionality.
    action_dim  : Number of continuous actions.
    hidden_dims : Widths of the hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim + action_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:  # noqa: D102
        return self.net(torch.cat([state, action], dim=-1))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DDPGTradingAgent(BaseAgent):
    """
    DDPG agent for continuous-action portfolio trading.

    Parameters
    ----------
    env    : A Gymnasium-compatible trading environment.
    config : ``DDPGConfig`` instance (or dict, for backward compatibility).
    """

    def __init__(self, env, config: Optional[Union[DDPGConfig, Dict]] = None) -> None:
        self.env = env

        if isinstance(config, dict):
            cfg = DDPGConfig(
                **{
                    k: v
                    for k, v in config.items()
                    if k in DDPGConfig.__dataclass_fields__
                }
            )
        elif config is None:
            cfg = DDPGConfig()
        else:
            cfg = config
        self.cfg = cfg

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu"
        )

        # Infer dimensions from environment
        sample = env.observation_space.sample()
        if isinstance(sample, dict):
            flat = np.concatenate([np.asarray(v).flatten() for v in sample.values()])
            state_dim = flat.shape[0]
        else:
            state_dim = int(np.prod(sample.shape))
        action_dim = int(np.prod(env.action_space.shape))

        self.actor = Actor(state_dim, action_dim, list(cfg.actor_hidden)).to(
            self.device
        )
        self.actor_target = Actor(state_dim, action_dim, list(cfg.actor_hidden)).to(
            self.device
        )
        self.critic = Critic(state_dim, action_dim, list(cfg.critic_hidden)).to(
            self.device
        )
        self.critic_target = Critic(state_dim, action_dim, list(cfg.critic_hidden)).to(
            self.device
        )

        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay_buffer = ReplayBuffer(cfg.buffer_size)
        self.noise = OUNoise(action_dim, sigma=cfg.noise_sigma, theta=cfg.noise_theta)

        self.episode_rewards: List[float] = []
        self.critic_losses: List[float] = []
        self.actor_losses: List[float] = []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(obs: Union[Dict, np.ndarray]) -> np.ndarray:
        if isinstance(obs, dict):
            return np.concatenate([np.asarray(v).flatten() for v in obs.values()])
        return np.asarray(obs).flatten()

    def _hard_update(self, target: nn.Module, source: nn.Module) -> None:
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        tau = self.cfg.tau
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(
        self, state: Union[Dict, np.ndarray], add_noise: bool = True
    ) -> np.ndarray:
        """Deterministic action + optional OU exploration noise."""
        flat = self._flatten(state)
        t = torch.FloatTensor(flat).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action: np.ndarray = self.actor(t).cpu().numpy().flatten()
        self.actor.train()
        if add_noise:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def update(self) -> Optional[Tuple[float, float]]:
        """
        One gradient step on actor and critic.

        Returns ``(critic_loss, actor_loss)`` or ``None`` when the replay
        buffer has fewer transitions than *batch_size*.
        """
        if len(self.replay_buffer) < self.cfg.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.cfg.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Critic loss (Bellman target)
        with torch.no_grad():
            next_a = self.actor_target(next_states)
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * self.critic_target(
                next_states, next_a
            )
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        # Actor loss (policy gradient)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item()

    def train(
        self,
        num_episodes: int = 1000,
        max_steps: int = 1000,
    ) -> Dict[str, List[float]]:
        """
        Full DDPG training loop.

        Returns
        -------
        dict
            Keys: ``episode_rewards``, ``critic_losses``, ``actor_losses``.
        """
        logger.info("DDPG training: %d episodes x %d steps", num_episodes, max_steps)
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            self.noise.reset()
            ep_reward = 0.0
            ep_c_loss = ep_a_loss = 0.0
            n_updates = 0

            for _ in range(max_steps):
                action = self.select_action(state, add_noise=True)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.replay_buffer.add(
                    state, action, float(reward), next_state, bool(done)
                )
                state = next_state
                ep_reward += float(reward)

                result = self.update()
                if result is not None:
                    ep_c_loss += result[0]
                    ep_a_loss += result[1]
                    n_updates += 1

                if done or truncated:
                    break

            self.episode_rewards.append(ep_reward)
            if n_updates > 0:
                self.critic_losses.append(ep_c_loss / n_updates)
                self.actor_losses.append(ep_a_loss / n_updates)

            if episode % 50 == 0:
                avg = np.mean(self.episode_rewards[-50:])
                logger.info(
                    "Episode %4d/%d | avg_reward=%.4f", episode, num_episodes, avg
                )

        logger.info("DDPG training complete.")
        return {
            "episode_rewards": self.episode_rewards,
            "critic_losses": self.critic_losses,
            "actor_losses": self.actor_losses,
        }

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Greedy evaluation (no exploration noise)."""
        rewards, lengths = [], []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            ep_r, steps, done = 0.0, 0, False
            while not done:
                action = self.select_action(state, add_noise=False)
                state, r, done, truncated, _ = self.env.step(action)
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
        """Persist actor, critic, and config to *path*."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        with open(os.path.join(path, "config.json"), "w") as f:
            import dataclasses

            json.dump(dataclasses.asdict(self.cfg), f, indent=2)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Restore actor and critic weights from *path*."""
        self.actor.load_state_dict(
            torch.load(os.path.join(path, "actor.pth"), map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(path, "critic.pth"), map_location=self.device)
        )
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        logger.info("Model loaded from %s", path)
