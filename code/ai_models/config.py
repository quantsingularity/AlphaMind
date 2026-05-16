"""
Central configuration dataclasses for all ai_models sub-packages.
Use these instead of raw dicts to get IDE auto-complete and type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DDPGConfig:
    """Hyperparameters for the DDPG trading agent."""

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup_steps: int = 1_000
    noise_sigma: float = 0.20
    noise_theta: float = 0.15
    noise_decay: float = 0.9995
    actor_hidden: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256])
    use_cuda: bool = True
    grad_clip: float = 1.0


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO portfolio agent."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50


@dataclass
class TransformerConfig:
    """Hyperparameters for the Transformer forecaster."""

    num_layers: int = 4
    d_model: int = 128
    num_heads: int = 8
    dff: int = 512
    input_seq_length: int = 60
    output_seq_length: int = 20
    dropout_rate: float = 0.10
    learning_rate: float = 1e-4


@dataclass
class GANConfig:
    """Hyperparameters for the financial GAN."""

    seq_length: int = 50
    n_features: int = 3
    latent_dim: int = 100
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    n_regimes: int = 3
    batch_size: int = 32


@dataclass
class TradingEnvConfig:
    """Configuration for the DDPG TradingEnvironment."""

    n_assets: int = 5
    window: int = 10
    n_macro: int = 5
    transaction_cost: float = 0.001
    max_steps: int = 252


@dataclass
class PortfolioEnvConfig:
    """Configuration for the PPO PortfolioGymEnv."""

    transaction_cost: float = 0.001
    window: int = 10
    n_macro: int = 5
    max_steps: int = 252
