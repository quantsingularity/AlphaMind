"""
Shared pytest fixtures for the AlphaMind ai_models test suite.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def n_assets() -> int:
    return 4


@pytest.fixture(scope="session")
def window() -> int:
    return 5


@pytest.fixture(scope="session")
def n_macro() -> int:
    return 3


@pytest.fixture
def trading_env(n_assets, window, n_macro):
    """Fresh TradingEnvironment for each test."""
    from ai_models.config import TradingEnvConfig
    from ai_models.environments import TradingEnvironment

    cfg = TradingEnvConfig(
        n_assets=n_assets, window=window, n_macro=n_macro, max_steps=50
    )
    return TradingEnvironment(config=cfg)


@pytest.fixture
def portfolio_env(n_assets, window, n_macro):
    """Fresh PortfolioGymEnv for each test."""
    from ai_models.config import PortfolioEnvConfig
    from ai_models.environments import PortfolioGymEnv

    cfg = PortfolioEnvConfig(window=window, n_macro=n_macro, max_steps=50)
    return PortfolioGymEnv(
        universe=[f"ASSET_{i}" for i in range(n_assets)],
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ddpg_agent(trading_env):
    """DDPGTradingAgent with minimal config."""
    from ai_models.agents import DDPGTradingAgent
    from ai_models.config import DDPGConfig

    cfg = DDPGConfig(batch_size=16, buffer_size=200, warmup_steps=10, use_cuda=False)
    return DDPGTradingAgent(trading_env, config=cfg)


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_market_data() -> Dict[str, Any]:
    rng = np.random.default_rng(42)
    return {
        "prices": rng.standard_normal((100, 4)),
        "timestamps": list(range(100)),
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    }


@pytest.fixture(scope="session")
def transformer_batch():
    """Small (batch, seq, feat) tensor for attention/transformer tests."""
    import tensorflow as tf

    return tf.random.normal((4, 10, 128), seed=0)


@pytest.fixture(scope="session")
def gan_batch():
    """Small real-sequence batch for GAN tests."""
    import tensorflow as tf

    return tf.random.normal((8, 20, 3), seed=1)
