"""
AlphaMind AI Models
===================
Production-grade machine-learning and reinforcement-learning components
for quantitative trading and portfolio management.

Sub-packages
------------
agents        -- RL trading agents (DDPG, PPO)
environments  -- Gymnasium trading environments
forecasting   -- Transformer-based time-series forecasting
generative    -- GAN-based synthetic market-data generation
examples      -- Runnable end-to-end example scripts
research      -- Research notebooks (Jupyter)
tests         -- Comprehensive test suite
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = [
    "agents",
    "environments",
    "forecasting",
    "generative",
]


def _optional(module: str):
    """Return the module if importable, else None. Never raises."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError:
        return None
