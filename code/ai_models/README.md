# AlphaMind AI Models

Production-grade ML and RL components for quantitative trading.

## Package Structure

```
ai_models/
├── agents/        RL trading agents (DDPG, PPO)
├── environments/  Gymnasium-compatible trading environments
├── forecasting/   Transformer time-series forecasting
├── generative/    GAN synthetic market-data generation
├── examples/      Runnable end-to-end scripts
├── research/      Jupyter research notebooks
└── tests/         Full pytest test suite
```

## Quick Start

```python
from ai_models.environments import TradingEnvironment, PortfolioGymEnv
from ai_models.agents import DDPGTradingAgent, PPOAgent
from ai_models.forecasting import FinancialTimeSeriesTransformer
from ai_models.generative import FinancialTimeSeriesGAN
```

## Running Tests

```bash
pytest ai_models/tests/ -v --tb=short
```
