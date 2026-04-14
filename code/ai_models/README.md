# AI Models in AlphaMind

This directory contains various AI models used in the AlphaMind trading platform.

## Models Overview

- **reinforcement_learning.py**: Contains the PPO-based reinforcement learning agent for portfolio optimization
- **ddpg_trading.py**: Implements a Deep Deterministic Policy Gradient (DDPG) agent for trading with PyTorch
- **attention_mechanism.py**: Implements attention mechanisms for time series analysis
- **generative_finance.py**: Contains generative models for financial data
- **transformer_timeseries**: Directory containing transformer-based models for time series forecasting

## DDPG Trading Module

The DDPG Trading module (`ddpg_trading.py`) implements a Deep Deterministic Policy Gradient agent for trading using PyTorch. This module provides:

- A complete DDPG implementation with Actor-Critic architecture
- Custom trading environment compatible with OpenAI Gym
- Experience replay buffer for stable learning
- Ornstein-Uhlenbeck process for exploration
- Comprehensive backtesting capabilities
- Hyperparameter tuning support
- Detailed logging and visualization

### Usage Example

```python
from backend.ai_models.ddpg_trading import DDPGAgent, TradingGymEnv, BacktestEngine

# Create environment
env = TradingGymEnv(
    data_stream=your_market_data,  # Your market data stream
    window_size=10,
    transaction_cost=0.001,
    reward_scaling=1.0,
    max_steps=252  # One trading year
)

# Create agent
agent = DDPGAgent(env)

# Train agent
agent.train(max_episodes=100, max_steps=252)

# Backtest
backtest = BacktestEngine(env, agent)
results = backtest.run(episodes=1, render=True)
```

### Hyperparameter Tuning

```python
from backend.ai_models.ddpg_trading import HyperparameterTuner, TradingGymEnv

# Define parameter grid
param_grid = {
    "actor_lr": [1e-4, 3e-4, 1e-3],
    "critic_lr": [1e-3, 3e-3, 1e-2],
    "gamma": [0.95, 0.97, 0.99],
    "tau": [0.001, 0.005, 0.01],
    "noise_sigma": [0.1, 0.2, 0.3]
}

# Create tuner
tuner = HyperparameterTuner(
    env_creator=lambda: TradingGymEnv(
        data_stream=your_market_data,
        window_size=10,
        transaction_cost=0.001,
        max_steps=100
    ),
    param_grid=param_grid,
    n_trials=20,
    episodes_per_trial=5,
    max_steps=100
)

# Run tuning
best_config, tuning_results = tuner.run()
```

## Integration with AlphaMind

The DDPG trading module is designed to integrate seamlessly with the AlphaMind platform. It can be used for:

1. Automated trading strategy development
2. Portfolio optimization
3. Market simulation and backtesting
4. Risk management

The module supports both historical backtesting and live trading with real-time market data streams.
