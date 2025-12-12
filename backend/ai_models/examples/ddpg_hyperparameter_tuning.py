"""
## Example script demonstrating hyperparameter tuning for the DDPG trading agent.
## This script shows how to use the HyperparameterTuner class to find optimal
## hyperparameters for the DDPG agent in various market conditions.
"""

from typing import Any
import os
from core.logging import get_logger

logger = get_logger(__name__)
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from AlphaMind.backend.ai_models.ddpg_trading import (
    BacktestEngine,
    DDPGAgent,
    HyperparameterTuner,
    TradingGymEnv,
)


def generate_sample_market_data(
    n_assets: Any = 3, n_days: Any = 252, seed: Any = 42
) -> Any:
    """
    Generate sample market data for testing

    Args:
        n_assets: Number of assets
        n_days: Number of trading days
        seed: Random seed for reproducibility

    Returns:
        DataFrame with asset prices
    """
    np.random.seed(seed)
    prices = {}
    prices["Asset1"] = np.cumprod(1 + np.random.normal(0.001, 0.01, n_days))
    noise = np.random.normal(0, 0.02, n_days)
    prices["Asset2"] = 100 + np.cumsum(noise - 0.3 * np.sign(np.cumsum(noise)))
    t = np.linspace(0, 4 * np.pi, n_days)
    prices["Asset3"] = (
        100 + 10 * np.sin(t) + np.cumsum(np.random.normal(0, 0.01, n_days))
    )
    for i in range(4, n_assets + 1):
        drift = np.random.uniform(-0.0005, 0.0015)
        vol = np.random.uniform(0.01, 0.02)
        prices[f"Asset{i}"] = np.cumprod(1 + np.random.normal(drift, vol, n_days))
    df = pd.DataFrame(prices)
    return df


def create_environment(
    data: Any = None,
    window_size: Any = 10,
    transaction_cost: Any = 0.001,
    max_steps: Any = 252,
) -> Any:
    """
    Create trading environment with optional data

    Args:
        data: Market data (DataFrame)
        window_size: Observation window size
        transaction_cost: Trading cost
        max_steps: Maximum steps per episode

    Returns:
        TradingGymEnv instance
    """
    if data is None:
        data = generate_sample_market_data()
    env = TradingGymEnv(
        data_stream=data,
        window_size=window_size,
        transaction_cost=transaction_cost,
        reward_scaling=1.0,
        max_steps=max_steps,
    )
    return env


def run_hyperparameter_tuning(
    param_grid: Any = None,
    n_trials: Any = 20,
    episodes_per_trial: Any = 5,
    max_steps: Any = 100,
) -> Any:
    """
    Run hyperparameter tuning with specified parameter grid

    Args:
        param_grid: Dictionary of parameters to tune
        n_trials: Number of random trials
        episodes_per_trial: Episodes per trial
        max_steps: Maximum steps per episode

    Returns:
        Best configuration and all results
    """
    data = generate_sample_market_data(n_assets=4, n_days=504)
    if param_grid is None:
        param_grid = {
            "actor_lr": [1e-05, 3e-05, 0.0001, 0.0003, 0.001],
            "critic_lr": [0.0001, 0.0003, 0.001, 0.003, 0.01],
            "gamma": [0.95, 0.97, 0.99],
            "tau": [0.001, 0.005, 0.01, 0.05],
            "noise_sigma": [0.1, 0.15, 0.2, 0.3],
            "batch_size": [32, 64, 128, 256],
            "buffer_capacity": [10000, 50000, 100000],
            "actor_hidden_dims": [(128, 64), (256, 128), (512, 256)],
            "critic_hidden_dims": [(128, 64), (256, 128), (512, 256)],
        }
    tuner = HyperparameterTuner(
        env_creator=lambda: create_environment(
            data=data, window_size=10, transaction_cost=0.001, max_steps=max_steps
        ),
        param_grid=param_grid,
        n_trials=n_trials,
        episodes_per_trial=episodes_per_trial,
        max_steps=max_steps,
    )
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
    best_config, tuning_results = tuner.run()
    logger.info(f"\nTuning completed!")
    logger.info(f"Best configuration: {best_config}")
    logger.info("\nValidating best configuration with extended run...")
    env = create_environment(data=data, max_steps=252)
    agent = DDPGAgent(env, config=best_config)
    agent.train(max_episodes=20, max_steps=252)
    backtest = BacktestEngine(env, agent)
    results = backtest.run(episodes=1)
    logger.info(f"Validation complete!")
    logger.info(f"Final metrics with best configuration:")
    logger.info(f" \tTotal Return: {results['metrics']['total_return']:.2%}")
    logger.info(f" \tSharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f" \tMax Drawdown: {results['metrics']['max_drawdown']:.2%}")
    return (best_config, tuning_results)


def analyze_tuning_results(results: Any) -> Any:
    """
    Analyze hyperparameter tuning results

    Args:
        results: List of tuning results

    Returns:
        DataFrame with analysis
    """
    data = []
    for result in results:
        config = result["config"]
        eval_reward = result["eval_reward"]
        row = {"trial": result["trial"], "eval_reward": eval_reward, **config}
        data.append(row)
    df = pd.DataFrame(data)
    plt.figure(figsize=(15, 10))
    params = [col for col in df.columns if col not in ["trial", "eval_reward"]]
    for i, param in enumerate(params):
        if isinstance(df[param].iloc[0], tuple):
            df[f"{param}_str"] = df[param].astype(str)
            param = f"{param}_str"
        plt.subplot(3, 3, i + 1)
        plt.scatter(df[param], df["eval_reward"])
        plt.title(f"{param} vs Reward")
        plt.xlabel(param)
        plt.ylabel("Reward")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("parameter_importance.png")
    return df


if __name__ == "__main__":
    best_config, tuning_results = run_hyperparameter_tuning(
        n_trials=10, episodes_per_trial=3, max_steps=100
    )
    analysis_df = analyze_tuning_results(tuning_results)
    import json

    with open("best_ddpg_config.json", "w") as f:
        serializable_config = {}
        for k, v in best_config.items():
            if isinstance(v, tuple):
                serializable_config[k] = list(v)
            else:
                serializable_config[k] = v
        json.dump(serializable_config, f, indent=4)
    logger.info(
        "\nAnalysis complete! Results saved to 'parameter_importance.png' and 'best_ddpg_config.json'"
    )
