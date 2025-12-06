"""
## Example script demonstrating hyperparameter tuning for the DDPG trading agent.
## This script shows how to use the HyperparameterTuner class to find optimal
## hyperparameters for the DDPG agent in various market conditions.
"""

import os
from core.logging import get_logger

logger = get_logger(__name__)

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from AlphaMind.backend.ai_models.ddpg_trading import (
    BacktestEngine,
    DDPGAgent,
    HyperparameterTuner,
    TradingGymEnv,
)


def generate_sample_market_data(n_assets=3, n_days=252, seed=42):
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

    # Generate price series with different characteristics
    prices = {}

    # Asset 1: Trending upward
    prices["Asset1"] = np.cumprod(1 + np.random.normal(0.001, 0.01, n_days))

    # Asset 2: Mean-reverting
    noise = np.random.normal(0, 0.02, n_days)
    prices["Asset2"] = 100 + np.cumsum(noise - 0.3 * np.sign(np.cumsum(noise)))

    # Asset 3: Cyclical
    t = np.linspace(0, 4 * np.pi, n_days)
    prices["Asset3"] = (
        100 + 10 * np.sin(t) + np.cumsum(np.random.normal(0, 0.01, n_days))
    )

    # Additional assets if needed
    for i in range(4, n_assets + 1):
        # Random walk with drift
        drift = np.random.uniform(-0.0005, 0.0015)
        vol = np.random.uniform(0.01, 0.02)
        prices[f"Asset{i}"] = np.cumprod(1 + np.random.normal(drift, vol, n_days))

    # Convert to DataFrame
    df = pd.DataFrame(prices)

    return df


def create_environment(
    data=None, window_size=10, transaction_cost=0.001, max_steps=252
):
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
    param_grid=None, n_trials=20, episodes_per_trial=5, max_steps=100
):
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
    # Generate sample data
    data = generate_sample_market_data(n_assets=4, n_days=504)  # 2 years of data

    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            "actor_lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            "critic_lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "gamma": [0.95, 0.97, 0.99],
            "tau": [0.001, 0.005, 0.01, 0.05],
            "noise_sigma": [0.1, 0.15, 0.2, 0.3],
            "batch_size": [32, 64, 128, 256],
            "buffer_capacity": [10000, 50000, 100000],
            "actor_hidden_dims": [(128, 64), (256, 128), (512, 256)],
            "critic_hidden_dims": [(128, 64), (256, 128), (512, 256)],
        }

    # Create tuner
    tuner = HyperparameterTuner(
        env_creator=lambda: create_environment(
            data=data, window_size=10, transaction_cost=0.001, max_steps=max_steps
        ),
        param_grid=param_grid,
        n_trials=n_trials,
        episodes_per_trial=episodes_per_trial,
        max_steps=max_steps,
    )

    # Run tuning
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
    best_config, tuning_results = tuner.run()

    logger.info(f"\nTuning completed!")
    logger.info(f"Best configuration: {best_config}")

    # Validate best configuration with longer run
    logger.info("\nValidating best configuration with extended run...")
    env = create_environment(data=data, max_steps=252)
    agent = DDPGAgent(env, config=best_config)

    # Train with best config
    agent.train(max_episodes=20, max_steps=252)

    # Backtest
    backtest = BacktestEngine(env, agent)
    results = backtest.run(episodes=1)

    logger.info(f"Validation complete!")
    logger.info(f"Final metrics with best configuration:")
    logger.info(f" 	Total Return: {results['metrics']['total_return']:.2%}")
    logger.info(f" 	Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f" 	Max Drawdown: {results['metrics']['max_drawdown']:.2%}")

    return best_config, tuning_results


def analyze_tuning_results(results):
    """
    Analyze hyperparameter tuning results

    Args:
        results: List of tuning results

    Returns:
        DataFrame with analysis
    """
    # Extract data
    data = []

    for result in results:
        config = result["config"]
        eval_reward = result["eval_reward"]

        row = {"trial": result["trial"], "eval_reward": eval_reward, **config}

        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Plot parameter importance
    plt.figure(figsize=(15, 10))

    # Get parameters (excluding trial and eval_reward)
    params = [col for col in df.columns if col not in ["trial", "eval_reward"]]

    # Plot each parameter's relationship with reward
    for i, param in enumerate(params):
        if isinstance(df[param].iloc[0], tuple):
            # Handle tuple parameters (like hidden_dims)
            # Convert to string for plotting
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
    # Run hyperparameter tuning with default settings
    best_config, tuning_results = run_hyperparameter_tuning(
        n_trials=10,  # Reduced for example, use 20+ for real tuning
        episodes_per_trial=3,
        max_steps=100,
    )

    # Analyze results
    analysis_df = analyze_tuning_results(tuning_results)

    # Save best configuration
    import json

    with open("best_ddpg_config.json", "w") as f:
        # Convert tuples to lists for JSON serialization
        serializable_config = {}
        for k, v in best_config.items():
            if isinstance(v, tuple):
                serializable_config[k] = list(v)
            else:
                serializable_config[k] = v

        json.dump(serializable_config, f, indent=4)

    print(
        "\nAnalysis complete! Results saved to 'parameter_importance.png' and 'best_ddpg_config.json'"
    )
