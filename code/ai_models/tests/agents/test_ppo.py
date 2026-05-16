"""Unit tests for PPOAgent (requires stable-baselines3)."""

import pytest


def _sb3_available() -> bool:
    try:
        import stable_baselines3  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _sb3_available(), reason="stable-baselines3 not installed"
)


class TestPPOAgent:
    def test_select_action_shape(self, portfolio_env):
        from ai_models.agents import PPOAgent

        agent = PPOAgent(portfolio_env)
        obs, _ = portfolio_env.reset(seed=0)
        action = agent.select_action(obs, add_noise=False)
        assert action.shape == portfolio_env.action_space.shape

    def test_save_and_load(self, portfolio_env, tmp_path):
        from ai_models.agents import PPOAgent

        agent = PPOAgent(portfolio_env)
        agent.save_model(str(tmp_path / "ppo_model"))
        loaded = PPOAgent.from_pretrained(str(tmp_path / "ppo_model"), portfolio_env)
        obs, _ = portfolio_env.reset(seed=1)
        a1 = agent.select_action(obs)
        a2 = loaded.select_action(obs)
        import numpy as np

        np.testing.assert_array_almost_equal(a1, a2)

    def test_evaluate_returns_metrics(self, portfolio_env):
        from ai_models.agents import PPOAgent

        agent = PPOAgent(portfolio_env)
        metrics = agent.evaluate(num_episodes=2)
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        import math

        assert math.isfinite(metrics["mean_reward"])
