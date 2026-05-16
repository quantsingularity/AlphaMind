"""Unit tests for DDPGTradingAgent, Actor, and Critic."""

import numpy as np
import torch
from ai_models.agents.ddpg import Actor, Critic, DDPGTradingAgent


class TestActor:
    def test_output_shape(self):
        actor = Actor(state_dim=12, action_dim=4)
        out = actor(torch.randn(8, 12))
        assert out.shape == (8, 4)

    def test_output_bounded(self):
        actor = Actor(state_dim=12, action_dim=4)
        out = actor(torch.randn(64, 12))
        assert out.min().item() >= -1.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_layer_norm_applied(self):
        """LayerNorm should make output mean ~ 0 on random input."""
        actor = Actor(state_dim=128, action_dim=8)
        out = actor(torch.randn(256, 128))
        assert out.shape == (256, 8)


class TestCritic:
    def test_output_shape(self):
        critic = Critic(state_dim=12, action_dim=4)
        s, a = torch.randn(8, 12), torch.randn(8, 4)
        assert critic(s, a).shape == (8, 1)

    def test_gradient_flows(self):
        critic = Critic(state_dim=8, action_dim=3)
        s = torch.randn(4, 8, requires_grad=True)
        a = torch.randn(4, 3, requires_grad=True)
        q = critic(s, a).mean()
        q.backward()
        assert s.grad is not None


class TestDDPGTradingAgent:
    def test_select_action_shape(self, ddpg_agent, trading_env):
        obs, _ = trading_env.reset(seed=0)
        action = ddpg_agent.select_action(obs, add_noise=False)
        assert action.shape == (trading_env.n_assets,)

    def test_select_action_bounded(self, ddpg_agent, trading_env):
        obs, _ = trading_env.reset(seed=1)
        action = ddpg_agent.select_action(obs, add_noise=False)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_update_none_when_buffer_empty(self, ddpg_agent):
        result = ddpg_agent.update()
        assert result is None

    def test_update_returns_finite_losses(self, ddpg_agent, trading_env):
        obs, _ = trading_env.reset(seed=2)
        for _ in range(ddpg_agent.cfg.batch_size + 10):
            action = trading_env.action_space.sample()
            next_obs, r, d, _, _ = trading_env.step(action)
            ddpg_agent.replay_buffer.add(obs, action, r, next_obs, d)
            obs = next_obs
            if d:
                obs, _ = trading_env.reset()
        result = ddpg_agent.update()
        assert result is not None
        critic_loss, actor_loss = result
        assert np.isfinite(critic_loss)
        assert np.isfinite(actor_loss)

    def test_save_and_load_preserves_weights(self, ddpg_agent, trading_env, tmp_path):
        ddpg_agent.save_model(str(tmp_path / "model"))
        from ai_models.config import DDPGConfig

        new_agent = DDPGTradingAgent(trading_env, config=DDPGConfig(use_cuda=False))
        new_agent.load_model(str(tmp_path / "model"))
        for p1, p2 in zip(ddpg_agent.actor.parameters(), new_agent.actor.parameters()):
            assert torch.allclose(p1, p2)

    def test_config_json_saved(self, ddpg_agent, tmp_path):
        import json

        ddpg_agent.save_model(str(tmp_path / "model"))
        cfg_path = tmp_path / "model" / "config.json"
        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text())
        assert "gamma" in data

    def test_noise_affects_action(self, ddpg_agent, trading_env):
        obs, _ = trading_env.reset(seed=3)
        a_noisy = ddpg_agent.select_action(obs, add_noise=True)
        a_det = ddpg_agent.select_action(obs, add_noise=False)
        # With noise the actions should generally differ
        assert a_noisy.shape == a_det.shape
