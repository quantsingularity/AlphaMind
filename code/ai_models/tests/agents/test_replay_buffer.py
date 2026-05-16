"""Unit tests for ReplayBuffer and OUNoise."""

import numpy as np
import torch
from ai_models.agents.replay_buffer import OUNoise, ReplayBuffer


class TestReplayBuffer:
    def test_add_increments_len(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
        assert len(buf) == 1

    def test_capacity_is_enforced(self):
        buf = ReplayBuffer(capacity=5)
        for _ in range(12):
            buf.add(np.zeros(3), np.zeros(2), 0.0, np.zeros(3), False)
        assert len(buf) == 5

    def test_sample_returns_five_tensors(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(30):
            buf.add(
                np.random.randn(5),
                np.random.randn(2),
                float(np.random.randn()),
                np.random.randn(5),
                False,
            )
        s, a, r, ns, d = buf.sample(16)
        assert isinstance(s, torch.Tensor)
        assert s.shape == (16, 5)
        assert a.shape == (16, 2)
        assert r.shape == (16, 1)
        assert ns.shape == (16, 5)
        assert d.shape == (16, 1)

    def test_sample_clamps_to_available(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(4):
            buf.add(np.zeros(3), np.zeros(2), 0.0, np.zeros(3), False)
        s, _, _, _, _ = buf.sample(32)
        assert s.shape[0] == 4

    def test_rewards_stored_as_float(self):
        buf = ReplayBuffer(capacity=50)
        buf.add(np.zeros(2), np.zeros(1), 3.14, np.zeros(2), True)
        _, _, r, _, d = buf.sample(1)
        assert abs(float(r[0, 0]) - 3.14) < 1e-5
        assert float(d[0, 0]) == 1.0

    def test_dict_state_flattened_correctly(self):
        buf = ReplayBuffer(capacity=50)
        state = {"prices": np.ones((3, 5)), "volumes": np.ones(3), "macro": np.ones(2)}
        buf.add(state, np.zeros(3), 0.0, state, False)
        s, _, _, _, _ = buf.sample(1)
        # 3*5 + 3 + 2 = 20
        assert s.shape == (1, 20)

    def test_repr_contains_size(self):
        buf = ReplayBuffer(capacity=10)
        assert "10" in repr(buf)


class TestOUNoise:
    def test_output_shape(self):
        noise = OUNoise(size=6)
        assert noise.sample().shape == (6,)

    def test_reset_returns_to_mu(self):
        noise = OUNoise(size=4, mu=0.0)
        for _ in range(50):
            noise.sample()
        noise.reset()
        np.testing.assert_array_almost_equal(noise.state, noise.mu)

    def test_consecutive_samples_differ(self):
        noise = OUNoise(size=3)
        s1, s2 = noise.sample().copy(), noise.sample().copy()
        assert not np.array_equal(s1, s2)

    def test_sigma_scales_variance(self):
        """Higher sigma should produce higher variance samples."""
        noise_lo = OUNoise(size=100, sigma=0.01)
        noise_hi = OUNoise(size=100, sigma=2.00)
        samples_lo = np.array([noise_lo.sample() for _ in range(200)])
        samples_hi = np.array([noise_hi.sample() for _ in range(200)])
        assert samples_hi.std() > samples_lo.std()

    def test_repr(self):
        noise = OUNoise(size=5)
        assert "OUNoise" in repr(noise)
