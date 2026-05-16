"""Unit tests for FinancialTimeSeriesGAN and MarketGAN."""

import numpy as np
import tensorflow as tf
from ai_models.config import GANConfig
from ai_models.generative import FinancialTimeSeriesGAN, MarketGAN


def small_cfg(**kw) -> GANConfig:
    defaults = dict(
        seq_length=10,
        n_features=3,
        latent_dim=16,
        g_lr=2e-4,
        d_lr=2e-4,
        n_regimes=3,
        batch_size=4,
    )
    defaults.update(kw)
    return GANConfig(**defaults)


class TestFinancialTimeSeriesGAN:
    def test_generate_shape(self):
        gan = FinancialTimeSeriesGAN(config=small_cfg())
        out = gan.generate(n_samples=8)
        assert out.shape == (8, 10, 3)

    def test_generate_finite(self):
        gan = FinancialTimeSeriesGAN(config=small_cfg())
        out = gan.generate(n_samples=4)
        assert tf.reduce_all(tf.math.is_finite(out)).numpy()

    def test_train_step_returns_losses(self, gan_batch):
        gan = FinancialTimeSeriesGAN(config=small_cfg())
        real = tf.cast(gan_batch[:, :10, :], tf.float32)
        res = gan.train_step(real)
        assert "g_loss" in res
        assert "d_loss" in res

    def test_train_step_losses_finite(self, gan_batch):
        gan = FinancialTimeSeriesGAN(config=small_cfg())
        real = tf.cast(gan_batch[:, :10, :], tf.float32)
        res = gan.train_step(real)
        assert np.isfinite(float(res["g_loss"]))
        assert np.isfinite(float(res["d_loss"]))

    def test_discriminator_output_range(self):
        gan = FinancialTimeSeriesGAN(config=small_cfg())
        seqs = tf.random.normal((6, 10, 3))
        out = gan.discriminator(seqs, training=False)
        assert float(tf.reduce_min(out)) >= 0.0 - 1e-6
        assert float(tf.reduce_max(out)) <= 1.0 + 1e-6


class TestMarketGAN:
    def _make(self) -> MarketGAN:
        cfg = small_cfg()
        gan = MarketGAN(config=cfg)
        gan.compile(
            g_optimizer=tf.keras.optimizers.Adam(cfg.g_lr),
            d_optimizer=tf.keras.optimizers.Adam(cfg.d_lr),
        )
        return gan

    def test_generate_shape(self):
        gan = self._make()
        out = gan.generate(n_samples=5)
        assert out.shape == (5, 10, 3)

    def test_train_step_dict_keys(self, gan_batch):
        gan = self._make()
        real = tf.cast(gan_batch[:, :10, :], tf.float32)
        res = gan.train_step(real)
        assert "g_loss" in res
        assert "d_loss" in res

    def test_aux_classifier_output_sums_to_one(self):
        from ai_models.generative import RegimeClassifier

        clf = RegimeClassifier(n_regimes=3)
        seqs = tf.random.normal((4, 10, 3))
        probs = clf(seqs)
        row_sums = tf.reduce_sum(probs, axis=1).numpy()
        np.testing.assert_array_almost_equal(row_sums, np.ones(4), decimal=5)
