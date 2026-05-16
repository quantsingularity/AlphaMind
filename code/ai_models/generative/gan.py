"""
FinancialTimeSeriesGAN  -- Vanilla conditional GAN for return generation.
MarketGAN               -- AC-GAN with auxiliary regime-consistency loss.
"""

from __future__ import annotations

import tensorflow as tf
from ai_models.config import GANConfig
from ai_models.generative.discriminator import TimeSeriesDiscriminator
from ai_models.generative.generator import TransformerGenerator
from ai_models.generative.regime import RegimeClassifier, regime_consistency_loss


class FinancialTimeSeriesGAN(tf.keras.Model):
    """
    Vanilla GAN for synthetic financial return-sequence generation.

    Parameters
    ----------
    config : ``GANConfig`` instance.
    """

    def __init__(self, config: GANConfig = None, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = config or GANConfig()
        self.cfg = cfg
        self.generator = TransformerGenerator(cfg.seq_length, cfg.n_features)
        self.discriminator = TimeSeriesDiscriminator(cfg.seq_length)
        self.g_optimizer = tf.keras.optimizers.Adam(cfg.g_lr, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(cfg.d_lr, beta_1=0.5)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.g_loss_m = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_m = tf.keras.metrics.Mean(name="d_loss")

    @tf.function
    def train_step(self, real_sequences: tf.Tensor):
        batch = tf.shape(real_sequences)[0]
        noise = tf.random.normal((batch, self.cfg.latent_dim))

        with tf.GradientTape() as d_tape:
            fake = self.generator(noise, training=True)
            real_pred = self.discriminator(real_sequences, training=True)
            fake_pred = self.discriminator(fake, training=True)
            d_loss = self.loss_fn(tf.ones_like(real_pred), real_pred) + self.loss_fn(
                tf.zeros_like(fake_pred), fake_pred
            )
        self.d_optimizer.apply_gradients(
            zip(
                d_tape.gradient(d_loss, self.discriminator.trainable_variables),
                self.discriminator.trainable_variables,
            )
        )

        with tf.GradientTape() as g_tape:
            fake = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake, training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_pred), fake_pred)
        self.g_optimizer.apply_gradients(
            zip(
                g_tape.gradient(g_loss, self.generator.trainable_variables),
                self.generator.trainable_variables,
            )
        )

        self.g_loss_m.update_state(g_loss)
        self.d_loss_m.update_state(d_loss)
        return {"g_loss": g_loss, "d_loss": d_loss}

    def generate(self, n_samples: int) -> tf.Tensor:
        """Generate ``n_samples`` synthetic sequences."""
        noise = tf.random.normal((n_samples, self.cfg.latent_dim))
        return self.generator(noise, training=False)


class MarketGAN(tf.keras.Model):
    """
    Auxiliary-classifier GAN for synthetic market-data generation.

    Extends ``FinancialTimeSeriesGAN`` with a ``RegimeClassifier`` that
    encourages the generator to produce regime-coherent sequences.

    Parameters
    ----------
    config : ``GANConfig`` instance.
    """

    def __init__(self, config: GANConfig = None, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = config or GANConfig()
        self.cfg = cfg
        self.generator = TransformerGenerator(cfg.seq_length, cfg.n_features)
        self.discriminator = TimeSeriesDiscriminator(cfg.seq_length)
        self.aux_classifier = RegimeClassifier(n_regimes=cfg.n_regimes)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.g_optimizer = None
        self.d_optimizer = None

    def compile(self, g_optimizer, d_optimizer, **kwargs):
        super().compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, real_sequences: tf.Tensor) -> dict:
        batch = tf.shape(real_sequences)[0]
        noise = tf.random.normal((batch, self.cfg.latent_dim))

        with tf.GradientTape() as d_tape:
            fake = self.generator(noise, training=True)
            real_pred = self.discriminator(real_sequences, training=True)
            fake_pred = self.discriminator(fake, training=True)
            d_loss = self.loss_fn(tf.ones_like(real_pred), real_pred) + self.loss_fn(
                tf.zeros_like(fake_pred), fake_pred
            )
        self.d_optimizer.apply_gradients(
            zip(
                d_tape.gradient(d_loss, self.discriminator.trainable_variables),
                self.discriminator.trainable_variables,
            )
        )

        with tf.GradientTape() as g_tape:
            fake = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake, training=True)
            regime_probs = self.aux_classifier(fake, training=True)
            g_loss = self.loss_fn(
                tf.ones_like(fake_pred), fake_pred
            ) + 0.1 * regime_consistency_loss(regime_probs)
        g_vars = (
            self.generator.trainable_variables + self.aux_classifier.trainable_variables
        )
        self.g_optimizer.apply_gradients(zip(g_tape.gradient(g_loss, g_vars), g_vars))
        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate(self, n_samples: int) -> tf.Tensor:
        """Generate ``n_samples`` synthetic sequences."""
        noise = tf.random.normal((n_samples, self.cfg.latent_dim))
        return self.generator(noise, training=False)
