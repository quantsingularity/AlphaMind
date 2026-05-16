"""TransformerGenerator: maps latent noise to synthetic return sequences."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape


class TransformerGenerator(tf.keras.layers.Layer):
    """
    Generator network for financial time-series synthesis.

    Maps a latent noise vector ``z`` of shape ``(batch, latent_dim)``
    to a synthetic sequence of shape ``(batch, seq_length, n_features)``.

    Parameters
    ----------
    seq_length : Number of time steps in the generated sequence.
    n_features : Number of features per time step.
    """

    def __init__(self, seq_length: int, n_features: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.n_features = n_features
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(seq_length * n_features)
        self.reshape = Reshape((seq_length, n_features))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # noqa: D102
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.reshape(x)

    def get_config(self):
        return {
            "seq_length": self.seq_length,
            "n_features": self.n_features,
            **super().get_config(),
        }
