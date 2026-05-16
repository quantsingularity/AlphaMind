"""TimeSeriesDiscriminator: Conv1D real/fake classifier."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU


class TimeSeriesDiscriminator(tf.keras.layers.Layer):
    """
    Discriminator network for financial time series.

    Uses strided 1-D convolutions to classify sequences as real or
    generated.

    Parameters
    ----------
    seq_length : Expected sequence length (informational only).
    """

    def __init__(self, seq_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.conv1 = Conv1D(64, kernel_size=3, strides=2, padding="same")
        self.conv2 = Conv1D(128, kernel_size=3, strides=2, padding="same")
        self.flatten = Flatten()
        self.dense = Dense(1, activation="sigmoid")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # noqa: D102
        x = LeakyReLU(0.2)(self.conv1(inputs))
        x = LeakyReLU(0.2)(self.conv2(x))
        return self.dense(self.flatten(x))

    def get_config(self):
        return {"seq_length": self.seq_length, **super().get_config()}
