"""RegimeClassifier and regime_consistency_loss."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense


class RegimeClassifier(tf.keras.layers.Layer):
    """
    LSTM-based market-regime classifier.

    Maps a return sequence to a probability distribution over
    ``n_regimes`` market regimes (e.g., bull / neutral / bear).

    Parameters
    ----------
    n_regimes   : Number of regime classes.
    lstm_units  : LSTM hidden-state size.
    dropout     : LSTM dropout probability.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        lstm_units: int = 64,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lstm = LSTM(lstm_units, return_sequences=False, dropout=dropout)
        self.dense = Dense(n_regimes, activation="softmax")

    def call(
        self, inputs: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:  # noqa: D102
        return self.dense(self.lstm(inputs, training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_regimes": self.dense.units})
        return cfg


def regime_consistency_loss(regime_probs: tf.Tensor) -> tf.Tensor:
    """
    Auxiliary loss that penalises batch-level regime incoherence.

    A lower value means the generated batch exhibits a consistent regime
    distribution across samples.

    Parameters
    ----------
    regime_probs : Softmax probabilities ``(batch, n_regimes)``.

    Returns
    -------
    tf.Tensor
        Scalar mean squared deviation from the batch-mean distribution.
    """
    mean = tf.reduce_mean(regime_probs, axis=0, keepdims=True)
    delta = regime_probs - mean
    return tf.reduce_mean(tf.reduce_sum(tf.square(delta), axis=1))
