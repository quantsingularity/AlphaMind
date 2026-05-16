"""
Attention primitives for financial time-series modelling.

MultiHeadAttention      -- Multi-head scaled dot-product attention (Keras layer)
TemporalAttentionBlock  -- Transformer encoder block (MHA + FFN + LayerNorm)
get_positional_encoding -- Sinusoidal positional encoding
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention mechanism for financial time series.

    Allows the model to jointly attend to information from different
    representation subspaces at different sequence positions.

    Parameters
    ----------
    d_model   : Total model dimensionality.
    num_heads : Number of parallel attention heads.
    """

    def __init__(self, d_model: int, num_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split last dim into (num_heads, depth) and transpose."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        v: tf.Tensor,
        k: tf.Tensor,
        q: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(
            tf.cast(tf.shape(k)[-1], tf.float32)
        )
        if mask is not None:
            logits += (1.0 - mask) * -1e9
        weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)


class TemporalAttentionBlock(tf.keras.layers.Layer):
    """
    Transformer encoder block for temporal financial data.

    Combines multi-head self-attention with a position-wise feed-forward
    sublayer, layer normalisation, and dropout.

    Parameters
    ----------
    d_model      : Model dimensionality.
    num_heads    : Number of attention heads.
    dff          : Feed-forward inner dimensionality.
    dropout_rate : Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        attn = self.mha(x, x, x, mask)
        attn = self.drop1(attn, training=training)
        out1 = self.norm1(x + attn)
        ffn = self.ffn(out1)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(out1 + ffn)


def get_positional_encoding(seq_length: int, d_model: int) -> tf.Tensor:
    """
    Compute sinusoidal positional encodings.

    Parameters
    ----------
    seq_length : Sequence length.
    d_model    : Embedding dimensionality.

    Returns
    -------
    tf.Tensor
        Shape ``(1, seq_length, d_model)``.
    """
    positions = np.arange(seq_length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10_000.0, (2 * (dims // 2)) / np.float32(d_model))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)
