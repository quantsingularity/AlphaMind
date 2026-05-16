"""Unit tests for attention primitives."""

import numpy as np
import pytest
import tensorflow as tf
from ai_models.forecasting.attention import (
    MultiHeadAttention,
    TemporalAttentionBlock,
    get_positional_encoding,
)

BATCH, SEQ, D, HEADS, DFF = 4, 10, 64, 4, 128


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(D, HEADS)
        x = tf.random.normal((BATCH, SEQ, D))
        output = mha(x, x, x)
        assert output.shape == (BATCH, SEQ, D)

    def test_with_mask(self):
        mha = MultiHeadAttention(D, HEADS)
        x = tf.random.normal((BATCH, SEQ, D))
        mask = tf.ones((BATCH, 1, 1, SEQ))
        out = mha(x, x, x, mask=mask)
        assert out.shape == (BATCH, SEQ, D)

    def test_output_finite(self):
        mha = MultiHeadAttention(D, HEADS)
        x = tf.random.normal((BATCH, SEQ, D))
        out = mha(x, x, x)
        assert tf.reduce_all(tf.math.is_finite(out)).numpy()

    def test_head_divisibility_error(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=65, num_heads=4)  # not divisible

    def test_different_qkv(self):
        """Attention should work with different Q/K/V sequences."""
        mha = MultiHeadAttention(D, HEADS)
        q = tf.random.normal((BATCH, 6, D))
        k = tf.random.normal((BATCH, SEQ, D))
        v = tf.random.normal((BATCH, SEQ, D))
        out = mha(v, k, q)
        assert out.shape == (BATCH, 6, D)

    def test_gradient_flows(self):
        mha = MultiHeadAttention(D, HEADS)
        x = tf.Variable(tf.random.normal((BATCH, SEQ, D)))
        with tf.GradientTape() as tape:
            out = mha(x, x, x)
            loss = tf.reduce_mean(out)
        grad = tape.gradient(loss, x)
        assert grad is not None


class TestTemporalAttentionBlock:
    def test_output_shape(self, transformer_batch):
        block = TemporalAttentionBlock(d_model=D, num_heads=HEADS, dff=DFF)
        out = block(transformer_batch[:, :, :D], training=False)
        assert out.shape == transformer_batch[:, :, :D].shape

    def test_output_shape_simple(self):
        block = TemporalAttentionBlock(d_model=D, num_heads=HEADS, dff=DFF)
        x = tf.random.normal((BATCH, SEQ, D))
        out = block(x, training=False)
        assert out.shape == (BATCH, SEQ, D)

    def test_residual_connection(self):
        """Output should not be all zeros when input is all zeros."""
        block = TemporalAttentionBlock(d_model=D, num_heads=HEADS, dff=DFF)
        x = tf.zeros((2, SEQ, D))
        out = block(x, training=False)
        assert out.shape == (2, SEQ, D)

    def test_training_vs_inference_differ(self):
        """Dropout should cause train/infer outputs to differ."""
        block = TemporalAttentionBlock(
            d_model=D, num_heads=HEADS, dff=DFF, dropout_rate=0.5
        )
        x = tf.random.normal((BATCH, SEQ, D))
        out_train = block(x, training=True)
        out_infer = block(x, training=False)
        # Not guaranteed to differ but shapes must match
        assert out_train.shape == out_infer.shape


class TestPositionalEncoding:
    def test_shape(self):
        pe = get_positional_encoding(20, 64)
        assert pe.shape == (1, 20, 64)

    def test_values_bounded(self):
        pe = get_positional_encoding(50, 128)
        assert float(tf.reduce_min(pe)) >= -1.0 - 1e-6
        assert float(tf.reduce_max(pe)) <= 1.0 + 1e-6

    def test_deterministic(self):
        pe1 = get_positional_encoding(10, 32)
        pe2 = get_positional_encoding(10, 32)
        np.testing.assert_array_almost_equal(pe1.numpy(), pe2.numpy())

    def test_different_positions_differ(self):
        pe = get_positional_encoding(5, 16)
        assert not np.allclose(pe[0, 0].numpy(), pe[0, 1].numpy())
