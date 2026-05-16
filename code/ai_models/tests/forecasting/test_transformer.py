"""Unit tests for FinancialTimeSeriesTransformer."""

import tensorflow as tf
from ai_models.config import TransformerConfig
from ai_models.forecasting.transformer import FinancialTimeSeriesTransformer

BATCH, SEQ, FEAT = 4, 10, 16


def make_model(out_seq=5) -> FinancialTimeSeriesTransformer:
    cfg = TransformerConfig(
        num_layers=2,
        d_model=32,
        num_heads=4,
        dff=64,
        input_seq_length=SEQ,
        output_seq_length=out_seq,
        dropout_rate=0.1,
    )
    return FinancialTimeSeriesTransformer(config=cfg)


class TestFinancialTimeSeriesTransformer:
    def test_output_shape(self):
        model = make_model(out_seq=5)
        x = tf.random.normal((BATCH, SEQ, 32))
        out = model(x, training=False)
        assert out.shape == (BATCH, SEQ, 5)

    def test_output_finite(self):
        model = make_model()
        x = tf.random.normal((BATCH, SEQ, 32))
        out = model(x, training=False)
        assert tf.reduce_all(tf.math.is_finite(out)).numpy()

    def test_training_mode_runs(self):
        model = make_model()
        x = tf.random.normal((BATCH, SEQ, 32))
        out = model(x, training=True)
        assert out.shape[0] == BATCH

    def test_gradient_update(self):
        model = make_model()
        tf.keras.optimizers.Adam(1e-3)
        x = tf.random.normal((BATCH, SEQ, 32))
        y = tf.random.normal((BATCH, SEQ, 5))
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean((pred - y) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in grads)

    def test_different_seq_lengths(self):
        model = make_model()
        for seq in [5, 10, 20]:
            x = tf.random.normal((2, seq, 32))
            out = model(x, training=False)
            assert out.shape[1] == seq
