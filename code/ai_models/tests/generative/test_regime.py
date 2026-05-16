"""Unit tests for RegimeClassifier and regime_consistency_loss."""

import numpy as np
import tensorflow as tf
from ai_models.generative import RegimeClassifier, regime_consistency_loss


class TestRegimeClassifier:
    def test_output_shape(self):
        clf = RegimeClassifier(n_regimes=3)
        x = tf.random.normal((8, 20, 4))
        out = clf(x)
        assert out.shape == (8, 3)

    def test_probabilities_sum_to_one(self):
        clf = RegimeClassifier(n_regimes=3)
        x = tf.random.normal((10, 15, 5))
        out = clf(x)
        row_sums = tf.reduce_sum(out, axis=1).numpy()
        np.testing.assert_array_almost_equal(row_sums, np.ones(10), decimal=5)

    def test_values_in_unit_interval(self):
        clf = RegimeClassifier(n_regimes=4)
        x = tf.random.normal((6, 12, 3))
        out = clf(x).numpy()
        assert np.all(out >= 0.0 - 1e-6) and np.all(out <= 1.0 + 1e-6)

    def test_training_flag_accepted(self):
        clf = RegimeClassifier()
        x = tf.random.normal((4, 10, 3))
        out = clf(x, training=True)
        assert out.shape[0] == 4

    def test_different_n_regimes(self):
        for n in [2, 3, 5]:
            clf = RegimeClassifier(n_regimes=n)
            x = tf.random.normal((3, 8, 2))
            out = clf(x)
            assert out.shape == (3, n)


class TestRegimeConsistencyLoss:
    def test_output_is_scalar(self):
        probs = tf.random.uniform((8, 3))
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
        loss = regime_consistency_loss(probs)
        assert loss.shape == ()

    def test_identical_rows_give_zero_loss(self):
        row = tf.constant([[0.2, 0.5, 0.3]])
        probs = tf.tile(row, [6, 1])
        loss = regime_consistency_loss(probs)
        assert abs(float(loss)) < 1e-5

    def test_diverse_rows_give_positive_loss(self):
        probs = tf.eye(4)  # maximally diverse
        loss = regime_consistency_loss(probs)
        assert float(loss) > 0.0

    def test_loss_finite(self):
        probs = tf.random.uniform((16, 3))
        probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
        assert np.isfinite(float(regime_consistency_loss(probs)))
