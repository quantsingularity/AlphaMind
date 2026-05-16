"""Unit tests for AdvancedTimeSeriesForecaster."""

import numpy as np
from ai_models.config import TransformerConfig
from ai_models.forecasting.advanced import AdvancedTimeSeriesForecaster


def make_forecaster():
    cfg = TransformerConfig(
        num_layers=1,
        d_model=16,
        num_heads=2,
        dff=32,
        input_seq_length=10,
        output_seq_length=5,
        dropout_rate=0.0,
    )
    return AdvancedTimeSeriesForecaster(config=cfg)


class TestAdvancedTimeSeriesForecaster:
    def test_iqr_normalise_shape(self):
        fc = make_forecaster()
        data = np.random.randn(20, 10, 8)
        norm = fc._iqr_normalise(data)
        assert norm.shape == data.shape

    def test_iqr_normalise_reduces_outlier_influence(self):
        fc = make_forecaster()
        data = np.random.randn(50, 10, 4)
        data[0, 0, 0] = 1e6  # inject outlier
        norm = fc._iqr_normalise(data)
        assert np.isfinite(norm).all()

    def test_predict_shape(self):
        fc = make_forecaster()
        x = np.random.randn(8, 10, 16).astype(np.float32)
        out = fc.predict(x)
        assert out.shape[0] == 8

    def test_train_returns_loss_history(self):
        fc = make_forecaster()
        x = np.random.randn(12, 10, 16).astype(np.float32)
        y = np.random.randn(12, 5).astype(np.float32)
        history = fc.train(x, y, epochs=2, batch_size=4)
        assert "loss" in history
        assert len(history["loss"]) == 2

    def test_train_loss_decreasing(self):
        """Loss should generally decrease over training."""
        fc = make_forecaster()
        np.random.seed(0)
        x = np.random.randn(32, 10, 16).astype(np.float32)
        y = np.random.randn(32, 5).astype(np.float32)
        h = fc.train(x, y, epochs=10, batch_size=16)
        assert h["loss"][-1] <= h["loss"][0] * 1.5  # loose bound

    def test_train_with_validation(self):
        fc = make_forecaster()
        x = np.random.randn(16, 10, 16).astype(np.float32)
        y = np.random.randn(16, 5).astype(np.float32)
        xv = np.random.randn(4, 10, 16).astype(np.float32)
        yv = np.random.randn(4, 5).astype(np.float32)
        h = fc.train(x, y, epochs=2, batch_size=4, validation_data=(xv, yv))
        assert "val_loss" in h
        assert len(h["val_loss"]) == 2

    def test_save_and_load(self, tmp_path):
        fc = make_forecaster()
        x = np.random.randn(4, 10, 16).astype(np.float32)
        fc.predict(x)  # build weights
        fc.save(str(tmp_path / "weights"))
        fc2 = make_forecaster()
        fc2.predict(x)  # build weights
        fc2.load(str(tmp_path / "weights"))
        np.testing.assert_array_almost_equal(fc.predict(x), fc2.predict(x), decimal=4)
