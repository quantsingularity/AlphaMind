# import os
# import sys

# import numpy as np
# import pandas as pd
# import pytest
# import tensorflow as tf

# Add the backend directory to the path
# sys.path.append("/home/ubuntu/alphamind_project/backend")

# from alpha_research.portfolio_optimization import PortfolioOptimizer


# Define the custom loss function globally for serialization
# @tf.keras.utils.register_keras_serializable()
# def portfolio_loss(y_true, y_pred):
#    """Custom loss function to maximize portfolio returns."""
##     portfolio_returns = tf.reduce_sum(y_true * y_pred, axis=1)
##     return -tf.reduce_mean(portfolio_returns)
#
#
## N_ASSETS = 3
## LOOKBACK_WINDOW = 10
## HIDDEN_UNITS = 32
## N_SAMPLES = 50
## N_MACRO_FACTORS = 5
#
#
## @pytest.fixture
## def optimizer():
#    """Fixture for a PortfolioOptimizer instance."""
    # Ensure the optimizer internally compiles its model with the registered loss
#     opt = PortfolioOptimizer(
#         n_assets=N_ASSETS, lookback_window=LOOKBACK_WINDOW, hidden_units=HIDDEN_UNITS
    # We assume the optimizer compiles the model with the custom loss internally.
    # If not, the test needs to be adjusted or the class needs modification.
    # For testing save/load, we might need to explicitly compile here if not done in __init__
#     try:
        # Check if already compiled with the correct loss
#         if opt.model.loss.__name__ != portfolio_loss.__name__:
#             opt.model.compile(
#                 optimizer="adam", loss=portfolio_loss
#             )  # Removed backslashes
#     except AttributeError:  # If model.loss is not set or model not compiled
#         opt.model.compile(optimizer="adam", loss=portfolio_loss)  # Removed backslashes
#     return opt


# @pytest.fixture
# def sample_data():
#    """Fixture for sample input data."""
##     np.random.seed(42)
##     price_data = np.random.rand(N_SAMPLES, N_ASSETS) * 100 + 50
##     volatility_data = np.random.rand(N_SAMPLES, N_ASSETS) * 0.2 + 0.1
##     macro_data = np.random.rand(N_SAMPLES, N_MACRO_FACTORS)
##     return price_data, volatility_data, macro_data
#
#
## def test_optimizer_init(optimizer):
#    """Test PortfolioOptimizer initialization."""
#     assert optimizer.n_assets == N_ASSETS
#     assert optimizer.lookback_window == LOOKBACK_WINDOW
#     assert isinstance(optimizer.model, tf.keras.Model)
#     assert isinstance(
#         optimizer.scaler, object
#     )  # StandardScaler is tricky to assert type directly
    # Check if the model is compiled with the correct loss
#     assert optimizer.model.loss.__name__ == portfolio_loss.__name__


# def test_build_model_structure(optimizer):
#    """Test the structure and output shape of the built model."""
##     model = optimizer.model
#    # Check input names and shapes (adjust based on actual model structure)
##     assert len(model.inputs) == 4
##     assert model.input_shape[0] == (None, LOOKBACK_WINDOW, N_ASSETS)  # price_input
##     assert model.input_shape[1] == (None, LOOKBACK_WINDOW, N_ASSETS)  # vol_input
##     assert model.input_shape[2] == (
##         None,
##         LOOKBACK_WINDOW,
##         N_MACRO_FACTORS,
##     )  # macro_input
##     assert model.input_shape[3] == (None, N_ASSETS)  # current_weights
#    # Check output shape
##     assert model.output_shape == (None, N_ASSETS)
#
#
## def test_portfolio_loss_calculation():
#    """Test the custom portfolio loss function calculation directly."""
#     y_true = tf.constant([[0.01, -0.005, 0.02], [0.005, 0.01, -0.01]], dtype=tf.float32)
#     y_pred = tf.constant(
#         [[0.4, 0.3, 0.3], [0.5, 0.2, 0.3]], dtype=tf.float32
#     )  # Example weights

#     loss = portfolio_loss(y_true, y_pred)

#     assert isinstance(loss, tf.Tensor)
    # Check if the shape is scalar
#     assert loss.shape == tf.TensorShape([])
    # Check the calculation logic
#     expected_return_1 = (
#         0.4 * 0.01 + 0.3 * (-0.005) + 0.3 * 0.02
#     )  # 0.004 - 0.0015 + 0.006 = 0.0085
#     expected_return_2 = (
#         0.5 * 0.005 + 0.2 * 0.01 + 0.3 * (-0.01)
#     )  # 0.0025 + 0.002 - 0.003 = 0.0015
#     expected_loss = -(0.0085 + 0.0015) / 2  # -0.01 / 2 = -0.005
#     assert tf.experimental.numpy.isclose(loss, expected_loss)


# def test_preprocess_data_shapes(optimizer, sample_data):
#    """Test the output shapes of the preprocess_data method."""
##     price_data, volatility_data, macro_data = sample_data
##     X_price, X_vol, X_macro, y = optimizer.preprocess_data(
##         price_data, volatility_data, macro_data
#    )
#
##     expected_samples = N_SAMPLES - LOOKBACK_WINDOW
##     assert X_price.shape == (expected_samples, LOOKBACK_WINDOW, N_ASSETS)
##     assert X_vol.shape == (expected_samples, LOOKBACK_WINDOW, N_ASSETS)
##     assert X_macro.shape == (expected_samples, LOOKBACK_WINDOW, N_MACRO_FACTORS)
##     assert y.shape == (expected_samples, N_ASSETS)
#
#
## def test_optimize_portfolio_output(optimizer, sample_data):
#    """Test the output of the optimize_portfolio method."""
#     price_data, volatility_data, macro_data = sample_data

    # Get data for one prediction step
#     recent_price = price_data[-LOOKBACK_WINDOW:]
#     recent_vol = volatility_data[-LOOKBACK_WINDOW:]
#     recent_macro = macro_data[-LOOKBACK_WINDOW:]
#     current_weights = np.ones(N_ASSETS) / N_ASSETS

#     optimal_weights = optimizer.optimize_portfolio(
#         recent_price, recent_vol, recent_macro, current_weights

#     assert optimal_weights.shape == (N_ASSETS,)
#     assert np.isclose(
#         np.sum(optimal_weights), 1.0
#     )  # Weights should sum to 1 (due to softmax)
#     assert np.all(optimal_weights >= 0)  # Weights should be non-negative


# def test_backtest_output_structure(optimizer, sample_data):
#    """Test the structure of the backtest output DataFrame and metrics."""
##     price_data, volatility_data, macro_data = sample_data
#
#    # Mock the optimize_portfolio method to return fixed weights for simplicity
##     optimizer.optimize_portfolio = lambda p, v, m, cw: np.ones(N_ASSETS) / N_ASSETS
#
##     results = optimizer.backtest(price_data, volatility_data, macro_data)
#
##     assert isinstance(results, pd.DataFrame)
##     assert "portfolio_value" in results.columns
##     assert "returns" in results.columns
##     assert "drawdown" in results.columns
##     assert len(results) == (N_SAMPLES - LOOKBACK_WINDOW + 1)
#
#    # Check performance metrics dictionary
##     assert hasattr(optimizer, "performance_metrics")
##     assert isinstance(optimizer.performance_metrics, dict)
##     assert "total_return" in optimizer.performance_metrics
##     assert "sharpe_ratio" in optimizer.performance_metrics
##     assert "max_drawdown" in optimizer.performance_metrics
##     assert "volatility" in optimizer.performance_metrics
#
#
## def test_save_load_model(optimizer, tmp_path):
#    """Test saving and loading the model."""
#     model_path = os.path.join(str(tmp_path), "test_portfolio_model.keras")
    # Ensure the model is compiled with the registered loss before saving
#     if (
#         not hasattr(optimizer.model, "loss")
#         or optimizer.model.loss.__name__ != portfolio_loss.__name__
    ):
#         optimizer.model.compile(
#             optimizer="adam", loss=portfolio_loss
#         )  # Removed backslashes

#     optimizer.save(model_path)  # Assume this saves the Keras model internally

#     assert os.path.exists(model_path)

    # Load the Keras model directly using tf.keras.models.load_model
    # The custom loss is registered, so it should load automatically
#     loaded_keras_model = tf.keras.models.load_model(model_path)

    # Create a new optimizer instance and assign the loaded model
#     new_optimizer = PortfolioOptimizer(
#         n_assets=N_ASSETS, lookback_window=LOOKBACK_WINDOW, hidden_units=HIDDEN_UNITS
#     new_optimizer.model = loaded_keras_model

    # Check if the loaded model is a Keras model
#     assert isinstance(new_optimizer.model, tf.keras.Model)
    # Check if the loss function is correctly loaded
#     assert (
#         new_optimizer.model.loss.__name__ == portfolio_loss.__name__
#     )  # Compare by name

    # Optional: Compare weights or predict with both models on same input
    # Ensure models are built before getting weights
#     dummy_price = np.random.rand(1, LOOKBACK_WINDOW, N_ASSETS)
#     dummy_vol = np.random.rand(1, LOOKBACK_WINDOW, N_ASSETS)
#     dummy_macro = np.random.rand(1, LOOKBACK_WINDOW, N_MACRO_FACTORS)
#     dummy_weights = np.random.rand(1, N_ASSETS)
#     _ = optimizer.model([dummy_price, dummy_vol, dummy_macro, dummy_weights])
#     _ = new_optimizer.model([dummy_price, dummy_vol, dummy_macro, dummy_weights])
#     assert len(optimizer.model.get_weights()) == len(new_optimizer.model.get_weights())
#     for w1, w2 in zip(optimizer.model.get_weights(), new_optimizer.model.get_weights()):
#         assert np.allclose(w1, w2)
