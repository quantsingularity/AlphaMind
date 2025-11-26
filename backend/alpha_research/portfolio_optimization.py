# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
# from tensorflow.keras.models import Model


# class PortfolioOptimizer:
#    """"""
##     Advanced portfolio optimization using deep reinforcement learning.
##     Extends the basic RL approach with more sophisticated state representation
##     and reward functions.
#    """"""

#     def __init__(self, n_assets, lookback_window=60, hidden_units=64):
#         self.n_assets = n_assets
#         self.lookback_window = lookback_window
#         self.hidden_units = hidden_units
#         self.model = self._build_model()
#         self.scaler = StandardScaler()

#     def _build_model(self):
#        """Build the portfolio optimization model architecture"""
#        # Input layers
##         price_input = Input(
##             shape=(self.lookback_window, self.n_assets), name="price_history"
#        )
##         vol_input = Input(
##             shape=(self.lookback_window, self.n_assets), name="volatility_history"
#        )
##         macro_input = Input(shape=(self.lookback_window, 5), name="macro_factors")
##         current_weights = Input(shape=(self.n_assets,), name="current_weights")
#
#        # Process price history
##         x1 = LSTM(self.hidden_units, return_sequences=True)(price_input)
##         x1 = BatchNormalization()(x1)
##         x1 = LSTM(self.hidden_units)(x1)
#
#        # Process volatility history
##         x2 = LSTM(self.hidden_units, return_sequences=True)(vol_input)
##         x2 = BatchNormalization()(x2)
##         x2 = LSTM(self.hidden_units)(x2)
#
#        # Process macro factors
##         x3 = LSTM(self.hidden_units)(macro_input)
#
#        # Combine all features
##         combined = tf.keras.layers.concatenate([x1, x2, x3, current_weights])
#
#        # Portfolio allocation layers
##         x = Dense(128, activation="relu")(combined)
##         x = Dropout(0.3)(x)
##         x = Dense(64, activation="relu")(x)
##         x = Dropout(0.3)(x)
#
#        # Output layer - portfolio weights
##         outputs = Dense(self.n_assets, activation="softmax", name="portfolio_weights")(
#            x
#        )
#
#        # Create model
##         model = Model(
##             inputs=[price_input, vol_input, macro_input, current_weights],
##             outputs=outputs,
#        )
#
#        # Compile model
##         model.compile(
##             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
##             loss=self._portfolio_loss,
#        )
#
##         return model
#
##     def _portfolio_loss(self, y_true, y_pred):
#        """"""
#         Custom loss function for portfolio optimization
#         Combines Sharpe ratio maximization with risk constraints
#        """"""
#        # Expected returns (assuming y_true contains asset returns)
##         portfolio_returns = tf.reduce_sum(y_pred * y_true, axis=1)
#
#        # Mean and standard deviation of returns
##         mean_returns = tf.reduce_mean(portfolio_returns)
##         std_returns = tf.math.reduce_std(portfolio_returns)
#
#        # Negative Sharpe ratio (we minimize loss, so negate Sharpe)
##         neg_sharpe = -mean_returns / (std_returns + 1e-6)
#
#        # Add regularization for diversification
##         concentration_penalty = tf.reduce_sum(tf.square(y_pred), axis=1)
#
#        # Add turnover penalty (assuming last column of y_true contains previous weights)
#        # turnover_penalty = tf.reduce_sum(tf.abs(y_pred - y_true[:, -self.n_assets:]), axis=1)
#
#        # Combined loss
##         return neg_sharpe + 0.1 * concentration_penalty  # + 0.2 * turnover_penalty
#
##     def preprocess_data(self, price_data, volatility_data, macro_data):
#        """"""
#         Preprocess input data for the model

#         Args:
#             price_data: Historical price data (n_samples, n_assets)
#             volatility_data: Historical volatility data (n_samples, n_assets)
#             macro_data: Macroeconomic factors (n_samples, n_macro_factors)

#         Returns:
#             Processed data ready for model input
#        """"""
#        # Scale data
##         price_scaled = self.scaler.fit_transform(price_data)
##         vol_scaled = self.scaler.fit_transform(volatility_data)
##         macro_scaled = self.scaler.fit_transform(macro_data)
#
#        # Create sequences
##         X_price, X_vol, X_macro, y = [], [], [], []
#
##         for i in range(len(price_scaled) - self.lookback_window):
##             X_price.append(price_scaled[i : i + self.lookback_window])
##             X_vol.append(vol_scaled[i : i + self.lookback_window])
##             X_macro.append(macro_scaled[i : i + self.lookback_window])
#
#            # Target is the next period return
##             y.append(
##                 price_data[i + self.lookback_window]
##                 / price_data[i + self.lookback_window - 1]
##                 - 1
#            )
#
##         return np.array(X_price), np.array(X_vol), np.array(X_macro), np.array(y)
#
##     def train(
##         self,
##         price_data,
##         volatility_data,
##         macro_data,
##         initial_weights=None,
##         epochs=50,
##         batch_size=32,
##         validation_split=0.2,
#    ):
#        """"""
#         Train the portfolio optimization model

#         Args:
#             price_data: Historical price data
#             volatility_data: Historical volatility data
#             macro_data: Macroeconomic factors
#             initial_weights: Initial portfolio weights (default: equal weights)
#             epochs: Number of training epochs
#             batch_size: Batch size for training
#             validation_split: Fraction of data to use for validation

#         Returns:
#             Training history
#        """"""
#        # Preprocess data
##         X_price, X_vol, X_macro, y = self.preprocess_data(
##             price_data, volatility_data, macro_data
#        )
#
#        # Create initial weights if not provided
##         if initial_weights is None:
##             initial_weights = np.ones((len(y), self.n_assets)) / self.n_assets
#
#        # Split data
##         indices = np.arange(len(y))
##         train_idx, val_idx = train_test_split(
##             indices, test_size=validation_split, shuffle=False
#        )
#
#        # Training data
##         train_X_price, train_X_vol = X_price[train_idx], X_vol[train_idx]
##         train_X_macro, train_y = X_macro[train_idx], y[train_idx]
##         train_weights = initial_weights[train_idx]
#
#        # Validation data
##         val_X_price, val_X_vol = X_price[val_idx], X_vol[val_idx]
##         val_X_macro, val_y = X_macro[val_idx], y[val_idx]
##         val_weights = initial_weights[val_idx]
#
#        # Train model
##         history = self.model.fit(
##             [train_X_price, train_X_vol, train_X_macro, train_weights],
##             train_y,
##             epochs=epochs,
##             batch_size=batch_size,
##             validation_data=([val_X_price, val_X_vol, val_X_macro, val_weights], val_y),
##             callbacks=[
##                 tf.keras.callbacks.EarlyStopping(
##                     monitor="val_loss", patience=10, restore_best_weights=True
#                )
#            ],
#        )
#
##         return history
#
##     def optimize_portfolio(
##         self, price_data, volatility_data, macro_data, current_weights
#    ):
#        """"""
#         Generate optimal portfolio weights

#         Args:
#             price_data: Recent price data (lookback_window, n_assets)
#             volatility_data: Recent volatility data (lookback_window, n_assets)
#             macro_data: Recent macroeconomic factors (lookback_window, n_macro_factors)
#             current_weights: Current portfolio weights (n_assets,)

#         Returns:
#             Optimal portfolio weights
#        """"""
#        # Ensure data has correct shape
##         price_data = np.array(price_data).reshape(
##             1, self.lookback_window, self.n_assets
#        )
##         volatility_data = np.array(volatility_data).reshape(
##             1, self.lookback_window, self.n_assets
#        )
##         macro_data = np.array(macro_data).reshape(1, self.lookback_window, 5)
##         current_weights = np.array(current_weights).reshape(1, self.n_assets)
#
#        # Generate optimal weights
##         optimal_weights = self.model.predict(
##             [price_data, volatility_data, macro_data, current_weights]
#        )
#
##         return optimal_weights[0]
#
##     def backtest(
##         self,
##         price_data,
##         volatility_data,
##         macro_data,
##         initial_capital=10000,
##         transaction_cost=0.001,
#    ):
#        """"""
#         Backtest the portfolio optimization strategy

#         Args:
#             price_data: Historical price data
#             volatility_data: Historical volatility data
#             macro_data: Macroeconomic factors
#             initial_capital: Starting capital for the backtest
#             transaction_cost: Transaction cost as a fraction of trade value

#         Returns:
#             DataFrame with backtest results and performance metrics
#        """"""
#        # Preprocess data
##         X_price, X_vol, X_macro, returns = self.preprocess_data(
##             price_data, volatility_data, macro_data
#        )
#
#        # Initialize backtest variables
##         portfolio_value = [initial_capital]
##         portfolio_weights = (
##             np.ones(self.n_assets) / self.n_assets
##         )  # Start with equal weights
#
#        # Run backtest
##         for i in range(len(X_price)):
#            # Get optimal weights for current state
##             optimal_weights = self.optimize_portfolio(
##                 X_price[i], X_vol[i], X_macro[i], portfolio_weights
#            )
#
#            # Calculate turnover
##             turnover = np.sum(np.abs(optimal_weights - portfolio_weights))
#
#            # Apply transaction costs
##             transaction_costs = turnover * transaction_cost * portfolio_value[-1]
#
#            # Update portfolio value
##             current_return = np.sum(optimal_weights * returns[i])
##             new_value = portfolio_value[-1] * (1 + current_return) - transaction_costs
##             portfolio_value.append(new_value)
#
#            # Update weights for next period
##             portfolio_weights = optimal_weights
#
#        # Calculate performance metrics
##         portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
##         sharpe_ratio = (
##             np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
#        )
#
#        # Calculate drawdowns
##         peak = np.maximum.accumulate(portfolio_value)
##         drawdown = (portfolio_value - peak) / peak
##         max_drawdown = np.min(drawdown)
#
#        # Store metrics
##         self.performance_metrics = {
#            "total_return": portfolio_value[-1] / initial_capital - 1,
#            "sharpe_ratio": sharpe_ratio,
#            "max_drawdown": max_drawdown,
#            "volatility": np.std(portfolio_returns) * np.sqrt(252),
#        }
#
#        # Create results DataFrame
##         results = pd.DataFrame(
#            {
#                "portfolio_value": portfolio_value,
#                "returns": np.append([0], portfolio_returns),
#                "drawdown": drawdown,
#            }
#        )
#
##         return results
#
##     def plot_results(self, results):
#        """Plot backtest results"""
#         plt.figure(figsize=(14, 10))

#         # Plot portfolio value
#         plt.subplot(2, 1, 1)
#         plt.plot(results["portfolio_value"])
#         plt.title("Portfolio Value")
#         plt.grid(True)

#         # Plot drawdowns
#         plt.subplot(2, 1, 2)
#         plt.fill_between(
#             range(len(results)), 0, results["drawdown"], color="red", alpha=0.3
#         )
#         plt.title("Drawdowns")
#         plt.grid(True)

#         plt.tight_layout()
#         return plt.gcf()

#     def save(self, filepath):
#        """Save the model to disk"""
##         self.model.save(filepath)
#
##     def load(self, filepath):
#        """Load the model from disk"""
#         self.model = tf.keras.models.load_model(
#             filepath, custom_objects={"_portfolio_loss": self._portfolio_loss}
#         )
