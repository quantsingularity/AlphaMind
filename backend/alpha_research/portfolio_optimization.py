import matplotlib.pyplot as plt
from core.logging import get_logger

logger = get_logger(__name__)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model


class PortfolioOptimizer:
    """Advanced portfolio optimization using deep reinforcement learning.
    Extends the basic RL approach with more sophisticated state representation
    and reward functions.
    """

    def __init__(self, n_assets, lookback_window=60, hidden_units=64):
        self.n_assets = n_assets
        self.lookback_window = lookback_window
        self.hidden_units = hidden_units
        self.scaler = StandardScaler()
        # Build model after setting attributes
        self.model = self._build_model()
        self.performance_metrics = {}

    def _build_model(self):
        """Build the portfolio optimization model architecture"""
        # Input layers
        # State inputs are time series of features
        price_input = Input(
            shape=(self.lookback_window, self.n_assets), name="price_history"
        )
        vol_input = Input(
            shape=(self.lookback_window, self.n_assets), name="volatility_history"
        )
        # Assuming 5 macroeconomic factors, this is hardcoded based on the prompt's `macro_input` shape
        macro_input = Input(shape=(self.lookback_window, 5), name="macro_factors")
        # Action space input: current portfolio weights
        current_weights = Input(shape=(self.n_assets,), name="current_weights")

        # Process price history
        x1 = LSTM(self.hidden_units, return_sequences=True)(price_input)
        x1 = BatchNormalization()(x1)
        x1 = LSTM(self.hidden_units)(x1)  # Flatten to (batch_size, hidden_units)

        # Process volatility history
        x2 = LSTM(self.hidden_units, return_sequences=True)(vol_input)
        x2 = BatchNormalization()(x2)
        x2 = LSTM(self.hidden_units)(x2)

        # Process macro factors
        x3 = LSTM(self.hidden_units)(macro_input)

        # Combine all features (The full state representation for the Agent)
        combined = tf.keras.layers.concatenate([x1, x2, x3, current_weights])

        # Portfolio allocation layers (Policy Network)
        x = Dense(128, activation="relu")(combined)
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)

        # Output layer - portfolio weights. Softmax ensures weights sum to 1 (long-only portfolio)
        outputs = Dense(self.n_assets, activation="softmax", name="portfolio_weights")(
            x
        )

        # Create model
        model = Model(
            inputs=[price_input, vol_input, macro_input, current_weights],
            outputs=outputs,
        )

        # Compile model
        # The DRL objective is proxied by minimizing the custom loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self._portfolio_loss,
        )

        return model

    def _portfolio_loss(self, y_true, y_pred):
        """
        Custom loss function for portfolio optimization.
        Proxies the negative Sharpe ratio combined with risk penalties.

        Args:
            y_true: The true next-period asset returns (n_samples, n_assets).
            y_pred: The predicted portfolio weights (n_samples, n_assets).

        Returns:
            The loss value (scalar).
        """
        # Expected returns (y_true contains asset returns R_t+1)
        # R_p = W_t * R_t+1
        # Need to reshape y_pred for element-wise multiplication if it's not (batch_size, n_assets)
        portfolio_returns = tf.reduce_sum(y_pred * y_true, axis=1)

        # Mean and standard deviation of returns for the batch
        mean_returns = tf.reduce_mean(portfolio_returns)
        std_returns = tf.math.reduce_std(portfolio_returns)

        # Negative Sharpe ratio (We minimize loss, so we negate Sharpe)
        neg_sharpe = -mean_returns / (std_returns + 1e-6)

        # Concentration Penalty (L2 norm of weights - encourages diversification)
        concentration_penalty = tf.reduce_sum(tf.square(y_pred), axis=1)

        # Turnover Penalty (This would require previous weights in the y_true array)
        # turnover_penalty = tf.reduce_sum(tf.abs(y_pred - y_true[:, -self.n_assets:]), axis=1)

        # Combined loss: Minimize (Negative Sharpe) + Penalties
        # The constants (0.1) are hyperparameters
        return neg_sharpe + 0.1 * tf.reduce_mean(concentration_penalty)

    def preprocess_data(self, price_data, volatility_data, macro_data):
        """
        Preprocess input data for the model: scale and create lookback sequences.

        Args:
            price_data: Historical price data (n_samples, n_assets)
            volatility_data: Historical volatility data (n_samples, n_assets)
            macro_data: Macroeconomic factors (n_samples, n_macro_factors)

        Returns:
            X_price, X_vol, X_macro, y (next period returns)
        """
        # Scale data
        price_scaled = self.scaler.fit_transform(price_data)
        vol_scaled = self.scaler.fit_transform(volatility_data)
        macro_scaled = self.scaler.fit_transform(macro_data)

        # Calculate next period returns *before* creating sequences
        # Return_t+1 = (Price_t+1 / Price_t) - 1
        returns = (price_data[1:] / price_data[:-1]) - 1

        # Create sequences
        X_price, X_vol, X_macro, y = [], [], [], []

        # The loop runs until the point where the last sequence ends and the next-period return is available
        for i in range(len(price_scaled) - self.lookback_window):
            # Sequence for period t: [t-L+1, ..., t]
            X_price.append(price_scaled[i : i + self.lookback_window])
            X_vol.append(vol_scaled[i : i + self.lookback_window])
            X_macro.append(macro_scaled[i : i + self.lookback_window])

            # Target is the return *after* the last day of the sequence (t+1)
            # This is returns[i + self.lookback_window - 1] since returns is one day shorter
            y.append(returns[i + self.lookback_window - 1])

        return np.array(X_price), np.array(X_vol), np.array(X_macro), np.array(y)

    def train(
        self,
        price_data,
        volatility_data,
        macro_data,
        initial_weights=None,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
    ):
        """
        Train the portfolio optimization model

        Args:
            price_data: Historical price data
            volatility_data: Historical volatility data
            macro_data: Macroeconomic factors
            initial_weights: Initial portfolio weights (default: equal weights)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation

        Returns:
            Training history
        """
        logger.info("Preprocessing data and creating sequences...")
        # Preprocess data
        X_price, X_vol, X_macro, y = self.preprocess_data(
            price_data, volatility_data, macro_data
        )

        # Create initial weights if not provided
        if initial_weights is None:
            # We need one set of current weights for every input sample (sequence)
            initial_weights = np.ones((len(y), self.n_assets)) / self.n_assets

        logger.info("Splitting data for training and validation...")
        # Split data (use shuffle=False to maintain time-series order)
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_split, shuffle=False
        )

        # Training data
        train_X_price, train_X_vol = X_price[train_idx], X_vol[train_idx]
        train_X_macro, train_y = X_macro[train_idx], y[train_idx]
        train_weights = initial_weights[
            train_idx
        ]  # The weights held at the start of the training period

        # Validation data
        val_X_price, val_X_vol = X_price[val_idx], X_vol[val_idx]
        val_X_macro, val_y = X_macro[val_idx], y[val_idx]
        val_weights = initial_weights[val_idx]

        logger.info("Starting model training...")
        # Train model
        history = self.model.fit(
            [train_X_price, train_X_vol, train_X_macro, train_weights],
            train_y,  # The target is the next period returns for loss calculation
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([val_X_price, val_X_vol, val_X_macro, val_weights], val_y),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                )
            ],
            verbose=1,
        )

        return history

    def optimize_portfolio(
        self, price_data, volatility_data, macro_data, current_weights
    ):
        """
        Generate optimal portfolio weights (next action) from the current state.

        Args:
            price_data: Recent price data (lookback_window, n_assets)
            volatility_data: Recent volatility data (lookback_window, n_assets)
            macro_data: Recent macroeconomic factors (lookback_window, n_macro_factors)
            current_weights: Current portfolio weights (n_assets,)

        Returns:
            Optimal portfolio weights (n_assets,)
        """
        # Scale data using the *fitted* scaler
        price_scaled = self.scaler.transform(price_data)
        volatility_scaled = self.scaler.transform(volatility_data)
        macro_scaled = self.scaler.transform(macro_data)

        # Ensure all inputs have the correct shape for prediction: (1, lookback_window, features) or (1, features)
        price_scaled = np.array(price_scaled).reshape(
            1, self.lookback_window, self.n_assets
        )
        volatility_scaled = np.array(volatility_scaled).reshape(
            1, self.lookback_window, self.n_assets
        )
        # The number of macro factors is assumed to be the last dimension of macro_data
        n_macro_factors = macro_data.shape[1]
        macro_scaled = np.array(macro_scaled).reshape(
            1, self.lookback_window, n_macro_factors
        )
        current_weights = np.array(current_weights).reshape(1, self.n_assets)

        # Generate optimal weights
        optimal_weights = self.model.predict(
            [price_scaled, volatility_scaled, macro_scaled, current_weights], verbose=0
        )

        # The prediction output is (1, n_assets), return the 1D array
        return optimal_weights[0]

    def backtest(
        self,
        price_data,
        volatility_data,
        macro_data,
        initial_capital=10000,
        transaction_cost=0.001,
    ):
        """
        Backtest the portfolio optimization strategy in a sequential, walk-forward manner.

        Args:
            price_data: Historical price data
            volatility_data: Historical volatility data
            macro_data: Macroeconomic factors
            initial_capital: Starting capital for the backtest
            transaction_cost: Transaction cost as a fraction of trade value

        Returns:
            DataFrame with backtest results and performance metrics
        """
        logger.info("\nStarting backtest (walk-forward simulation)...")
        # Preprocess data to get aligned sequences and next-period returns
        X_price, X_vol, X_macro, returns = self.preprocess_data(
            price_data, volatility_data, macro_data
        )

        # Initialize backtest variables
        portfolio_value = [initial_capital]
        # Start with equal weights
        portfolio_weights = np.ones(self.n_assets) / self.n_assets
        drawdown_history = [0]
        portfolio_returns = []

        # Run backtest - loop through each available trading day (sequence)
        for i in range(len(X_price)):
            # 1. Get optimal weights (action) for current state
            optimal_weights = self.optimize_portfolio(
                X_price[i], X_vol[i], X_macro[i], portfolio_weights
            )

            # 2. Calculate turnover and transaction costs
            # Turnover is the sum of absolute changes in weights
            turnover = np.sum(np.abs(optimal_weights - portfolio_weights))

            # Costs are applied to the total portfolio value
            transaction_costs = turnover * transaction_cost * portfolio_value[-1]

            # 3. Apply the predicted portfolio weights to the *next* day's returns
            # returns[i] is the return from the last day of the sequence (t) to the next day (t+1)
            current_return = np.sum(optimal_weights * returns[i])
            portfolio_returns.append(current_return)

            # 4. Update portfolio value
            new_value = portfolio_value[-1] * (1 + current_return) - transaction_costs
            portfolio_value.append(new_value)

            # 5. Update weights for next period
            # The weights for the *next* day's optimization are the weights just used
            portfolio_weights = optimal_weights

            # Calculate and store drawdown for the new value
            peak = np.max(portfolio_value)
            drawdown_history.append((new_value - peak) / peak)

        # --- Performance Metrics Calculation ---
        portfolio_values_series = pd.Series(portfolio_value[1:])
        portfolio_returns_series = pd.Series(portfolio_returns)

        # Annualized Sharpe Ratio (assuming daily returns and 252 trading days)
        # R_f (risk-free rate) is assumed to be 0
        sharpe_ratio = (
            portfolio_returns_series.mean()
            / portfolio_returns_series.std()
            * np.sqrt(252)
        )

        # Maximum Drawdown
        max_drawdown = np.min(drawdown_history)

        # Annualized Volatility
        volatility = portfolio_returns_series.std() * np.sqrt(252)

        # Store metrics
        self.performance_metrics = {
            "total_return": portfolio_value[-1] / initial_capital - 1,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "annualized_volatility": volatility,
        }

        # Create results DataFrame
        # The first entry in portfolio_value is initial_capital, which has a 0 return
        results = pd.DataFrame(
            {
                "portfolio_value": portfolio_value[1:],  # Align with returns
                "returns": portfolio_returns,
                "drawdown": drawdown_history[1:],
            },
            index=price_data.index[
                self.lookback_window :
            ],  # Align index with available returns
        )

        logger.info("Backtest Complete. Performance:")
        logger.info(pd.Series(self.performance_metrics))
        return results

    def plot_results(self, results):
        """Plot backtest results"""
        plt.figure(figsize=(14, 10))

        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(results["portfolio_value"])
        plt.title("Portfolio Value")
        plt.ylabel("Value ($)")
        plt.grid(True)

        # Plot drawdowns
        plt.subplot(2, 1, 2)
        plt.fill_between(results.index, 0, results["drawdown"], color="red", alpha=0.3)
        plt.title("Drawdowns")
        plt.ylabel("Drawdown (%)")
        plt.xlabel("Date")
        plt.grid(True)

        plt.tight_layout()
        return plt.gcf()

    def save(self, filepath):
        """Save the model to disk"""
        logger.info(f"Saving model to {filepath}")
        self.model.save(filepath)

    def load(self, filepath):
        """Load the model from disk"""
        logger.info(f"Loading model from {filepath}")
        # Need to provide the custom loss function when loading
        self.model = tf.keras.models.load_model(
            filepath, custom_objects={"_portfolio_loss": self._portfolio_loss}
        )
