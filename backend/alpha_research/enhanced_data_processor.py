# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.ensemble import IsolationForest  # Moved import to top level
# from sklearn.preprocessing import (  # Added MinMaxScaler, RobustScaler
#     MinMaxScaler,
#     RobustScaler,
#     StandardScaler,
)
# import tensorflow as tf


# class EnhancedDataProcessor:
#    """"""
##     Advanced data processing pipeline for financial time series data.
##     Includes data cleaning, feature engineering, and anomaly detection.
#    """"""

#     def __init__(self):
#         self.scalers = {}
#         self.pca_models = {}
#         self.anomaly_detector = None

#     def clean_data(self, data, fill_method="ffill", outlier_std_threshold=3.0):
#        """"""
##         Clean financial data by handling missing values and outliers
#
##         Args:
##             data: DataFrame containing financial time series data
##             fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
##             outlier_std_threshold: Threshold (in standard deviations) to identify outliers
#
##         Returns:
##             Cleaned DataFrame
#        """"""
        # Create a copy to avoid modifying the original data
#         cleaned_data = data.copy()

        # Handle missing values
#         if fill_method == "ffill":
#             cleaned_data = cleaned_data.fillna(method="ffill")
            # If there are still NaNs at the beginning, fill them with the next valid value
#             cleaned_data = cleaned_data.fillna(method="bfill")
#         elif fill_method == "bfill":
#             cleaned_data = cleaned_data.fillna(method="bfill")
            # If there are still NaNs at the end, fill them with the previous valid value
#             cleaned_data = cleaned_data.fillna(method="ffill")
#         elif fill_method == "interpolate":
#             cleaned_data = cleaned_data.interpolate(method="time")
            # Fill any remaining NaNs at the edges
#             cleaned_data = cleaned_data.fillna(method="ffill").fillna(method="bfill")

        # Handle outliers - replace with threshold values
#         numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             col_mean = cleaned_data[col].mean()
#             col_std = cleaned_data[col].std()

            # Define upper and lower bounds
#             upper_bound = col_mean + outlier_std_threshold * col_std
#             lower_bound = col_mean - outlier_std_threshold * col_std

            # Replace outliers
#             cleaned_data[col] = np.where(
#                 cleaned_data[col] > upper_bound,
#                 upper_bound,
#                 np.where(
#                     cleaned_data[col] < lower_bound, lower_bound, cleaned_data[col]
                ),
            )

#         return cleaned_data

#     def engineer_features(self, data, window_sizes=[5, 10, 20, 50], include_ta=True):
#        """"""
##         Engineer features from financial time series data
#
##         Args:
##             data: DataFrame with at least OHLCV columns (open, high, low, close, volume)
##             window_sizes: List of window sizes for rolling calculations
##             include_ta: Whether to include technical indicators
#
##         Returns:
##             DataFrame with engineered features
#        """"""
        # Create a copy to avoid modifying the original data
#         df = data.copy()

        # Ensure required columns exist
#         required_cols = ["open", "high", "low", "close", "volume"]
#         missing_cols = [col for col in required_cols if col not in df.columns]

#         if missing_cols:
#             raise ValueError(f"Missing required columns: {missing_cols}")

        # Basic price features
#         df["returns"] = df["close"].pct_change()
#         df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Range features
#         df["daily_range"] = df["high"] - df["low"]
#         df["daily_range_pct"] = df["daily_range"] / df["close"]

        # Volume features
#         df["volume_change"] = df["volume"].pct_change()
#         df["volume_ma"] = df["volume"].rolling(window=20).mean()
#         df["relative_volume"] = df["volume"] / df["volume_ma"]

        # Rolling window features
#         for window in window_sizes:
            # Price momentum
#             df[f"return_{window}d"] = df["close"].pct_change(window)

            # Volatility
#             df[f"volatility_{window}d"] = df["returns"].rolling(window=window).std()

            # Moving averages
#             df[f"ma_{window}d"] = df["close"].rolling(window=window).mean()

            # Price relative to moving average
#             df[f"close_to_ma_{window}d"] = df["close"] / df[f"ma_{window}d"]

            # Volume features
#             df[f"volume_ma_{window}d"] = df["volume"].rolling(window=window).mean()

        # Technical indicators (if requested)
#         if include_ta:
            # RSI (Relative Strength Index)
#             delta = df["close"].diff()
#             gain = delta.where(delta > 0, 0)
#             loss = -delta.where(delta < 0, 0)

#             avg_gain = gain.rolling(window=14).mean()
#             avg_loss = loss.rolling(window=14).mean()

#             rs = avg_gain / avg_loss
#             df["rsi_14"] = 100 - (100 / (1 + rs))

            # MACD (Moving Average Convergence Divergence)
#             ema_12 = df["close"].ewm(span=12, adjust=False).mean()
#             ema_26 = df["close"].ewm(span=26, adjust=False).mean()
#             df["macd"] = ema_12 - ema_26
#             df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
#             df["macd_hist"] = df["macd"] - df["macd_signal"]

            # Bollinger Bands
#             df["bb_middle"] = df["close"].rolling(window=20).mean()
#             df["bb_std"] = df["close"].rolling(window=20).std()
#             df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
#             df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
#             df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

            # ATR (Average True Range)
#             tr1 = df["high"] - df["low"]
#             tr2 = abs(df["high"] - df["close"].shift(1))
#             tr3 = abs(df["low"] - df["close"].shift(1))
#             df["true_range"] = pd.DataFrame([tr1, tr2, tr3]).max()
#             df["atr_14"] = df["true_range"].rolling(window=14).mean()

        # Drop NaN values resulting from calculations
#         df = df.dropna()

#         return df

#     def normalize_data(self, data, method="standard", fit=True, group_name="default"):
#        """"""
##         Normalize data using various scaling methods
#
##         Args:
##             data: DataFrame or array to normalize
##             method: Normalization method ('standard', 'minmax', 'robust')
##             fit: Whether to fit a new scaler or use a previously fit one
##             group_name: Identifier for the scaler group
#
##         Returns:
##             Normalized data
#        """"""
#         if method == "standard":
#             if fit or group_name not in self.scalers:
#                 self.scalers[group_name] = StandardScaler()
#                 return self.scalers[group_name].fit_transform(data)
#             else:
#                 return self.scalers[group_name].transform(data)

#         elif method == "minmax":
#             if fit or group_name not in self.scalers:
#                 self.scalers[group_name] = MinMaxScaler()
#                 return self.scalers[group_name].fit_transform(data)
#             else:
#                 return self.scalers[group_name].transform(data)

#         elif method == "robust":
#             if fit or group_name not in self.scalers:
#                 self.scalers[group_name] = RobustScaler()
#                 return self.scalers[group_name].fit_transform(data)
#             else:
#                 return self.scalers[group_name].transform(data)

#         else:
#             raise ValueError(f"Unknown normalization method: {method}")

#     def reduce_dimensions(
#         self, data, n_components=0.95, fit=True, group_name="default"
    ):
#        """"""
##         Reduce dimensionality of data using PCA
#
##         Args:
##             data: DataFrame or array to reduce
##             n_components: Number of components or variance ratio to preserve
##             fit: Whether to fit a new PCA model or use a previously fit one
##             group_name: Identifier for the PCA model group
#
##         Returns:
##             Reduced data
#        """"""
#         if fit or group_name not in self.pca_models:
#             self.pca_models[group_name] = PCA(n_components=n_components)
#             return self.pca_models[group_name].fit_transform(data)
#         else:
#             return self.pca_models[group_name].transform(data)

#     def detect_anomalies(self, data, contamination=0.05, fit=True):
#        """"""
##         Detect anomalies in financial data
#
##         Args:
##             data: DataFrame or array to analyze
##             contamination: Expected proportion of anomalies
##             fit: Whether to fit a new anomaly detector or use a previously fit one
#
##         Returns:
##             Boolean array indicating anomalies (True) and normal points (False)
#        """"""

#         if fit or self.anomaly_detector is None:
#             self.anomaly_detector = IsolationForest(
#                 contamination=contamination, random_state=42
            )
#             self.anomaly_detector.fit(data)

        # Predict returns -1 for anomalies and 1 for normal points
#         predictions = self.anomaly_detector.predict(data)

        # Convert to boolean array (True for anomalies)
#         return predictions == -1

#     def create_sequences(self, data, seq_length, target_col=None, target_horizon=1):
#        """"""
##         Create sequences for time series modeling
#
##         Args:
##             data: DataFrame or array of features
##             seq_length: Length of input sequences
##             target_col: Target column name or index (if None, uses all columns)
##             target_horizon: Forecast horizon for the target
#
##         Returns:
##             X: Input sequences
##             y: Target values (if target_col is provided)
#        """"""
        X = []
#         y = [] if target_col is not None else None

#         for i in range(
#             len(data)
#             - seq_length
#             - (target_horizon if target_col is not None else 0)
#             + 1
        ):
#             X.append(data[i : i + seq_length])

#             if target_col is not None:
#                 if isinstance(target_col, str):
#                     target_idx = data.columns.get_loc(target_col)
#                 else:
#                     target_idx = target_col

#                 y.append(data.iloc[i + seq_length + target_horizon - 1, target_idx])

#         if target_col is not None:
#             return np.array(X), np.array(y)
#         else:
#             return np.array(X)

#     def identify_market_regimes(self, returns, n_regimes=3, window=252):
#        """"""
##         Identify market regimes using clustering on volatility and returns
#
##         Args:
##             returns: Series of asset returns
##             n_regimes: Number of regimes to identify
##             window: Rolling window size for regime features
#
##         Returns:
##             DataFrame with regime labels and features
#        """"""
        # Calculate regime features
#         rolling_vol = returns.rolling(window=window).std() * np.sqrt(
#             252
#         )  # Annualized volatility
#         rolling_ret = returns.rolling(window=window).mean() * 252  # Annualized return

        # Combine features
#         regime_features = pd.DataFrame(
#             {"volatility": rolling_vol, "returns": rolling_ret}
#         ).dropna()

        # Normalize features
#         scaler = StandardScaler()
#         normalized_features = scaler.fit_transform(regime_features)

        # Cluster to identify regimes
#         kmeans = KMeans(n_clusters=n_regimes, random_state=42)
#         regime_labels = kmeans.fit_predict(normalized_features)

        # Add labels to features
#         regime_features["regime"] = regime_labels

        # Analyze regimes
#         regime_stats = regime_features.groupby("regime").agg(
#             {"volatility": ["mean", "std"], "returns": ["mean", "std"]}
        )

        # Label regimes based on characteristics
#         regime_names = []
#         for i in range(n_regimes):
#             vol = regime_stats.loc[i, ("volatility", "mean")]
#             ret = regime_stats.loc[i, ("returns", "mean")]

#             if ret > 0 and vol < regime_stats["volatility"]["mean"].median():
#                 name = "Bull (Low Vol)"
#             elif ret > 0 and vol >= regime_stats["volatility"]["mean"].median():
#                 name = "Bull (High Vol)"
#             elif ret <= 0 and vol < regime_stats["volatility"]["mean"].median():
#                 name = "Bear (Low Vol)"
#             else:
#                 name = "Bear (High Vol)"

#             regime_names.append(name)

        # Map numeric labels to names
#         regime_map = {i: name for i, name in enumerate(regime_names)}
#         regime_features["regime_name"] = regime_features["regime"].map(regime_map)

#         return regime_features

#     def plot_regimes(self, prices, regimes):
#        """"""
##         Plot asset prices with identified market regimes
#
##         Args:
##             prices: Series of asset prices
##             regimes: DataFrame with regime information
#
##         Returns:
##             Matplotlib figure
#        """"""
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot prices
#         ax1.plot(prices)
#         ax1.set_title("Asset Prices")
#         ax1.grid(True)

        # Plot regimes
#         for regime in regimes["regime"].unique():
#             regime_data = regimes[regimes["regime"] == regime]
#             regime_name = regime_data["regime_name"].iloc[0]

            # Get color based on regime
#             if "Bull" in regime_name:
#                 color = "green" if "Low Vol" in regime_name else "lime"
#             else:
#                 color = "red" if "High Vol" in regime_name else "salmon"

            # Highlight regime periods
#             for start_idx in range(len(regime_data) - 1):
#                 if regime_data.index[start_idx + 1] == regime_data.index[start_idx] + 1:
#                     continue

#                 ax1.axvspan(
#                     regime_data.index[start_idx],
#                     regime_data.index[start_idx + 1],
#                     alpha=0.2,
#                     color=color,
                )

        # Plot regime features
#         ax2.plot(regimes["volatility"], label="Volatility")
#         ax2.plot(regimes["returns"], label="Returns")
#         ax2.set_title("Regime Features")
#         ax2.grid(True)
#         ax2.legend()

#         plt.tight_layout()
#         return fig
