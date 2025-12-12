from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class DataProcessor:
    """Advanced data processing pipeline for financial time series data.
    Includes data cleaning, feature engineering, and anomaly detection.
    """

    def __init__(self) -> Any:
        self.scalers = {}
        self.pca_models = {}
        self.anomaly_detector = None

    def clean_data(
        self, data: Any, fill_method: Any = "ffill", outlier_std_threshold: Any = 3.0
    ) -> Any:
        """Clean financial data by handling missing values and outliers

        Args:
            data: DataFrame containing financial time series data
            fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
            outlier_std_threshold: Threshold (in standard deviations) to identify outliers

        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        if fill_method == "ffill":
            cleaned_data = cleaned_data.fillna(method="ffill")
            cleaned_data = cleaned_data.fillna(method="bfill")
        elif fill_method == "bfill":
            cleaned_data = cleaned_data.fillna(method="bfill")
            cleaned_data = cleaned_data.fillna(method="ffill")
        elif fill_method == "interpolate":
            cleaned_data = cleaned_data.interpolate(method="time")
            cleaned_data = cleaned_data.fillna(method="ffill").fillna(method="bfill")
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_mean = cleaned_data[col].mean()
            col_std = cleaned_data[col].std()
            upper_bound = col_mean + outlier_std_threshold * col_std
            lower_bound = col_mean - outlier_std_threshold * col_std
            cleaned_data[col] = np.where(
                cleaned_data[col] > upper_bound,
                upper_bound,
                np.where(
                    cleaned_data[col] < lower_bound, lower_bound, cleaned_data[col]
                ),
            )
        return cleaned_data

    def engineer_features(
        self, data: Any, window_sizes: Any = [5, 10, 20, 50], include_ta: Any = True
    ) -> Any:
        """Engineer features from financial time series data

        Args:
            data: DataFrame with at least OHLCV columns (open, high, low, close, volume)
            window_sizes: List of window sizes for rolling calculations
            include_ta: Whether to include technical indicators

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for feature engineering: {missing_cols}"
            )
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["daily_range"] = df["high"] - df["low"]
        df["daily_range_pct"] = df["daily_range"] / df["close"]
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ma"] = df["volume"].rolling(window=20).mean()
        df["relative_volume"] = df["volume"] / df["volume_ma"]
        for window in window_sizes:
            df[f"return_{window}d"] = df["close"].pct_change(window)
            df[f"volatility_{window}d"] = df["returns"].rolling(window=window).std()
            df[f"ma_{window}d"] = df["close"].rolling(window=window).mean()
            df[f"close_to_ma_{window}d"] = df["close"] / df[f"ma_{window}d"]
            df[f"volume_ma_{window}d"] = df["volume"].rolling(window=window).mean()
        if include_ta:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df["rsi_14"] = 100 - 100 / (1 + rs)
            df["rsi_14"] = df["rsi_14"].fillna(50)
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            df["bb_std"] = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
            df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift(1))
            tr3 = abs(df["low"] - df["close"].shift(1))
            df["true_range"] = pd.DataFrame([tr1, tr2, tr3]).max()
            df["atr_14"] = df["true_range"].rolling(window=14).mean()
        df = df.dropna()
        return df

    def normalize_data(
        self,
        data: Any,
        method: Any = "standard",
        fit: Any = True,
        group_name: Any = "default",
    ) -> Any:
        """Normalize data using various scaling methods

        Args:
            data: DataFrame or array to normalize
            method: Normalization method ('standard', 'minmax', 'robust')
            fit: Whether to fit a new scaler or use a previously fit one
            group_name: Identifier for the scaler group

        Returns:
            Normalized data
        """
        if method == "standard":
            ScalerClass = StandardScaler
        elif method == "minmax":
            ScalerClass = MinMaxScaler
        elif method == "robust":
            ScalerClass = RobustScaler
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        if fit or group_name not in self.scalers:
            self.scalers[group_name] = ScalerClass()
            return self.scalers[group_name].fit_transform(data)
        else:
            return self.scalers[group_name].transform(data)

    def reduce_dimensions(
        self,
        data: Any,
        n_components: Any = 0.95,
        fit: Any = True,
        group_name: Any = "default",
    ) -> Any:
        """Reduce dimensionality of data using PCA

        Args:
            data: DataFrame or array to reduce
            n_components: Number of components or variance ratio to preserve
            fit: Whether to fit a new PCA model or use a previously fit one
            group_name: Identifier for the PCA model group

        Returns:
            Reduced data
        """
        if fit or group_name not in self.pca_models:
            self.pca_models[group_name] = PCA(n_components=n_components)
            return self.pca_models[group_name].fit_transform(data)
        else:
            return self.pca_models[group_name].transform(data)

    def detect_anomalies(
        self, data: Any, contamination: Any = 0.05, fit: Any = True
    ) -> Any:
        """Detect anomalies in financial data using Isolation Forest

        Args:
            data: DataFrame or array to analyze
            contamination: Expected proportion of anomalies
            fit: Whether to fit a new anomaly detector or use a previously fit one

        Returns:
            Boolean array indicating anomalies (True) and normal points (False)
        """
        if fit or self.anomaly_detector is None:
            self.anomaly_detector = IsolationForest(
                contamination=contamination, random_state=42
            )
            self.anomaly_detector.fit(data)
        predictions = self.anomaly_detector.predict(data)
        return predictions == -1

    def create_sequences(
        self,
        data: Any,
        seq_length: Any,
        target_col: Any = None,
        target_horizon: Any = 1,
    ) -> Any:
        """Create sequences for time series modeling (e.g., RNN/LSTM)

        Args:
            data: DataFrame or array of features
            seq_length: Length of input sequences
            target_col: Target column name or index (if None, creates unsupervised sequences)
            target_horizon: Forecast horizon for the target (e.g., 1 for next day)

        Returns:
            X: Input sequences (numpy array)
            y: Target values (numpy array, if target_col is provided)
        """
        X = []
        y = [] if target_col is not None else None
        end_idx = (
            len(data)
            - seq_length
            - (target_horizon - 1 if target_col is not None else 0)
        )
        for i in range(end_idx):
            X.append(
                data.iloc[i : i + seq_length].values
                if isinstance(data, pd.DataFrame)
                else data[i : i + seq_length]
            )
            if target_col is not None:
                target_idx = target_col
                if isinstance(target_col, str):
                    target_idx = data.columns.get_loc(target_col)
                y.append(
                    data.iloc[i + seq_length + target_horizon - 1, target_idx]
                    if isinstance(data, pd.DataFrame)
                    else data[i + seq_length + target_horizon - 1, target_idx]
                )
        if target_col is not None:
            return (np.array(X), np.array(y))
        else:
            return np.array(X)

    def identify_market_regimes(
        self, returns: Any, n_regimes: Any = 3, window: Any = 252
    ) -> Any:
        """Identify market regimes using K-Means clustering on rolling volatility and returns

        Args:
            returns: Series of asset returns
            n_regimes: Number of regimes to identify
            window: Rolling window size for regime features (e.g., 252 for 1 year)

        Returns:
            DataFrame with regime labels and features
        """
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_ret = returns.rolling(window=window).mean() * 252
        regime_features = pd.DataFrame(
            {"volatility": rolling_vol, "returns": rolling_ret}
        ).dropna()
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(regime_features)
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init="auto")
        regime_labels = kmeans.fit_predict(normalized_features)
        regime_features["regime"] = regime_labels
        regime_stats = regime_features.groupby("regime").agg(
            {"volatility": ["mean", "std"], "returns": ["mean", "std"]}
        )
        vol_median = regime_stats["volatility"]["mean"].median()
        regime_names = []
        for i in range(n_regimes):
            vol = regime_stats.loc[i, ("volatility", "mean")]
            ret = regime_stats.loc[i, ("returns", "mean")]
            if ret > 0:
                ret_label = "Bull"
            else:
                ret_label = "Bear"
            if vol < vol_median:
                vol_label = "Low Vol"
            else:
                vol_label = "High Vol"
            name = f"{ret_label} ({vol_label})"
            regime_names.append(name)
        regime_map = {i: name for i, name in enumerate(regime_names)}
        regime_features["regime_name"] = regime_features["regime"].map(regime_map)
        return regime_features

    def plot_regimes(self, prices: Any, regimes: Any) -> Any:
        """Plot asset prices with identified market regimes

        Args:
            prices: Series of asset prices
            regimes: DataFrame with regime information

        Returns:
            Matplotlib figure
        """
        aligned_df = pd.DataFrame({"price": prices}).join(regimes, how="inner").dropna()
        prices_aligned = aligned_df["price"]
        regimes_aligned = aligned_df["regime_name"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        color_map = {
            "Bull (Low Vol)": "green",
            "Bull (High Vol)": "lime",
            "Bear (Low Vol)": "salmon",
            "Bear (High Vol)": "red",
        }
        ax1.plot(prices_aligned, label="Price", color="black", linewidth=1)
        ax1.set_title("Asset Prices with Market Regimes")
        ax1.grid(True, linestyle="--", alpha=0.6)
        unique_regimes = regimes_aligned.unique()
        handles = []
        for regime_name in unique_regimes:
            color = color_map.get(regime_name, "gray")
            is_regime = regimes_aligned == regime_name
            starts = is_regime.shift(1, fill_value=False) != is_regime
            ends = is_regime.shift(-1, fill_value=False) != is_regime
            for start, end in zip(
                regimes_aligned.index[starts], regimes_aligned.index[ends]
            ):
                ax1.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color=color,
                    label=(
                        regime_name
                        if regime_name not in [h.get_label() for h in handles]
                        else "_nolegend_"
                    ),
                )
            if regime_name not in [h.get_label() for h in handles]:
                handles.append(
                    ax1.axvspan(0, 0, alpha=0.2, color=color, label=regime_name)
                )
        ax1.legend(loc="upper left")
        ax2.plot(
            aligned_df["volatility"],
            label="Annualized Volatility",
            color="blue",
            linewidth=1.5,
        )
        ax2.plot(
            aligned_df["returns"],
            label="Annualized Returns",
            color="orange",
            linewidth=1.5,
        )
        ax2.set_title("Regime Defining Features")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()
        plt.tight_layout()
        return fig
