import pickle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class MarketSentimentAnalyzer:
    """
    A sentiment analysis model for financial news and social media data.
    Uses NLP techniques (Bi-LSTM) to extract sentiment signals for trading strategies.
    """

    def __init__(
        self, vocab_size: Any = 10000, embedding_dim: Any = 128, max_length: Any = 200
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = None
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build the Bi-LSTM sentiment analysis model architecture"""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.max_length,)),
                tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(64, return_sequences=True)
                ),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model

    def prepare_tokenizer(self, texts: List[str]) -> Any:
        """Initialize and fit the tokenizer on the training texts"""
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size, oov_token="<OOV>"
        )
        self.tokenizer.fit_on_texts(texts)

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        """Convert texts to padded sequences for model input"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_tokenizer first.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding="post", truncating="post"
        )
        return padded_sequences

    def train(
        self,
        texts: List[str],
        labels: List[int],
        validation_split: Any = 0.2,
        epochs: Any = 10,
        batch_size: Any = 32,
    ) -> tf.keras.callbacks.History:
        """Train the sentiment analysis model"""
        if self.tokenizer is None:
            self.prepare_tokenizer(texts)
        padded_sequences = self.preprocess_text(texts)
        history = self.model.fit(
            padded_sequences,
            np.array(labels),
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                )
            ],
        )
        return history

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict sentiment probabilities for new texts"""
        padded_sequences = self.preprocess_text(texts)
        predictions = self.model.predict(padded_sequences)
        return predictions

    def get_sentiment_score(self, texts: List[str]) -> np.ndarray:
        """
        Convert model predictions to a single, continuous sentiment score.
        Returns values between -1 (negative) and 1 (positive).
        """
        predictions = self.predict(texts)
        sentiment_scores = predictions[:, 2] - predictions[:, 0]
        return sentiment_scores

    def save(self, model_path: str, tokenizer_path: str) -> Any:
        """Save the model and tokenizer"""
        self.model.save(model_path)
        with open(tokenizer_path, "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, model_path: str, tokenizer_path: str) -> Any:
        """Load the model and tokenizer"""
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)


class SentimentBasedStrategy:
    """
    Trading strategy that incorporates sentiment analysis signals
    alongside traditional technical indicators.
    """

    def __init__(
        self, sentiment_analyzer: MarketSentimentAnalyzer, price_data: pd.DataFrame
    ) -> None:
        self.sentiment_analyzer = sentiment_analyzer
        self.price_data = price_data.copy()
        self.performance_metrics: Dict[str, float] = {}

    def calculate_signals(
        self, news_data: pd.DataFrame, lookback_window: Any = 5
    ) -> pd.DataFrame:
        """
        Calculate trading signals based on sentiment and price data.

        Args:
            news_data: DataFrame with 'date' (datetime) and 'text' columns.
            lookback_window: Number of days to aggregate sentiment (for MA).

        Returns:
            DataFrame with trading signals (position, combined_signal, etc.).
        """
        sentiment_scores = self.sentiment_analyzer.get_sentiment_score(
            news_data["text"].values
        )
        news_data["sentiment_score"] = sentiment_scores
        daily_sentiment = (
            news_data.groupby("date")["sentiment_score"].mean().reset_index()
        )
        merged_data = pd.merge(self.price_data, daily_sentiment, on="date", how="left")
        merged_data["sentiment_score"].fillna(0, inplace=True)
        merged_data["sentiment_ma"] = (
            merged_data["sentiment_score"].rolling(window=lookback_window).mean()
        )
        merged_data["sentiment_signal"] = np.where(
            merged_data["sentiment_ma"] > 0.3,
            1,
            np.where(merged_data["sentiment_ma"] < -0.3, -1, 0),
        )
        merged_data["price_ma_short"] = merged_data["close"].rolling(window=5).mean()
        merged_data["price_ma_long"] = merged_data["close"].rolling(window=20).mean()
        merged_data["price_signal"] = np.where(
            merged_data["price_ma_short"] > merged_data["price_ma_long"], 1, -1
        )
        merged_data["combined_signal"] = (
            0.5 * merged_data["sentiment_signal"] + 0.5 * merged_data["price_signal"]
        )
        merged_data["position"] = np.where(
            merged_data["combined_signal"] > 0.25,
            1,
            np.where(merged_data["combined_signal"] < -0.25, -1, 0),
        )
        self.price_data = merged_data.dropna()
        return self.price_data

    def backtest(self, initial_capital: Any = 10000) -> pd.DataFrame:
        """
        Backtest the sentiment-based strategy and calculate performance metrics.

        Args:
            initial_capital: Starting capital for the backtest.

        Returns:
            DataFrame with backtest results, equity curves, and drawdowns.
        """
        if "position" not in self.price_data.columns:
            raise ValueError("Calculate signals first using calculate_signals()")
        backtest_data = self.price_data.copy()
        backtest_data["market_return"] = backtest_data["close"].pct_change()
        backtest_data["strategy_return"] = (
            backtest_data["position"].shift(1) * backtest_data["market_return"]
        )
        backtest_data.fillna(0, inplace=True)
        backtest_data["cumulative_market_return"] = (
            1 + backtest_data["market_return"]
        ).cumprod()
        backtest_data["cumulative_strategy_return"] = (
            1 + backtest_data["strategy_return"]
        ).cumprod()
        backtest_data["market_equity"] = (
            initial_capital * backtest_data["cumulative_market_return"]
        )
        backtest_data["strategy_equity"] = (
            initial_capital * backtest_data["cumulative_strategy_return"]
        )
        backtest_data["market_peak"] = backtest_data["market_equity"].cummax()
        backtest_data["strategy_peak"] = backtest_data["strategy_equity"].cummax()
        backtest_data["market_drawdown"] = (
            backtest_data["market_equity"] - backtest_data["market_peak"]
        ) / backtest_data["market_peak"]
        backtest_data["strategy_drawdown"] = (
            backtest_data["strategy_equity"] - backtest_data["strategy_peak"]
        ) / backtest_data["strategy_peak"]
        trading_days = len(backtest_data)
        annualization_factor = 252
        total_return = backtest_data["cumulative_strategy_return"].iloc[-1] - 1
        annual_return = (1 + total_return) ** (annualization_factor / trading_days) - 1
        annual_volatility = backtest_data["strategy_return"].std() * np.sqrt(
            annualization_factor
        )
        sharpe_ratio = (
            annual_return / annual_volatility if annual_volatility != 0 else 0
        )
        max_drawdown = backtest_data["strategy_drawdown"].min()
        self.performance_metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }
        return backtest_data

    def plot_results(self, backtest_data: pd.DataFrame) -> plt.Figure:
        """Plot backtest results (Equity, Drawdown, Signals)"""
        plt.figure(figsize=(14, 10))
        plt.subplot(3, 1, 1)
        plt.plot(
            backtest_data["date"],
            backtest_data["market_equity"],
            label="Market",
            color="gray",
        )
        plt.plot(
            backtest_data["date"],
            backtest_data["strategy_equity"],
            label="Strategy",
            color="green",
        )
        plt.title("Equity Curves (Strategy vs. Market)")
        plt.xlabel("Date")
        plt.ylabel("Capital ($)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.subplot(3, 1, 2)
        plt.plot(
            backtest_data["date"],
            backtest_data["strategy_drawdown"],
            label="Strategy Drawdown",
            color="red",
        )
        plt.title("Drawdowns (Strategy)")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.subplot(3, 1, 3)
        plt.plot(
            backtest_data["date"],
            backtest_data["sentiment_ma"],
            label="Sentiment MA",
            color="blue",
            alpha=0.7,
        )
        plt.step(
            backtest_data["date"],
            backtest_data["position"],
            where="post",
            label="Position (1=Long, -1=Short)",
            color="purple",
        )
        plt.axhline(0.3, color="r", linestyle=":", alpha=0.5)
        plt.axhline(-0.3, color="r", linestyle=":", alpha=0.5)
        plt.title("Sentiment Momentum and Trading Positions")
        plt.xlabel("Date")
        plt.ylabel("Sentiment Score / Position")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        return plt.gcf()
