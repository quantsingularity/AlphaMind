# import pickle

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf


# class MarketSentimentAnalyzer:
#    """"""
##     A sentiment analysis model for financial news and social media data.
##     Uses NLP techniques to extract sentiment signals for trading strategies.
#    """"""

#     def __init__(self, vocab_size=10000, embedding_dim=128, max_length=200):
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.max_length = max_length
#         self.tokenizer = None
#         self.model = self._build_model()

#     def _build_model(self):
#        """Build the sentiment analysis model architecture"""
##         model = tf.keras.Sequential(
#            [
##                 tf.keras.layers.Input(shape=(self.max_length,)),
##                 tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim),
##                 tf.keras.layers.Bidirectional(
##                     tf.keras.layers.LSTM(64, return_sequences=True)
#                ),
##                 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
##                 tf.keras.layers.Dense(64, activation="relu"),
##                 tf.keras.layers.Dropout(0.5),
##                 tf.keras.layers.Dense(
##                     3, activation="softmax"
##                 ),  # 3 classes: negative, neutral, positive
#            ]
#        )
#
##         model.compile(
##             loss="sparse_categorical_crossentropy",
##             optimizer="adam",
##             metrics=["accuracy"],
#        )
#
##         return model
#
##     def prepare_tokenizer(self, texts):
#        """Initialize and fit the tokenizer on the training texts"""
#         self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
#             num_words=self.vocab_size, oov_token="<OOV>"
#         )
#         self.tokenizer.fit_on_texts(texts)

#     def preprocess_text(self, texts):
#        """Convert texts to padded sequences for model input"""
##         if self.tokenizer is None:
##             raise ValueError("Tokenizer not initialized. Call prepare_tokenizer first.")
#
##         sequences = self.tokenizer.texts_to_sequences(texts)
##         padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
##             sequences, maxlen=self.max_length, padding="post", truncating="post"
#        )
##         return padded_sequences
#
##     def train(self, texts, labels, validation_split=0.2, epochs=10, batch_size=32):
#        """Train the sentiment analysis model"""
#         # Prepare tokenizer if not already done
#         if self.tokenizer is None:
#             self.prepare_tokenizer(texts)

#         # Preprocess text data
#         padded_sequences = self.preprocess_text(texts)

#         # Train the model
#         history = self.model.fit(
#             padded_sequences,
#             np.array(labels),
#             validation_split=validation_split,
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=[
#                 tf.keras.callbacks.EarlyStopping(
#                     monitor="val_loss", patience=3, restore_best_weights=True
#                 )
#             ],
#         )

#         return history

#     def predict(self, texts):
#        """Predict sentiment scores for new texts"""
##         padded_sequences = self.preprocess_text(texts)
##         predictions = self.model.predict(padded_sequences)
##         return predictions
#
##     def get_sentiment_score(self, texts):
#        """"""
#         Convert model predictions to a single sentiment score
#         Returns values between -1 (negative) and 1 (positive)
#        """"""
##         predictions = self.predict(texts)
#        # Convert 3-class probabilities to a single score between -1 and 1
#        # [neg_prob, neutral_prob, pos_prob] -> score
##         sentiment_scores = predictions[:, 2] - predictions[:, 0]  # pos_prob - neg_prob
##         return sentiment_scores
#
##     def save(self, model_path, tokenizer_path):
#        """Save the model and tokenizer"""
#         self.model.save(model_path)

#         with open(tokenizer_path, "wb") as handle:
#             pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     def load(self, model_path, tokenizer_path):
#        """Load the model and tokenizer"""
##         self.model = tf.keras.models.load_model(model_path)
#
##         with open(tokenizer_path, "rb") as handle:
##             self.tokenizer = pickle.load(handle)
#
#
## class SentimentBasedStrategy:
#    """"""
#     Trading strategy that incorporates sentiment analysis signals
#     alongside traditional technical indicators.
#    """"""
#
##     def __init__(self, sentiment_analyzer, price_data):
##         self.sentiment_analyzer = sentiment_analyzer
##         self.price_data = price_data
#
##     def calculate_signals(self, news_data, lookback_window=5):
#        """"""
#         Calculate trading signals based on sentiment and price data

#         Args:
#             news_data: DataFrame with 'date' and 'text' columns
#             lookback_window: Number of days to aggregate sentiment

#         Returns:
#             DataFrame with trading signals
#        """"""
#        # Get sentiment scores for news data
##         sentiment_scores = self.sentiment_analyzer.get_sentiment_score(
##             news_data["text"].values
#        )
##         news_data["sentiment_score"] = sentiment_scores
#
#        # Aggregate sentiment by date
##         daily_sentiment = (
##             news_data.groupby("date")["sentiment_score"].mean().reset_index()
#        )
#
#        # Merge with price data
##         merged_data = pd.merge(self.price_data, daily_sentiment, on="date", how="left")
#
#        # Fill missing sentiment values
##         merged_data["sentiment_score"].fillna(0, inplace=True)
#
#        # Calculate rolling average sentiment
##         merged_data["sentiment_ma"] = (
##             merged_data["sentiment_score"].rolling(window=lookback_window).mean()
#        )
#
#        # Generate signals based on sentiment momentum
##         merged_data["sentiment_signal"] = np.where(
##             merged_data["sentiment_ma"] > 0.3,
##             1,  # Strong positive sentiment
##             np.where(
##                 merged_data["sentiment_ma"] < -0.3, -1, 0
##             ),  # Strong negative sentiment
#        )
#
#        # Combine with price-based signals (simple moving average crossover)
##         merged_data["price_ma_short"] = merged_data["close"].rolling(window=5).mean()
##         merged_data["price_ma_long"] = merged_data["close"].rolling(window=20).mean()
##         merged_data["price_signal"] = np.where(
##             merged_data["price_ma_short"] > merged_data["price_ma_long"], 1, -1
#        )
#
#        # Combined signal (equal weight to sentiment and price signals)
##         merged_data["combined_signal"] = (
##             0.5 * merged_data["sentiment_signal"] + 0.5 * merged_data["price_signal"]
#        )
#
#        # Final trading decision
##         merged_data["position"] = np.where(
##             merged_data["combined_signal"] > 0.25,
##             1,  # Long position
##             np.where(merged_data["combined_signal"] < -0.25, -1, 0),  # Short position
#        )
#
##         return merged_data
#
##     def backtest(self, initial_capital=10000):
#        """"""
#         Backtest the sentiment-based strategy

#         Args:
#             initial_capital: Starting capital for the backtest

#         Returns:
#             DataFrame with backtest results and performance metrics
#        """"""
#        # Ensure we have signals calculated
##         if "position" not in self.price_data.columns:
##             raise ValueError("Calculate signals first using calculate_signals()")
#
#        # Copy the data to avoid modifying the original
##         backtest_data = self.price_data.copy()
#
#        # Calculate returns
##         backtest_data["market_return"] = backtest_data["close"].pct_change()
##         backtest_data["strategy_return"] = (
##             backtest_data["position"].shift(1) * backtest_data["market_return"]
#        )
#
#        # Calculate cumulative returns
##         backtest_data["cumulative_market_return"] = (
##             1 + backtest_data["market_return"]
##         ).cumprod()
##         backtest_data["cumulative_strategy_return"] = (
##             1 + backtest_data["strategy_return"]
##         ).cumprod()
#
#        # Calculate equity curves
##         backtest_data["market_equity"] = (
##             initial_capital * backtest_data["cumulative_market_return"]
#        )
##         backtest_data["strategy_equity"] = (
##             initial_capital * backtest_data["cumulative_strategy_return"]
#        )
#
#        # Calculate drawdowns
##         backtest_data["market_peak"] = backtest_data["market_equity"].cummax()
##         backtest_data["strategy_peak"] = backtest_data["strategy_equity"].cummax()
##         backtest_data["market_drawdown"] = (
##             backtest_data["market_equity"] - backtest_data["market_peak"]
##         ) / backtest_data["market_peak"]
##         backtest_data["strategy_drawdown"] = (
##             backtest_data["strategy_equity"] - backtest_data["strategy_peak"]
##         ) / backtest_data["strategy_peak"]
#
#        # Calculate performance metrics
##         total_return = backtest_data["cumulative_strategy_return"].iloc[-1] - 1
##         annual_return = (1 + total_return) ** (252 / len(backtest_data)) - 1
##         annual_volatility = backtest_data["strategy_return"].std() * np.sqrt(252)
##         sharpe_ratio = (
##             annual_return / annual_volatility if annual_volatility != 0 else 0
#        )
##         max_drawdown = backtest_data["strategy_drawdown"].min()
#
#        # Store metrics
##         self.performance_metrics = {
#            "total_return": total_return,
#            "annual_return": annual_return,
#            "annual_volatility": annual_volatility,
#            "sharpe_ratio": sharpe_ratio,
#            "max_drawdown": max_drawdown,
#        }
#
##         return backtest_data
#
##     def plot_results(self, backtest_data):
#        """Plot backtest results"""
#         plt.figure(figsize=(14, 10))

#         # Plot equity curves
#         plt.subplot(3, 1, 1)
#         plt.plot(backtest_data["date"], backtest_data["market_equity"], label="Market")
#         plt.plot(
#             backtest_data["date"], backtest_data["strategy_equity"], label="Strategy"
#         )
#         plt.title("Equity Curves")
#         plt.legend()
#         plt.grid(True)

#         # Plot drawdowns
#         plt.subplot(3, 1, 2)
#         plt.plot(
#             backtest_data["date"], backtest_data["market_drawdown"], label="Market"
#         )
#         plt.plot(
#             backtest_data["date"], backtest_data["strategy_drawdown"], label="Strategy"
#         )
#         plt.title("Drawdowns")
#         plt.legend()
#         plt.grid(True)

#         # Plot sentiment and positions
#         plt.subplot(3, 1, 3)
#         plt.plot(
#             backtest_data["date"], backtest_data["sentiment_ma"], label="Sentiment MA"
#         )
#         plt.plot(backtest_data["date"], backtest_data["position"], label="Position")
#         plt.title("Sentiment and Positions")
#         plt.legend()
#         plt.grid(True)

#         plt.tight_layout()
#         return plt.gcf()
