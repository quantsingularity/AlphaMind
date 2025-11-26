# import os
# import pickle
# import sys

# import numpy as np
# import pandas as pd
# import pytest
# import tensorflow as tf

# Add the backend directory to the path
# sys.path.append("/home/ubuntu/alphamind_project/backend")

# Corrected imports based on project structure
# from models.attention_mechanism import FinancialTimeSeriesTransformer # Incorrect path
# from models.portfolio_optimization import PortfolioOptimizer # Incorrect path
# from alternative_data.sentiment_analysis import (
#     MarketSentimentAnalyzer,
#     SentimentBasedStrategy,

# Assuming these imports are correct based on the project structure
# from infrastructure.authentication import AuthenticationSystem

# Constants for testing
# VOCAB_SIZE = 500
# EMBEDDING_DIM = 16
# MAX_LENGTH = 20
# MODEL_SAVE_DIR = "/home/ubuntu/alphamind_tests/backend/saved_models"
# TOKENIZER_SAVE_DIR = "/home/ubuntu/alphamind_tests/backend/saved_tokenizers"

# Create directories if they don't exist
# os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# os.makedirs(TOKENIZER_SAVE_DIR, exist_ok=True)


# @pytest.fixture
# def analyzer():
#    """Fixture for a MarketSentimentAnalyzer instance."""
##     return MarketSentimentAnalyzer(
##         vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_length=MAX_LENGTH
#    )
#
#
## @pytest.fixture
## def sample_texts():
#    """Fixture for sample text data."""
#     return [
#         "Stock market surges on positive economic news",
#         "Company reports record profits, shares jump",
#         "Market dips amid uncertainty",
#         "Negative outlook for the tech sector",
#         "Neutral report from the central bank",
#     ]


# @pytest.fixture
# def sample_labels():
#    """Fixture for sample labels (0: neg, 1: neu, 2: pos)."""
##     return [2, 2, 0, 0, 1]
#
#
## def test_analyzer_init(analyzer):
#    """Test MarketSentimentAnalyzer initialization."""
#     assert analyzer.vocab_size == VOCAB_SIZE
#     assert analyzer.embedding_dim == EMBEDDING_DIM
#     assert analyzer.max_length == MAX_LENGTH
#     assert isinstance(analyzer.model, tf.keras.Model)
#     assert analyzer.tokenizer is None


# def test_build_model_structure(analyzer):
#    """Test the structure of the sentiment analysis model."""
##     model = analyzer.model
##     assert len(model.layers) > 4  # Check for multiple layers
##     assert isinstance(model.layers[0], tf.keras.layers.Embedding)
##     assert model.layers[0].input_dim == VOCAB_SIZE
##     assert model.layers[0].output_dim == EMBEDDING_DIM
##     assert model.output_shape == (None, 3)  # 3 classes
#
#
## def test_prepare_tokenizer(analyzer, sample_texts):
#    """Test tokenizer preparation."""
#     analyzer.prepare_tokenizer(sample_texts)
#     assert analyzer.tokenizer is not None
#     assert isinstance(analyzer.tokenizer, tf.keras.preprocessing.text.Tokenizer)
#     assert analyzer.tokenizer.num_words == VOCAB_SIZE
#     assert len(analyzer.tokenizer.word_index) > 5  # Check if words were indexed


# def test_preprocess_text(analyzer, sample_texts):
#    """Test text preprocessing."""
##     analyzer.prepare_tokenizer(sample_texts)
##     padded_sequences = analyzer.preprocess_text(sample_texts)
##     assert isinstance(padded_sequences, np.ndarray)
##     assert padded_sequences.shape == (len(sample_texts), MAX_LENGTH)
##     assert padded_sequences.dtype == np.int32
#
#
## def test_preprocess_text_no_tokenizer(analyzer, sample_texts):
#    """Test calling preprocess_text before tokenizer is ready."""
#     with pytest.raises(ValueError, match="Tokenizer not initialized"):
#         analyzer.preprocess_text(sample_texts)


# def test_predict_output_shape(analyzer, sample_texts):
#    """Test the shape of the predict method output."""
##     analyzer.prepare_tokenizer(sample_texts)
#    # Mock model predict to avoid actual prediction
##     analyzer.model.predict = lambda x: np.random.rand(len(x), 3)
##     predictions = analyzer.predict(sample_texts)
##     assert predictions.shape == (len(sample_texts), 3)
#
#
## def test_get_sentiment_score(analyzer, sample_texts):
#    """Test the output of get_sentiment_score."""
#     analyzer.prepare_tokenizer(sample_texts)
#     # Mock model predict with known probabilities
#     mock_predictions = np.array(
#         [
#             [0.1, 0.1, 0.8],  # Positive
#             [0.7, 0.2, 0.1],  # Negative
#             [0.2, 0.6, 0.2],  # Neutral
#         ]
#     analyzer.model.predict = lambda x: mock_predictions[: len(x)]

#     scores = analyzer.get_sentiment_score(sample_texts[:3])
#     assert scores.shape == (3,)
#     assert np.isclose(scores[0], 0.7)  # 0.8 - 0.1
#     assert np.isclose(scores[1], -0.6)  # 0.1 - 0.7
#     assert np.isclose(scores[2], 0.0)  # 0.2 - 0.2


# def test_save_load_analyzer(analyzer, sample_texts, tmp_path):
#    """Test saving and loading the analyzer model and tokenizer."""
#    # Use .keras extension for saving the model
##     model_path = os.path.join(str(tmp_path), "sentiment_model.keras")
##     tokenizer_path = os.path.join(str(tmp_path), "tokenizer.pkl")
#
##     analyzer.prepare_tokenizer(sample_texts)
#    # Ensure the model is built before compiling and saving
##     dummy_input = np.zeros((1, MAX_LENGTH), dtype=np.int32)
##     _ = analyzer.model(dummy_input)  # Build the model by calling it
#    # Corrected compile line - no escaped quotes
##     analyzer.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
#
##     analyzer.save(model_path, tokenizer_path)
#
##     assert os.path.exists(model_path)
##     assert os.path.exists(tokenizer_path)
#
##     new_analyzer = MarketSentimentAnalyzer(
##         vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_length=MAX_LENGTH
#    )
#    # Rebuild the model in the new instance before loading weights
##     _ = new_analyzer.model(dummy_input)
##     new_analyzer.load(model_path, tokenizer_path)
#
##     assert isinstance(new_analyzer.model, tf.keras.Model)
##     assert new_analyzer.tokenizer is not None
##     assert isinstance(new_analyzer.tokenizer, tf.keras.preprocessing.text.Tokenizer)
##     assert new_analyzer.tokenizer.num_words == VOCAB_SIZE
#
#
## --- Tests for SentimentBasedStrategy ---
#
#
## @pytest.fixture
## def sample_price_data():
#    """Fixture for sample price data DataFrame."""
#     # Ensure date column is datetime64[ns]
#     dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=30, freq="D"))
#     prices = np.random.rand(30) * 10 + 100
#     # Set date as index for consistency
#     df = pd.DataFrame({"date": dates, "close": prices})
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.set_index("date")
#     return df


# @pytest.fixture
# def sample_news_data():
#    """Fixture for sample news data DataFrame."""
#    # Ensure date column is datetime64[ns] and use 'h' for frequency
##     dates = pd.to_datetime(
##         pd.date_range(start="2023-01-01", periods=50, freq="12h")
##     )  # Use 'h' instead of 'H'
##     texts = [f"Sample news text {i}" for i in range(50)]
#    # Ensure date column is datetime for merging
##     df = pd.DataFrame({"date": dates, "text": texts})
##     df["date"] = pd.to_datetime(df["date"])
##     return df
#
#
## @pytest.fixture
## def strategy(analyzer, sample_price_data):
#    """Fixture for a SentimentBasedStrategy instance."""
#     # Mock analyzer methods for strategy testing
#     analyzer.get_sentiment_score = (
#         lambda texts: np.random.rand(len(texts)) * 2 - 1
#     )  # Random scores [-1, 1]
#     return SentimentBasedStrategy(
#         analyzer, sample_price_data.copy()
#     )  # Use a copy to avoid modifying fixture


# def test_strategy_init(strategy, analyzer, sample_price_data):
#    """Test SentimentBasedStrategy initialization."""
##     assert strategy.sentiment_analyzer == analyzer
#    # Compare after resetting index if necessary, or ensure index consistency
##     pd.testing.assert_frame_equal(
##         strategy.price_data.reset_index(drop=False),
##         sample_price_data.reset_index(drop=False),
#    )
#
#
## def test_calculate_signals(strategy, sample_news_data):
#    """Test the calculate_signals method."""
#     # Ensure price_data has a datetime index
#     if not isinstance(strategy.price_data.index, pd.DatetimeIndex):
#         strategy.price_data["date"] = pd.to_datetime(strategy.price_data["date"])
#         strategy.price_data = strategy.price_data.set_index("date")

#     # Ensure news_data date column is datetime
#     sample_news_data["date"] = pd.to_datetime(sample_news_data["date"])

#     signals_df = strategy.calculate_signals(sample_news_data, lookback_window=5)

#     assert isinstance(signals_df, pd.DataFrame)
#     assert len(signals_df) == len(strategy.price_data)  # Should match price data length
#     assert "sentiment_score" in signals_df.columns
#     assert "sentiment_ma" in signals_df.columns
#     assert "sentiment_signal" in signals_df.columns
#     assert "price_ma_short" in signals_df.columns
#     assert "price_ma_long" in signals_df.columns
#     assert "price_signal" in signals_df.columns
#     assert "combined_signal" in signals_df.columns
#     assert "position" in signals_df.columns
#     # Check if NaNs are handled (e.g., in rolling means)
#     # The first few rows will have NaNs due to rolling calculations
#     assert not signals_df["sentiment_ma"].iloc[5:].isnull().any()
#     assert not signals_df["position"].isnull().any()


# def test_backtest_structure(strategy, sample_news_data):
#    """Test the structure of the backtest output."""
#    # Ensure price_data has a datetime index
##     if not isinstance(strategy.price_data.index, pd.DatetimeIndex):
##         strategy.price_data["date"] = pd.to_datetime(strategy.price_data["date"])
##         strategy.price_data = strategy.price_data.set_index("date")
#
#    # Ensure news_data date column is datetime
##     sample_news_data["date"] = pd.to_datetime(sample_news_data["date"])
#
#    # Calculate signals first
##     strategy.price_data = strategy.calculate_signals(sample_news_data)
#
##     backtest_results = strategy.backtest()
#
##     assert isinstance(backtest_results, pd.DataFrame)
##     assert len(backtest_results) == len(strategy.price_data)
##     assert "market_return" in backtest_results.columns
##     assert "strategy_return" in backtest_results.columns
##     assert "cumulative_market_return" in backtest_results.columns
##     assert "cumulative_strategy_return" in backtest_results.columns
##     assert "market_equity" in backtest_results.columns
##     assert "strategy_equity" in backtest_results.columns
##     assert "market_drawdown" in backtest_results.columns
##     assert "strategy_drawdown" in backtest_results.columns
#
#    # Check performance metrics
##     assert hasattr(strategy, "performance_metrics")
##     assert isinstance(strategy.performance_metrics, dict)
##     assert "total_return" in strategy.performance_metrics
##     assert "sharpe_ratio" in strategy.performance_metrics
##     assert "max_drawdown" in strategy.performance_metrics
#
#
## def test_backtest_no_signals(strategy):
#    """Test calling backtest before calculating signals."""
#     with pytest.raises(ValueError, match="Calculate signals first"):
#         strategy.backtest()
