from typing import Any
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from alternative_data.sentiment_analysis import (
    MarketSentimentAnalyzer,
    SentimentBasedStrategy,
)

VOCAB_SIZE = 500
EMBEDDING_DIM = 16
MAX_LENGTH = 20


@pytest.fixture
def analyzer() -> Any:
    """Fixture for a MarketSentimentAnalyzer instance."""
    return MarketSentimentAnalyzer(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_length=MAX_LENGTH
    )


@pytest.fixture
def sample_texts() -> Any:
    """Fixture for sample text data."""
    return [
        "Stock market surges on positive economic news",
        "Company reports record profits, shares jump",
        "Market dips amid uncertainty",
        "Negative outlook for the tech sector",
        "Neutral report from the central bank",
    ]


@pytest.fixture
def sample_labels() -> Any:
    """Fixture for sample labels (0: neg, 1: neu, 2: pos)."""
    return [2, 2, 0, 0, 1]


def test_analyzer_init(analyzer: Any) -> Any:
    """Test MarketSentimentAnalyzer initialization."""
    assert analyzer.vocab_size == VOCAB_SIZE
    assert analyzer.embedding_dim == EMBEDDING_DIM
    assert analyzer.max_length == MAX_LENGTH
    assert isinstance(analyzer.model, tf.keras.Model)
    assert analyzer.tokenizer is None


def test_build_model_structure(analyzer: Any) -> Any:
    """Test the structure of the sentiment analysis model."""
    model = analyzer.model
    assert len(model.layers) > 4
    assert isinstance(model.layers[0], tf.keras.layers.Embedding)
    assert model.layers[0].input_dim == VOCAB_SIZE
    assert model.layers[0].output_dim == EMBEDDING_DIM
    assert model.output_shape == (None, 3)


def test_prepare_tokenizer(analyzer: Any, sample_texts: Any) -> Any:
    """Test tokenizer preparation."""
    analyzer.prepare_tokenizer(sample_texts)
    assert analyzer.tokenizer is not None
    assert isinstance(analyzer.tokenizer, tf.keras.preprocessing.text.Tokenizer)
    assert analyzer.tokenizer.num_words == VOCAB_SIZE
    assert len(analyzer.tokenizer.word_index) > 5


def test_preprocess_text(analyzer: Any, sample_texts: Any) -> Any:
    """Test text preprocessing."""
    analyzer.prepare_tokenizer(sample_texts)
    padded_sequences = analyzer.preprocess_text(sample_texts)
    assert isinstance(padded_sequences, np.ndarray)
    assert padded_sequences.shape == (len(sample_texts), MAX_LENGTH)
    assert padded_sequences.dtype == np.int32


def test_preprocess_text_no_tokenizer(analyzer: Any, sample_texts: Any) -> Any:
    """Test calling preprocess_text before tokenizer is ready."""
    with pytest.raises(ValueError, match="Tokenizer not initialized"):
        analyzer.preprocess_text(sample_texts)


def test_predict_output_shape(analyzer: Any, sample_texts: Any) -> Any:
    """Test the shape of the predict method output."""
    analyzer.prepare_tokenizer(sample_texts)
    analyzer.model.predict = lambda x: np.random.rand(len(x), 3)
    predictions = analyzer.predict(sample_texts)
    assert predictions.shape == (len(sample_texts), 3)


def test_get_sentiment_score(analyzer: Any, sample_texts: Any) -> Any:
    """Test the output of get_sentiment_score."""
    analyzer.prepare_tokenizer(sample_texts)
    mock_predictions = np.array([[0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.2, 0.6, 0.2]])
    analyzer.model.predict = lambda x: mock_predictions[: len(x)]
    scores = analyzer.get_sentiment_score(sample_texts[:3])
    assert scores.shape == (3,)
    assert np.isclose(scores[0], 0.7)
    assert np.isclose(scores[1], -0.6)
    assert np.isclose(scores[2], 0.0)


@pytest.fixture
def sample_price_data() -> Any:
    """Fixture for sample price data DataFrame."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=30, freq="D"))
    prices = np.random.rand(30) * 10 + 100
    df = pd.DataFrame({"date": dates, "close": prices})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


@pytest.fixture
def sample_news_data() -> Any:
    """Fixture for sample news data DataFrame."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=50, freq="12h"))
    texts = [f"Sample news text {i}" for i in range(50)]
    df = pd.DataFrame({"date": dates, "text": texts})
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def strategy(analyzer: Any, sample_price_data: Any) -> Any:
    """Fixture for a SentimentBasedStrategy instance."""
    analyzer.get_sentiment_score = lambda texts: np.random.rand(len(texts)) * 2 - 1
    return SentimentBasedStrategy(analyzer, sample_price_data.copy())


def test_strategy_init(strategy: Any, analyzer: Any, sample_price_data: Any) -> Any:
    """Test SentimentBasedStrategy initialization."""
    assert strategy.sentiment_analyzer == analyzer
    pd.testing.assert_frame_equal(
        strategy.price_data.reset_index(drop=False),
        sample_price_data.reset_index(drop=False),
    )


def test_calculate_signals(strategy: Any, sample_news_data: Any) -> Any:
    """Test the calculate_signals method."""
    if not isinstance(strategy.price_data.index, pd.DatetimeIndex):
        strategy.price_data["date"] = pd.to_datetime(strategy.price_data["date"])
        strategy.price_data = strategy.price_data.set_index("date")
    sample_news_data["date"] = pd.to_datetime(sample_news_data["date"])
    signals_df = strategy.calculate_signals(sample_news_data, lookback_window=5)
    assert isinstance(signals_df, pd.DataFrame)
    assert len(signals_df) == len(strategy.price_data)
    required_cols = [
        "sentiment_score",
        "sentiment_ma",
        "sentiment_signal",
        "price_ma_short",
        "price_ma_long",
        "price_signal",
        "combined_signal",
        "position",
    ]
    for col in required_cols:
        assert col in signals_df.columns
    assert not signals_df["sentiment_ma"].iloc[5:].isnull().any()
    assert not signals_df["position"].isnull().any()


def test_backtest_no_signals(strategy: Any) -> Any:
    """Test calling backtest before calculating signals."""
    with pytest.raises(ValueError, match="Calculate signals first"):
        strategy.backtest()
