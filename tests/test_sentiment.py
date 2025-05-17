import os
import sys
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf

# Correct the path to the backend directory within the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from alternative_data.sentiment_analysis import MarketSentimentAnalyzer, SentimentBasedStrategy


class TestMarketSentimentAnalyzer(unittest.TestCase):
    """Test suite for the MarketSentimentAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vocab_size = 5000
        self.embedding_dim = 64
        self.max_length = 100
        self.analyzer = MarketSentimentAnalyzer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length
        )
        
        # Create sample data
        self.sample_texts = [
            "The company reported strong earnings, exceeding analyst expectations.",
            "The stock market crashed today as investors panicked over economic data.",
            "Neutral outlook for the tech sector as competition increases.",
            "Impressive growth in revenue despite challenging market conditions.",
            "Disappointing quarterly results led to a significant drop in share price."
        ]
        
        # Labels: 0=negative, 1=neutral, 2=positive
        self.sample_labels = [2, 0, 1, 2, 0]
        
    def test_initialization(self):
        """Test that the MarketSentimentAnalyzer initializes correctly"""
        self.assertEqual(self.analyzer.vocab_size, self.vocab_size)
        self.assertEqual(self.analyzer.embedding_dim, self.embedding_dim)
        self.assertEqual(self.analyzer.max_length, self.max_length)
        self.assertIsNone(self.analyzer.tokenizer)
        self.assertIsNotNone(self.analyzer.model)
        
    def test_model_architecture(self):
        """Test the model architecture"""
        # Check input shape
        self.assertEqual(self.analyzer.model.input_shape, (None, self.max_length))
        
        # Check output shape - 3 classes (negative, neutral, positive)
        self.assertEqual(self.analyzer.model.output_shape, (None, 3))
        
        # Check that the model has the expected number of layers
        # This is a rough check - the exact number might change with implementation details
        self.assertGreaterEqual(len(self.analyzer.model.layers), 5)
        
    def test_prepare_tokenizer(self):
        """Test the tokenizer preparation"""
        self.analyzer.prepare_tokenizer(self.sample_texts)
        
        # Check that tokenizer is initialized
        self.assertIsNotNone(self.analyzer.tokenizer)
        
        # Check that tokenizer has the expected vocabulary size
        self.assertEqual(self.analyzer.tokenizer.num_words, self.vocab_size)
        
        # Check that tokenizer has learned some words
        self.assertGreater(len(self.analyzer.tokenizer.word_index), 10)
        
    def test_preprocess_text(self):
        """Test text preprocessing"""
        # Prepare tokenizer first
        self.analyzer.prepare_tokenizer(self.sample_texts)
        
        # Preprocess the texts
        padded_sequences = self.analyzer.preprocess_text(self.sample_texts)
        
        # Check shape
        self.assertEqual(padded_sequences.shape, (len(self.sample_texts), self.max_length))
        
        # Check that sequences are padded correctly
        self.assertEqual(padded_sequences[0, -1], 0)  # Last token should be padding (0)
        
        # Test error when tokenizer not initialized
        new_analyzer = MarketSentimentAnalyzer()
        with self.assertRaises(ValueError):
            new_analyzer.preprocess_text(self.sample_texts)
            
    def test_train(self):
        """Test the training function"""
        # Train for just 1 epoch to keep the test fast
        history = self.analyzer.train(self.sample_texts, self.sample_labels, epochs=1)
        
        # Check that history contains expected metrics
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)
        
        # Check that tokenizer was initialized
        self.assertIsNotNone(self.analyzer.tokenizer)
        
    def test_predict(self):
        """Test the prediction function"""
        # Prepare tokenizer and train the model
        self.analyzer.prepare_tokenizer(self.sample_texts)
        self.analyzer.train(self.sample_texts, self.sample_labels, epochs=1)
        
        # Make predictions
        predictions = self.analyzer.predict(self.sample_texts)
        
        # Check shape - should be (n_samples, 3) for 3 sentiment classes
        self.assertEqual(predictions.shape, (len(self.sample_texts), 3))
        
        # Check that predictions are probabilities (sum to 1)
        for pred in predictions:
            self.assertAlmostEqual(np.sum(pred), 1.0, places=5)
            
    def test_get_sentiment_score(self):
        """Test the sentiment score function"""
        # Prepare tokenizer and train the model
        self.analyzer.prepare_tokenizer(self.sample_texts)
        self.analyzer.train(self.sample_texts, self.sample_labels, epochs=1)
        
        # Get sentiment scores
        scores = self.analyzer.get_sentiment_score(self.sample_texts)
        
        # Check shape - should be (n_samples,)
        self.assertEqual(scores.shape, (len(self.sample_texts),))
        
        # Check that scores are between -1 and 1
        self.assertTrue(np.all(scores >= -1))
        self.assertTrue(np.all(scores <= 1))
        
    def test_save_load(self):
        """Test saving and loading the model and tokenizer"""
        # Create temporary file paths
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_sentiment')
        os.makedirs(temp_dir, exist_ok=True)
        temp_model_path = os.path.join(temp_dir, 'model.h5')
        temp_tokenizer_path = os.path.join(temp_dir, 'tokenizer.pkl')
        
        try:
            # Prepare tokenizer and train the model
            self.analyzer.prepare_tokenizer(self.sample_texts)
            self.analyzer.train(self.sample_texts, self.sample_labels, epochs=1)
            
            # Get predictions before saving
            predictions_before = self.analyzer.predict(self.sample_texts)
            
            # Save the model and tokenizer
            self.analyzer.save(temp_model_path, temp_tokenizer_path)
            
            # Check that files exist
            self.assertTrue(os.path.exists(temp_model_path))
            self.assertTrue(os.path.exists(temp_tokenizer_path))
            
            # Create a new analyzer
            new_analyzer = MarketSentimentAnalyzer(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                max_length=self.max_length
            )
            
            # Load the model and tokenizer
            new_analyzer.load(temp_model_path, temp_tokenizer_path)
            
            # Get predictions after loading
            predictions_after = new_analyzer.predict(self.sample_texts)
            
            # Predictions should be identical
            np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5)
            
        finally:
            # Clean up
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestSentimentBasedStrategy(unittest.TestCase):
    """Test suite for the SentimentBasedStrategy class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a sentiment analyzer
        self.analyzer = MarketSentimentAnalyzer(vocab_size=1000, embedding_dim=32, max_length=50)
        
        # Create sample price data with string dates instead of numpy.datetime64
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        string_dates = [date.strftime('%Y-%m-%d') for date in dates]
        self.price_data = pd.DataFrame({
            'date': string_dates,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
        
        # Create sample news data with string dates instead of numpy.datetime64
        news_dates = np.random.choice(dates, size=50, replace=True)
        news_texts = [
            f"Market news for {date.strftime('%Y-%m-%d')}: " + 
            np.random.choice([
                "Positive outlook for the economy.",
                "Concerns about inflation are growing.",
                "Central bank maintains current policy.",
                "Strong corporate earnings reported.",
                "Market volatility increases due to geopolitical tensions."
            ])
            for date in news_dates
        ]
        
        # Convert numpy.datetime64 to string dates for compatibility
        string_dates = [date.strftime('%Y-%m-%d') for date in news_dates]
        
        self.news_data = pd.DataFrame({
            'date': string_dates,
            'text': news_texts
        })
        
        # Prepare the analyzer
        self.analyzer.prepare_tokenizer(self.news_data['text'])
        
        # Mock the predict method to return predictable sentiment scores
        self.original_predict = self.analyzer.predict
        self.analyzer.predict = lambda texts: np.array([
            [0.2, 0.3, 0.5],  # Positive
            [0.5, 0.3, 0.2],  # Negative
            [0.3, 0.4, 0.3],  # Neutral
            [0.1, 0.2, 0.7],  # Very positive
            [0.7, 0.2, 0.1],  # Very negative
        ] * (len(texts) // 5 + 1))[:len(texts)]
        
        # Create the strategy
        self.strategy = SentimentBasedStrategy(self.analyzer, self.price_data)
        
    def tearDown(self):
        """Clean up after tests"""
        # Restore original predict method
        if hasattr(self, 'original_predict'):
            self.analyzer.predict = self.original_predict
        
    def test_initialization(self):
        """Test that the SentimentBasedStrategy initializes correctly"""
        self.assertEqual(self.strategy.sentiment_analyzer, self.analyzer)
        self.assertEqual(self.strategy.price_data.equals(self.price_data), True)
        
    def test_calculate_signals(self):
        """Test the signal calculation"""
        # Calculate signals
        signals = self.strategy.calculate_signals(self.news_data)
        
        # Check that the result is a DataFrame
        self.assertIsInstance(signals, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = [
            'date', 'open', 'high', 'low', 'close', 'volume',
            'sentiment_score', 'sentiment_ma', 'sentiment_signal',
            'price_ma_short', 'price_ma_long', 'price_signal',
            'combined_signal', 'position'
        ]
        for col in expected_columns:
            self.assertIn(col, signals.columns)
            
        # Check that sentiment scores are calculated
        self.assertTrue(np.all(np.isfinite(signals['sentiment_score'])))
        
        # Check that positions are -1, 0, or 1
        self.assertTrue(np.all(np.isin(signals['position'], [-1, 0, 1])))
        
    def test_backtest(self):
        """Test the backtest function"""
        # Calculate signals first
        self.strategy.price_data = self.strategy.calculate_signals(self.news_data)
        
        # Run backtest
        results = self.strategy.backtest(initial_capital=10000)
        
        # Check that performance metrics were calculated
        self.assertIsNotNone(self.strategy.performance_metrics)
        self.assertIn('total_return', self.strategy.performance_metrics)
        self.assertIn('annual_return', self.strategy.performance_metrics)
        self.assertIn('annual_volatility', self.strategy.performance_metrics)
        self.assertIn('sharpe_ratio', self.strategy.performance_metrics)
        self.assertIn('max_drawdown', self.strategy.performance_metrics)
        
        # Check that results contain expected columns
        expected_columns = [
            'market_return', 'strategy_return', 
            'cumulative_market_return', 'cumulative_strategy_return',
            'market_equity', 'strategy_equity',
            'market_peak', 'strategy_peak',
            'market_drawdown', 'strategy_drawdown'
        ]
        for col in expected_columns:
            self.assertIn(col, results.columns)
            
        # Check that equity values are positive
        self.assertTrue(np.all(results['market_equity'] > 0))
        self.assertTrue(np.all(results['strategy_equity'] > 0))
        
        # Check that drawdowns are <= 0
        self.assertTrue(np.all(results['market_drawdown'] <= 0))
        self.assertTrue(np.all(results['strategy_drawdown'] <= 0))
        
    def test_error_handling(self):
        """Test error handling in the backtest function"""
        # Try to run backtest without calculating signals first
        self.strategy.price_data = self.price_data  # Reset to original data without signals
        
        with self.assertRaises(ValueError):
            self.strategy.backtest()


if __name__ == "__main__":
    unittest.main()
