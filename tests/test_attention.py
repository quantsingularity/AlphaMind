import os
import sys
import unittest
import numpy as np
import tensorflow as tf

# Correct the path to the backend directory within the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from ai_models.attention_mechanism import (
    MultiHeadAttention, 
    TemporalAttentionBlock, 
    get_positional_encoding,
    FinancialTimeSeriesTransformer
)


class TestMultiHeadAttention(unittest.TestCase):
    """Test suite for the MultiHeadAttention class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.d_model = 64
        self.num_heads = 4
        self.batch_size = 8
        self.seq_length = 10
        self.attention = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        
        # Create sample input tensors
        self.query = tf.random.normal((self.batch_size, self.seq_length, self.d_model))
        self.key = tf.random.normal((self.batch_size, self.seq_length, self.d_model))
        self.value = tf.random.normal((self.batch_size, self.seq_length, self.d_model))
        
    def test_initialization(self):
        """Test that the MultiHeadAttention initializes correctly"""
        self.assertEqual(self.attention.d_model, self.d_model)
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.depth, self.d_model // self.num_heads)
        
    def test_split_heads(self):
        """Test the split_heads method"""
        x = tf.random.normal((self.batch_size, self.seq_length, self.d_model))
        split = self.attention.split_heads(x, self.batch_size)
        
        # Check shape: (batch_size, num_heads, seq_length, depth)
        expected_shape = (self.batch_size, self.num_heads, self.seq_length, self.d_model // self.num_heads)
        self.assertEqual(split.shape, expected_shape)
        
    def test_call(self):
        """Test the call method (forward pass)"""
        output = self.attention(self.value, self.key, self.query)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.d_model)
        self.assertEqual(output.shape, expected_shape)
        
    def test_mask(self):
        """Test attention with mask"""
        # Create a mask where the first half of the sequence can't attend to the second half
        # This creates a more significant masking effect than the previous approach
        mask = np.ones((self.batch_size, 1, self.seq_length, self.seq_length))
        mask_half = self.seq_length // 2
        mask[:, :, :mask_half, mask_half:] = 0  # First half can't see second half
        mask = tf.constant(mask, dtype=tf.float32)
        
        # Run attention with mask
        output_with_mask = self.attention(self.value, self.key, self.query, mask=mask)
        output_without_mask = self.attention(self.value, self.key, self.query)
        
        # Outputs should be different with and without mask
        # Use a smaller tolerance to detect subtle differences
        self.assertFalse(np.allclose(output_with_mask.numpy(), output_without_mask.numpy(), atol=1e-3))
        
        # Check output shape remains the same
        expected_shape = (self.batch_size, self.seq_length, self.d_model)
        self.assertEqual(output_with_mask.shape, expected_shape)


class TestTemporalAttentionBlock(unittest.TestCase):
    """Test suite for the TemporalAttentionBlock class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.d_model = 64
        self.num_heads = 4
        self.dff = 256
        self.batch_size = 8
        self.seq_length = 10
        self.block = TemporalAttentionBlock(
            d_model=self.d_model, 
            num_heads=self.num_heads, 
            dff=self.dff
        )
        
        # Create sample input tensor
        self.x = tf.random.normal((self.batch_size, self.seq_length, self.d_model))
        
    def test_initialization(self):
        """Test that the TemporalAttentionBlock initializes correctly"""
        self.assertIsInstance(self.block.mha, MultiHeadAttention)
        self.assertEqual(self.block.mha.d_model, self.d_model)
        self.assertEqual(self.block.mha.num_heads, self.num_heads)
        
    def test_call(self):
        """Test the call method (forward pass)"""
        output = self.block(self.x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.d_model)
        self.assertEqual(output.shape, expected_shape)
        
    def test_training_mode(self):
        """Test the block in training and inference modes"""
        output_training = self.block(self.x, training=True)
        output_inference = self.block(self.x, training=False)
        
        # Outputs should be different in training vs inference mode due to dropout
        # Note: This test might occasionally fail due to randomness in dropout
        # We use a high tolerance to account for this
        self.assertFalse(np.allclose(output_training.numpy(), output_inference.numpy(), atol=1e-2))
        
    def test_mask(self):
        """Test the block with a mask"""
        # Create a simple mask
        mask = tf.zeros((self.batch_size, 1, 1, self.seq_length))
        
        # Run with mask
        output_with_mask = self.block(self.x, mask=mask)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_length, self.d_model)
        self.assertEqual(output_with_mask.shape, expected_shape)


class TestPositionalEncoding(unittest.TestCase):
    """Test suite for the positional encoding function"""
    
    def test_shape(self):
        """Test the shape of the positional encoding"""
        seq_length = 20
        d_model = 64
        pos_encoding = get_positional_encoding(seq_length, d_model)
        
        # Check shape: (1, seq_length, d_model)
        expected_shape = (1, seq_length, d_model)
        self.assertEqual(pos_encoding.shape, expected_shape)
        
    def test_values(self):
        """Test specific values in the positional encoding"""
        seq_length = 10
        d_model = 8
        pos_encoding = get_positional_encoding(seq_length, d_model).numpy()[0]
        
        # Check that the encoding follows the sine/cosine pattern
        # Even indices should use sine, odd indices should use cosine
        for pos in range(seq_length):
            for i in range(0, d_model, 2):
                # Check sine pattern for even indices
                expected_value = np.sin(pos / (10000 ** (i / d_model)))
                self.assertAlmostEqual(pos_encoding[pos, i], expected_value, places=5)
                
                # Check cosine pattern for odd indices
                if i + 1 < d_model:
                    expected_value = np.cos(pos / (10000 ** (i / d_model)))
                    self.assertAlmostEqual(pos_encoding[pos, i+1], expected_value, places=5)


class TestFinancialTimeSeriesTransformer(unittest.TestCase):
    """Test suite for the FinancialTimeSeriesTransformer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.num_layers = 2
        self.d_model = 64
        self.num_heads = 4
        self.dff = 256
        self.input_seq_length = 20
        self.output_seq_length = 5
        self.batch_size = 8
        
        self.transformer = FinancialTimeSeriesTransformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            input_seq_length=self.input_seq_length,
            output_seq_length=self.output_seq_length
        )
        
        # Create sample input tensor
        self.x = tf.random.normal((self.batch_size, self.input_seq_length, self.d_model))
        
    def test_initialization(self):
        """Test that the transformer initializes correctly"""
        self.assertEqual(self.transformer.d_model, self.d_model)
        self.assertEqual(self.transformer.num_layers, self.num_layers)
        self.assertEqual(self.transformer.input_seq_length, self.input_seq_length)
        self.assertEqual(self.transformer.output_seq_length, self.output_seq_length)
        self.assertEqual(len(self.transformer.enc_layers), self.num_layers)
        
    def test_call(self):
        """Test the call method (forward pass)"""
        output = self.transformer(self.x)
        
        # Check output shape - should match the batch size and output sequence length
        expected_shape = (self.batch_size, self.input_seq_length, self.output_seq_length)
        self.assertEqual(output.shape, expected_shape)
        
    def test_training_mode(self):
        """Test the transformer in training and inference modes"""
        output_training = self.transformer(self.x, training=True)
        output_inference = self.transformer(self.x, training=False)
        
        # Outputs should be different in training vs inference mode due to dropout
        self.assertFalse(np.allclose(output_training.numpy(), output_inference.numpy(), atol=1e-2))
        
    def test_mask(self):
        """Test the transformer with a mask"""
        # Create a simple mask
        mask = tf.zeros((self.batch_size, 1, 1, self.input_seq_length))
        
        # Run with mask
        output_with_mask = self.transformer(self.x, mask=mask)
        
        # Check output shape
        expected_shape = (self.batch_size, self.input_seq_length, self.output_seq_length)
        self.assertEqual(output_with_mask.shape, expected_shape)
        
    def test_end_to_end(self):
        """Test an end-to-end forward pass with random data"""
        # Create random time series data
        input_data = tf.random.normal((self.batch_size, self.input_seq_length, self.d_model))
        
        # Forward pass
        output = self.transformer(input_data)
        
        # Check output shape
        expected_shape = (self.batch_size, self.input_seq_length, self.output_seq_length)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output values are finite (no NaNs or Infs)
        self.assertTrue(np.all(np.isfinite(output.numpy())))


if __name__ == "__main__":
    unittest.main()
