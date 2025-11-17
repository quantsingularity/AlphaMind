# Tests for ai_models/attention_mechanism.py
import sys

import numpy as np
import pytest
import tensorflow as tf

# Add the backend directory to the path to allow imports
sys.path.append("/home/ubuntu/alphamind_project/backend")

from ai_models.attention_mechanism import (
    FinancialTimeSeriesTransformer,
    MultiHeadAttention,
    TemporalAttentionBlock,
    get_positional_encoding,
)

# Constants for testing
BATCH_SIZE = 4
SEQ_LENGTH = 10
D_MODEL = 128
NUM_HEADS = 8
DFF = 512  # Feed-forward network dimension
NUM_LAYERS = 2
OUTPUT_SEQ_LENGTH = 5


@pytest.fixture
def sample_input():
    """Provides sample input tensor."""
    return tf.random.normal((BATCH_SIZE, SEQ_LENGTH, D_MODEL))


@pytest.fixture
def sample_mask():
    """Provides a sample mask tensor."""
    # Example mask: mask the last 3 positions for each sequence in the batch
    mask = tf.ones((BATCH_SIZE, 1, 1, SEQ_LENGTH))
    mask_indices = tf.range(SEQ_LENGTH - 3, SEQ_LENGTH)
    updates = tf.zeros((BATCH_SIZE, 1, 1, 3), dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[i, 0, 0, j] for i in range(BATCH_SIZE) for j in mask_indices],
        tf.reshape(updates, [-1]),
    )
    return mask  # Shape (batch_size, 1, 1, seq_length)


# --- Test MultiHeadAttention ---
def test_multi_head_attention_output_shape(sample_input):
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output = mha(sample_input, sample_input, sample_input)  # Q, K, V
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_multi_head_attention_with_mask(sample_input, sample_mask):
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output = mha(sample_input, sample_input, sample_input, mask=sample_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)
    # Further checks could involve verifying that masked positions have near-zero attention weights, but this is complex.


# --- Test TemporalAttentionBlock ---
def test_temporal_attention_block_output_shape(sample_input):
    block = TemporalAttentionBlock(d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF)
    output = block(
        sample_input, training=False
    )  # Set training=False for deterministic behavior (dropout)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_temporal_attention_block_with_mask(sample_input, sample_mask):
    block = TemporalAttentionBlock(d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF)
    output = block(sample_input, training=False, mask=sample_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


# --- Test get_positional_encoding ---
def test_get_positional_encoding_shape():
    pos_encoding = get_positional_encoding(seq_length=SEQ_LENGTH, d_model=D_MODEL)
    # Expected shape includes batch dimension added inside the function
    assert pos_encoding.shape == (1, SEQ_LENGTH, D_MODEL)


def test_get_positional_encoding_values():
    pos_encoding = get_positional_encoding(seq_length=50, d_model=100)
    # Check specific values based on the sin/cos formula
    # Example: Check first element (pos=0)
    assert np.isclose(
        pos_encoding[0, 0, 0], np.sin(0 / (10000 ** (0 / 100)))
    )  # sin(0) = 0
    assert np.isclose(
        pos_encoding[0, 0, 1], np.cos(0 / (10000 ** (0 / 100)))
    )  # cos(0) = 1
    # Example: Check second element (pos=1)
    assert np.isclose(pos_encoding[0, 1, 0], np.sin(1 / (10000 ** (0 / 100))))  # sin(1)
    assert np.isclose(pos_encoding[0, 1, 1], np.cos(1 / (10000 ** (0 / 100))))  # cos(1)


# --- Test FinancialTimeSeriesTransformer ---
def test_financial_transformer_output_shape(sample_input):
    transformer = FinancialTimeSeriesTransformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_seq_length=SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
    )
    output = transformer(sample_input, training=False)
    # The final layer projects the output for each position in the sequence
    # to the desired output sequence length (or number of output features per position)
    # The output shape should be (batch_size, input_seq_length, output_seq_length)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SEQ_LENGTH)


def test_financial_transformer_with_mask(sample_input, sample_mask):
    transformer = FinancialTimeSeriesTransformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_seq_length=SEQ_LENGTH,
        output_seq_length=OUTPUT_SEQ_LENGTH,
    )
    output = transformer(sample_input, training=False, mask=sample_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SEQ_LENGTH)
