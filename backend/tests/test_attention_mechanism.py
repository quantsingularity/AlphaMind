import sys
import numpy as np
import pytest
import tensorflow as tf

# Add backend directory to path
# Adjust this path if your backend directory is elsewhere
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
DFF = 512
NUM_LAYERS = 2
OUTPUT_SEQ_LENGTH = 5


# ---------------- Fixtures ---------------- #


@pytest.fixture
def sample_input():
    """Provides sample input tensor."""
    return tf.random.normal((BATCH_SIZE, SEQ_LENGTH, D_MODEL))


@pytest.fixture
def sample_mask():
    """Provides a sample mask tensor."""
    mask = tf.ones((BATCH_SIZE, 1, 1, SEQ_LENGTH), dtype=tf.float32)

    # Mask last 3 positions in each sequence
    tf.range(SEQ_LENGTH - 3, SEQ_LENGTH)
    zeros = tf.zeros((BATCH_SIZE, 1, 1, 3), dtype=tf.float32)

    # Prepare scatter indices
    scatter_indices = []
    for i in range(BATCH_SIZE):
        for j in range(SEQ_LENGTH - 3, SEQ_LENGTH):
            scatter_indices.append([i, 0, 0, j])

    mask = tf.tensor_scatter_nd_update(mask, scatter_indices, tf.reshape(zeros, [-1]))
    return mask


# -------------- MultiHeadAttention Tests -------------- #


def test_multi_head_attention_output_shape(sample_input):
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output = mha(sample_input, sample_input, sample_input)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_multi_head_attention_with_mask(sample_input, sample_mask):
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output = mha(sample_input, sample_input, sample_input, mask=sample_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


# -------------- TemporalAttentionBlock Tests -------------- #


def test_temporal_attention_block_output_shape(sample_input):
    block = TemporalAttentionBlock(d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF)
    output = block(sample_input, training=False)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_temporal_attention_block_with_mask(sample_input, sample_mask):
    block = TemporalAttentionBlock(d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF)
    output = block(sample_input, training=False, mask=sample_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


# -------------- Positional Encoding Tests -------------- #


def test_get_positional_encoding_shape():
    pos_encoding = get_positional_encoding(seq_length=SEQ_LENGTH, d_model=D_MODEL)
    assert pos_encoding.shape == (1, SEQ_LENGTH, D_MODEL)


def test_get_positional_encoding_values():
    pos_encoding = get_positional_encoding(seq_length=50, d_model=100)

    # Check first position (sin & cos)
    assert np.isclose(pos_encoding[0, 0, 0], np.sin(0.0))
    assert np.isclose(pos_encoding[0, 0, 1], np.cos(0.0))

    # Check second position (pos=1)
    assert np.isclose(pos_encoding[0, 1, 0], np.sin(1.0))
    assert np.isclose(pos_encoding[0, 1, 1], np.cos(1.0))


# -------------- Financial Transformer Tests -------------- #


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
