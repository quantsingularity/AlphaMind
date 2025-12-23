from typing import Any, List
import sys
import numpy as np
import pytest
import tensorflow as tf

sys.path.append("alphamind/backend")
from ai_models.attention_mechanism import (
    FinancialTimeSeriesTransformer,
    MultiHeadAttention,
    TemporalAttentionBlock,
    get_positional_encoding,
)

BATCH_SIZE = 4
SEQ_LENGTH = 10
D_MODEL = 128
NUM_HEADS = 8
DFF = 512
NUM_LAYERS = 2
OUTPUT_SEQ_LENGTH = 5


@pytest.fixture
def sample_input() -> Any:
    """Provides sample input tensor."""
    return tf.random.normal((BATCH_SIZE, SEQ_LENGTH, D_MODEL))


@pytest.fixture
def sample_mask() -> Any:
    """Provides a sample mask tensor."""
    mask = tf.ones((BATCH_SIZE, 1, 1, SEQ_LENGTH), dtype=tf.float32)
    tf.range(SEQ_LENGTH - 3, SEQ_LENGTH)
    zeros = tf.zeros((BATCH_SIZE, 1, 1, 3), dtype=tf.float32)
    scatter_indices: List[Any] = []
    for i in range(BATCH_SIZE):
        for j in range(SEQ_LENGTH - 3, SEQ_LENGTH):
            scatter_indices.append([i, 0, 0, j])
    mask = tf.tensor_scatter_nd_update(mask, scatter_indices, tf.reshape(zeros, [-1]))
    return mask


def test_multi_head_attention_output_shape(sample_input: Any) -> Any:
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output = mha(sample_input, sample_input, sample_input)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_multi_head_attention_with_mask(sample_input: Any, sample_mask: Any) -> Any:
    mha = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
    output = mha(sample_input, sample_input, sample_input, mask=sample_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_temporal_attention_block_output_shape(sample_input: Any) -> Any:
    block = TemporalAttentionBlock(d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF)
    output = block(sample_input, training=False)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_temporal_attention_block_with_mask(sample_input: Any, sample_mask: Any) -> Any:
    block = TemporalAttentionBlock(d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF)
    output = block(sample_input, training=False, mask=sample_mask)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)


def test_get_positional_encoding_shape() -> Any:
    pos_encoding = get_positional_encoding(seq_length=SEQ_LENGTH, d_model=D_MODEL)
    assert pos_encoding.shape == (1, SEQ_LENGTH, D_MODEL)


def test_get_positional_encoding_values() -> Any:
    pos_encoding = get_positional_encoding(seq_length=50, d_model=100)
    assert np.isclose(pos_encoding[0, 0, 0], np.sin(0.0))
    assert np.isclose(pos_encoding[0, 0, 1], np.cos(0.0))
    assert np.isclose(pos_encoding[0, 1, 0], np.sin(1.0))
    assert np.isclose(pos_encoding[0, 1, 1], np.cos(1.0))


def test_financial_transformer_output_shape(sample_input: Any) -> Any:
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


def test_financial_transformer_with_mask(sample_input: Any, sample_mask: Any) -> Any:
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
