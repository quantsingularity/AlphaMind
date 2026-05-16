"""
Transformer-based Time-Series Forecasting
------------------------------------------
MultiHeadAttention            -- Scaled dot-product multi-head attention
TemporalAttentionBlock        -- Transformer encoder block
get_positional_encoding       -- Sinusoidal positional encoding
FinancialTimeSeriesTransformer -- Full Transformer encoder
AdvancedTimeSeriesForecaster  -- High-level training/inference wrapper
"""

from ai_models.forecasting.advanced import AdvancedTimeSeriesForecaster
from ai_models.forecasting.attention import (
    MultiHeadAttention,
    TemporalAttentionBlock,
    get_positional_encoding,
)
from ai_models.forecasting.transformer import FinancialTimeSeriesTransformer

__all__ = [
    "MultiHeadAttention",
    "TemporalAttentionBlock",
    "get_positional_encoding",
    "FinancialTimeSeriesTransformer",
    "AdvancedTimeSeriesForecaster",
]
