"""
FinancialTimeSeriesTransformer -- Full Transformer encoder model.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from ai_models.config import TransformerConfig
from ai_models.forecasting.attention import (
    TemporalAttentionBlock,
    get_positional_encoding,
)


class FinancialTimeSeriesTransformer(tf.keras.Model):
    """
    Stacked Transformer encoder for multi-horizon return forecasting.

    Parameters
    ----------
    config : ``TransformerConfig`` instance.
    """

    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = config or TransformerConfig()
        self.cfg = cfg
        self.input_proj = tf.keras.layers.Dense(cfg.d_model)
        self.enc_layers = [
            TemporalAttentionBlock(
                cfg.d_model, cfg.num_heads, cfg.dff, cfg.dropout_rate
            )
            for _ in range(cfg.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(cfg.dropout_rate)
        self.out_layer = tf.keras.layers.Dense(cfg.output_seq_length)

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        x = self.input_proj(x)
        x = x + get_positional_encoding(seq_len, self.cfg.d_model)
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=mask)
        return self.out_layer(x)

    def get_config(self):
        import dataclasses

        return {"config": dataclasses.asdict(self.cfg)}
