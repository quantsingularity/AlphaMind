from typing import Optional, Any
import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention mechanism for time series data.
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """

    def __init__(self, d_model: Any, num_heads: Any) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x: Any, batch_size: Any) -> Any:
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v: Any, k: Any, q: Any, mask: Optional[Any] = None) -> Any:
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += mask * -1000000000.0
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(output)


class TemporalAttentionBlock(tf.keras.layers.Layer):
    """
    Temporal attention block specifically designed for financial time series.
    Incorporates positional encoding to maintain sequence order information.
    """

    def __init__(self, d_model: Any, num_heads: Any, dff: Any, rate: Any = 0.1) -> None:
        super(TemporalAttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-06)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-06)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self, d_model: Any, dff: Any) -> Any:
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )

    def call(self, x: Any, training: Any = True, mask: Optional[Any] = None) -> Any:
        attn_output = self.mha(v=x, k=x, q=x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


def get_positional_encoding(seq_length: Any, d_model: Any) -> Any:
    """
    Create positional encodings for the input sequence.
    This helps the model understand the order of elements in the sequence.
    """
    pos_encoding = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = np.sin(pos / 10000 ** (i / d_model))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = np.cos(pos / 10000 ** (i / d_model))
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class FinancialTimeSeriesTransformer(tf.keras.Model):
    """
    Transformer model adapted for financial time series prediction.
    Incorporates temporal attention mechanisms and positional encoding.
    """

    def __init__(
        self,
        num_layers: Any,
        d_model: Any,
        num_heads: Any,
        dff: Any,
        input_seq_length: Any,
        output_seq_length: Any,
        rate: Any = 0.1,
    ) -> None:
        super(FinancialTimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length
        self.pos_encoding = get_positional_encoding(input_seq_length, d_model)
        self.enc_layers = [
            TemporalAttentionBlock(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(output_seq_length)

    def call(self, x: Any, training: Any = True, mask: Optional[Any] = None) -> Any:
        seq_len = tf.shape(x)[1]
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        output = self.final_layer(x)
        return output
