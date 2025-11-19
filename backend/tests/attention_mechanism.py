import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention mechanism for time series data.
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Add the mask to the scaled tensor (if provided)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)

        # Reshape to (batch_size, seq_len_q, d_model)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(output)

class TemporalAttentionBlock(tf.keras.layers.Layer):
    """
    Temporal attention block specifically designed for financial time series.
    Incorporates positional encoding to maintain sequence order information.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TemporalAttentionBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=\'relu\'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, training=None, mask=None): # Use training=None as default for Keras
        # Self attention
        attn_output = self.mha(v=x, k=x, q=x, mask=mask)  # Use keyword args for clarity
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_len, d_model)

        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_len, d_model)

        return out2

def get_positional_encoding(seq_length, d_model):
    """
    Create positional encodings for the input sequence.
    This helps the model understand the order of elements in the sequence.
    """
    pos_encoding = np.zeros((seq_length, d_model))
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    # Add batch dimension
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

class FinancialTimeSeriesTransformer(tf.keras.Model):
    """
    Transformer model adapted for financial time series prediction.
    Incorporates temporal attention mechanisms and positional encoding.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_seq_length,
                 output_seq_length, rate=0.1):
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

        # Output projection layers
        self.final_layer = tf.keras.layers.Dense(output_seq_length)

    def call(self, x, training=None, mask=None): # Use training=None as default for Keras
        seq_len = tf.shape(x)[1]

        # Adding positional encoding
        # Ensure positional encoding matches input shape if necessary
        x_pos = x + self.pos_encoding[:, :seq_len, :]

        x_dropout = self.dropout(x_pos, training=training)

        # Encoder layers
        x_encoded = x_dropout
        for i in range(self.num_layers):
            # Pass arguments as keywords
            x_encoded = self.enc_layers[i](x_encoded, training=training, mask=mask)

        # Final projection to output sequence length
        output = self.final_layer(x_encoded)

        return output
