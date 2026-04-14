from typing import Any

import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        num_layers: Any = 4,
        d_model: Any = 64,
        num_heads: Any = 8,
        dff: Any = 128,
        rate: Any = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(
                {
                    "mha": tf.keras.layers.MultiHeadAttention(
                        num_heads=num_heads, key_dim=d_model
                    ),
                    "ffn1": tf.keras.layers.Dense(dff, activation="relu"),
                    "ffn2": tf.keras.layers.Dense(d_model),
                    "ln1": tf.keras.layers.LayerNormalization(epsilon=1e-6),
                    "ln2": tf.keras.layers.LayerNormalization(epsilon=1e-6),
                    "drop1": tf.keras.layers.Dropout(rate),
                    "drop2": tf.keras.layers.Dropout(rate),
                }
            )

    def call(self, inputs: Any, training: Any = True, mask: Any = None) -> Any:
        x = inputs
        for block in self.blocks:
            attn_output = block["mha"](query=x, key=x, value=x, attention_mask=mask)
            attn_output = block["drop1"](attn_output, training=training)
            out1 = block["ln1"](x + attn_output)
            ffn_output = block["ffn2"](block["ffn1"](out1))
            ffn_output = block["drop2"](ffn_output, training=training)
            x = block["ln2"](out1 + ffn_output)
        return x


class TemporalFusionDecoder(tf.keras.layers.Layer):

    def __init__(
        self, num_heads: Any = 8, future_steps: Any = 24, hidden_size: Any = 32
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.future_steps = future_steps
        self.hidden_size = hidden_size
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size
        )
        self.context_proj = tf.keras.layers.Dense(hidden_size)
        self.decoder_proj = tf.keras.layers.Dense(hidden_size)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_size * 2, activation="relu"),
                tf.keras.layers.Dense(hidden_size),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs: Any, training: Any = True) -> Any:
        context = self.context_proj(inputs["context"])
        decoder_features = self.decoder_proj(inputs["decoder_features"])
        attn_output = self.attention(query=decoder_features, key=context, value=context)
        out1 = self.layernorm1(attn_output + decoder_features)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(ffn_output + out1)
        return self.output_layer(out2)


class TemporalFusionTransformer(tf.keras.Model):

    def __init__(self, num_encoder_steps: Any, num_features: Any) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=4, d_model=64, num_heads=8, dff=128, rate=0.1
        )
        self.decoder = TemporalFusionDecoder(
            num_heads=8, future_steps=24, hidden_size=32
        )

    def call(self, inputs: Any) -> Any:
        context = self.encoder(inputs["encoder_features"])
        predictions = self.decoder(
            {"context": context, "decoder_features": inputs["decoder_features"]}
        )
        return {
            "short_term": predictions[:, :6],
            "medium_term": predictions[:, 6:18],
            "long_term": predictions[:, 18:],
        }
