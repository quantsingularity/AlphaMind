import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers=4, d_model=64):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.encoder_layers = [
            tf.keras.layers.TransformerEncoderLayer(d_model, num_heads=8)
            for _ in range(num_layers)
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class TemporalFusionDecoder(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, future_steps=24, hidden_size=32):
        super().__init__()
        self.num_heads = num_heads
        self.future_steps = future_steps
        self.hidden_size = hidden_size
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_size * 2, activation="relu"),
                tf.keras.layers.Dense(hidden_size),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        context = inputs["context"]
        decoder_features = inputs["decoder_features"]

        # Self-attention
        attn_output = self.attention(query=decoder_features, key=context, value=context)

        # Add & Norm
        out1 = self.layernorm1(attn_output + decoder_features)

        # Feed Forward
        ffn_output = self.ffn(out1)

        # Add & Norm
        out2 = self.layernorm2(ffn_output + out1)

        # Output projection
        return self.output_layer(out2)


class TemporalFusionTransformer(tf.keras.Model):
    def __init__(self, num_encoder_steps, num_features):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers=4, d_model=64)
        self.decoder = TemporalFusionDecoder(
            num_heads=8, future_steps=24, hidden_size=32
        )

    def call(self, inputs):
        context = self.encoder(inputs["encoder_features"])
        predictions = self.decoder(
            {"context": context, "decoder_features": inputs["decoder_features"]}
        )
        return {
            "short_term": predictions[:, :6],
            "medium_term": predictions[:, 6:18],
            "long_term": predictions[:, 18:],
        }
