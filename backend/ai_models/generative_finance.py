# import numpy as np
# import tensorflow as tf


# class TransformerGenerator(tf.keras.layers.Layer):
#     def __init__(self, seq_length, n_features):
#         super().__init__()
#         self.seq_length = seq_length
#         self.n_features = n_features
#         self.dense1 = tf.keras.layers.Dense(128, activation="relu")
#         self.dense2 = tf.keras.layers.Dense(seq_length * n_features)
#         self.reshape = tf.keras.layers.Reshape((seq_length, n_features))

#     def call(self, inputs):
#         x = self.dense1(inputs)
#         x = self.dense2(x)
#         return self.reshape(x)


# class TimeSeriesDiscriminator(tf.keras.layers.Layer):
#     def __init__(self, seq_length):
#         super().__init__()
#         self.seq_length = seq_length
#         self.conv1 = tf.keras.layers.Conv1D(
#             64, kernel_size=3, strides=2, padding="same"
        )
#         self.conv2 = tf.keras.layers.Conv1D(
#             128, kernel_size=3, strides=2, padding="same"
        )
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = tf.keras.layers.LeakyReLU(0.2)(x)
#         x = self.conv2(x)
#         x = tf.keras.layers.LeakyReLU(0.2)(x)
#         x = self.flatten(x)
#         return self.dense(x)


# class RegimeClassifier(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#         self.lstm = tf.keras.layers.LSTM(64)
#         self.dense = tf.keras.layers.Dense(3, activation="softmax")  # 3 market regimes

#     def call(self, inputs):
#         x = self.lstm(inputs)
#         return self.dense(x)


# def regime_consistency_loss(regime_match):
#    """Calculate consistency loss for regime classification"""
#    # Penalize if generated data doesn't match expected regime distribution
##     expected_distribution = tf.constant([0.6, 0.3, 0.1])  # Normal, Volatile, Crisis
##     kl_div = tf.reduce_sum(
##         expected_distribution * tf.math.log(expected_distribution / regime_match)
#    )
##     return kl_div
#
#
## class MarketGAN(tf.keras.Model):
##     def __init__(self, seq_length, n_features):
##         super().__init__()
##         self.seq_length = seq_length
##         self.n_features = n_features
##         self.latent_dim = 100
##         self.batch_size = 32
##         self.generator = TransformerGenerator(seq_length, n_features)
##         self.discriminator = TimeSeriesDiscriminator(seq_length)
##         self.aux_classifier = RegimeClassifier()
##         self.loss_fn = tf.keras.losses.BinaryCrossentropy()
#
##     def compile(self, g_optimizer, d_optimizer):
##         super().compile()
##         self.g_optimizer = g_optimizer
##         self.d_optimizer = d_optimizer
#
##     def train_step(self, real_data):
#        # Train discriminator
##         noise = tf.random.normal((self.batch_size, self.seq_length, self.latent_dim))
##         fake_data = self.generator(noise)
#
##         real_labels = tf.ones((self.batch_size, 1))
##         fake_labels = tf.zeros((self.batch_size, 1))
#
##         with tf.GradientTape() as d_tape:
##             real_pred = self.discriminator(real_data)
##             fake_pred = self.discriminator(fake_data)
##             d_loss = self.loss_fn(real_labels, real_pred) + self.loss_fn(
##                 fake_labels, fake_pred
#            )
#
#        # Train generator
##         with tf.GradientTape() as g_tape:
##             fake_data = self.generator(noise)
##             validity = self.discriminator(fake_data)
##             regime_match = self.aux_classifier(fake_data)
##             g_loss = self.loss_fn(real_labels, validity) + regime_consistency_loss(
##                 regime_match
#            )
#
##         return {"d_loss": d_loss, "g_loss": g_loss}
