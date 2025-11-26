import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LeakyReLU, Reshape, LSTM

# --- Model Components ---


class TransformerGenerator(tf.keras.layers.Layer):
    def __init__(self, seq_length, n_features):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        # Layer 1: Takes latent vector as input
        self.dense1 = Dense(128, activation="relu")
        # Layer 2: Projects to the flattened time series shape
        self.dense2 = Dense(seq_length * n_features)
        # Layer 3: Reshapes to (seq_length, n_features)
        self.reshape = Reshape((seq_length, n_features))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.reshape(x)


class TimeSeriesDiscriminator(tf.keras.layers.Layer):
    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length
        # Conv1D layers for feature extraction in time series
        self.conv1 = Conv1D(64, kernel_size=3, strides=2, padding="same")
        self.conv2 = Conv1D(128, kernel_size=3, strides=2, padding="same")
        self.flatten = Flatten()
        # Output layer for binary classification (Real/Fake)
        self.dense = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = LeakyReLU(0.2)(x)
        x = self.flatten(x)
        return self.dense(x)


class RegimeClassifier(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # LSTM layer to process the time series sequence
        self.lstm = LSTM(64)
        # Output layer for multi-class classification (3 market regimes: e.g., Normal, Volatile, Crisis)
        self.dense = Dense(3, activation="softmax")  # 3 market regimes

    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)


def regime_consistency_loss(regime_match, expected_distribution=None):
    """
    Calculate consistency loss for regime classification using KL Divergence.

    Penalizes the Generator if the predicted regime distribution of the
    fake data deviates significantly from a target 'expected_distribution'.
    """
    if expected_distribution is None:
        # Default target distribution: 60% Normal, 30% Volatile, 10% Crisis
        expected_distribution = tf.constant([0.6, 0.3, 0.1], dtype=tf.float32)

    # Ensure regime_match (predicted distribution) is not zero to avoid log(0)
    regime_match = tf.clip_by_value(regime_match, 1e-8, 1.0)

    # Calculate KL Divergence
    # KL(P || Q) = sum(P * log(P / Q)), where P is expected (target) and Q is generated (regime_match)
    kl_div = tf.reduce_sum(
        expected_distribution * tf.math.log(expected_distribution / regime_match)
    )
    return kl_div


# --------------------------------


class MarketGAN(tf.keras.Model):
    """
    Generative Adversarial Network with an Auxiliary Classifier (AC-GAN structure)
    for generating synthetic market time series data.
    """

    def __init__(self, seq_length, n_features):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.latent_dim = 100
        # NOTE: Batch size should be set during dataset creation or passed to train_step
        self.batch_size = 32

        # Core GAN components
        self.generator = TransformerGenerator(seq_length, n_features)
        self.discriminator = TimeSeriesDiscriminator(seq_length)

        # Auxiliary component for regime-aware generation
        self.aux_classifier = RegimeClassifier()

        # Loss function for binary classification
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def compile(self, g_optimizer, d_optimizer):
        """Configure optimizers for the generator and discriminator."""
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @tf.function
    def train_step(self, real_data):
        """
        Custom training logic for one step of the GAN.

        Args:
            real_data: A batch of real time series data.
        """
        self.batch_size = tf.shape(real_data)[0]

        # 1. Train Discriminator
        noise = tf.random.normal((self.batch_size, self.latent_dim))
        fake_data = self.generator(noise)

        # Create labels: 1 for real, 0 for fake
        real_labels = tf.ones((self.batch_size, 1))
        fake_labels = tf.zeros((self.batch_size, 1))

        with tf.GradientTape() as d_tape:
            # Predictions
            real_pred = self.discriminator(real_data)
            fake_pred = self.discriminator(fake_data)

            # Loss: BCE for real (should be 1) + BCE for fake (should be 0)
            d_loss_real = self.loss_fn(real_labels, real_pred)
            d_loss_fake = self.loss_fn(fake_labels, fake_pred)
            d_loss = d_loss_real + d_loss_fake

        # Apply gradients
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        # 2. Train Generator (with Auxiliary Classifier loss)
        noise = tf.random.normal((self.batch_size, self.latent_dim))

        # The generator aims to fool the discriminator (label=1) AND
        # to match the target regime distribution (via aux_classifier)
        with tf.GradientTape() as g_tape:
            fake_data = self.generator(noise)
            validity = self.discriminator(
                fake_data
            )  # Generator wants this to be close to 1

            # Auxiliary Loss Component (Regime Consistency)
            # This loss is calculated on the fake data to control its market properties
            regime_match = self.aux_classifier(fake_data)
            aux_loss = regime_consistency_loss(regime_match)

            # Generator Loss: Validity Loss (BCE) + Auxiliary Loss (KL Divergence)
            g_loss = self.loss_fn(real_labels, validity) + aux_loss

        # Apply gradients
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        # 3. Train Auxiliary Classifier (optional, but often needed to ensure robust classification)
        # The auxiliary classifier is trained on real data with ground-truth regime labels
        # (This part is typically handled outside the core GAN train_step as it requires external labels).
        # For simplicity within this structure, we skip explicit AC training here,
        # but in a full MarketGAN, you would train AC on real data with its true regime labels.

        return {"d_loss": d_loss, "g_loss": g_loss, "aux_loss": aux_loss}
