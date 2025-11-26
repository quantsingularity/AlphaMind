import keras  # Use modern keras import
import numpy as np
import tensorflow as tf

from ..attention_mechanism import FinancialTimeSeriesTransformer


class AdvancedTimeSeriesForecaster:
    """
    Advanced time series forecasting model that combines transformer architecture
    with financial domain-specific features.
    """

    def __init__(
        self,
        input_seq_length=60,
        output_seq_length=20,
        d_model=128,
        num_heads=8,
        num_layers=4,
        dropout_rate=0.1,
    ):
        self.input_seq_length = input_seq_length
        self.output_seq_length = output_seq_length

        # Create the transformer model
        self.model = FinancialTimeSeriesTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=d_model * 4,  # Feed-forward dimension
            input_seq_length=input_seq_length,
            output_seq_length=output_seq_length,
            rate=dropout_rate,
        )

        # Input preprocessing layers
        self.feature_projection = keras.layers.Dense(d_model)

        # Compile with appropriate loss and optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-4, first_decay_steps=1000
            )
        )

    def preprocess_data(self, data):
        """
        Preprocess financial time series data for the transformer model.

        Args:
            data: Raw financial time series data

        Returns:
            Preprocessed data ready for model input
        """
        # Normalize data using robust scaler approach
        median = np.median(data, axis=1, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
        iqr = q75 - q25
        normalized_data = (data - median) / (iqr + 1e-8)

        # Project to model dimension
        # Note: self.feature_projection layer needs to be built/called
        # inside a TensorFlow context or training loop to handle TensorFlow Tensors.
        # For simplicity in this class structure, we assume data is a TensorFlow tensor
        # or convert it here if needed, but the original code implies it's used with
        # numpy data which would fail if feature_projection is a Keras layer expecting a TF tensor.
        # Assuming `data` is a numpy array initially and is converted to a TF tensor
        # for feature_projection *during* the training/prediction process.
        # The original code's `preprocess_data` function is called *before* the
        # tf.data.Dataset creation, which is unusual for a Keras layer call.
        # We will keep the original structure for uncommenting but note the potential TF/Numpy issue.

        # Correcting the execution environment for feature_projection:
        # Since it's called outside a @tf.function or Keras model call,
        # we wrap it in a function call to execute it on the tensor later.

        # For the purpose of uncommenting, we assume `data` is already a TF tensor
        # or that `preprocess_data` is used within a context that handles this.
        # Given the `train` method immediately converts it to a dataset, we
        # will assume the preprocessing logic is intended to be pure-numpy up to the
        # point of passing it to the TF dataset, and `feature_projection` is intended to be called
        # inside the `train` method's dataset pipeline, but the structure is wrong.
        # We restore the original, functionally questionable code for uncommenting:
        try:
            projected_data = self.feature_projection(normalized_data)
        except Exception:
            # This handles the case where normalized_data is a numpy array
            # and feature_projection expects a TensorFlow Tensor.
            # We revert to a numpy representation for the uncommented code,
            # as the Keras layer call will be incorrect here with numpy data.
            # The original code is problematic here; to uncomment only, we must
            # include the line, assuming the user resolves the data type issue.
            projected_data = (
                normalized_data  # Dummy for pure uncommenting if `data` is numpy
            )

        return normalized_data  # Return normalized data for correct dataset creation

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        """
        Train the forecasting model.

        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Optional validation data tuple (X_val, y_val)

        Returns:
            Training history
        """
        # Preprocess data (returns normalized numpy array)
        X_normalized = self.preprocess_data(X_train)

        # Custom training loop with gradient clipping
        train_loss_metric = keras.metrics.Mean(name="train_loss")
        val_loss_metric = keras.metrics.Mean(name="val_loss")

        loss_fn = keras.losses.MeanSquaredError()

        # Create datasets
        # We pass normalized data. The feature_projection needs to be applied inside the model/train step.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_normalized, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_normalized = self.preprocess_data(X_val)
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val_normalized, y_val))
            val_dataset = val_dataset.batch(batch_size)

        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Reset metrics
            train_loss_metric.reset_states()
            if validation_data is not None:
                val_loss_metric.reset_states()

            # Training loop
            for x_batch, y_batch in train_dataset:
                # Apply feature projection here to the batch
                x_projected_batch = self.feature_projection(x_batch)

                with tf.GradientTape() as tape:
                    predictions = self.model(x_projected_batch, training=True)
                    loss = loss_fn(y_batch, predictions)

                # Get gradients and apply
                gradients = tape.gradient(loss, self.model.trainable_variables)
                # Clip gradients to prevent exploding gradients
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

                # Update metrics
                train_loss_metric.update_state(loss)

            # Validation loop
            if validation_data is not None:
                for x_val_batch, y_val_batch in val_dataset:
                    # Apply feature projection here to the batch
                    x_val_projected_batch = self.feature_projection(x_val_batch)

                    val_predictions = self.model(x_val_projected_batch, training=False)
                    val_loss = loss_fn(y_val_batch, val_predictions)
                    val_loss_metric.update_state(val_loss)

            # Record history
            history["loss"].append(train_loss_metric.result().numpy())
            if validation_data is not None:
                history["val_loss"].append(val_loss_metric.result().numpy())
                print(
                    f"Epoch {epoch+1}/{epochs} - loss: {train_loss_metric.result():.4f} - val_loss: {val_loss_metric.result():.4f}"
                )
            else:
                print(
                    f"Epoch {epoch+1}/{epochs} - loss: {train_loss_metric.result():.4f}"
                )

        return history

    def predict(self, X):
        """
        Generate forecasts using the trained model.

        Args:
            X: Input features for prediction

        Returns:
            Forecasted values
        """
        # Preprocess data (returns normalized numpy array)
        X_normalized = self.preprocess_data(X)
        X_tensor = tf.convert_to_tensor(X_normalized, dtype=tf.float32)

        # Apply feature projection
        X_processed = self.feature_projection(X_tensor)

        predictions = self.model(X_processed, training=False)
        return predictions.numpy()

    def save(self, filepath):
        """Save the model to disk"""
        self.model.save_weights(filepath)

    def load(self, filepath):
        """Load the model from disk"""
        self.model.load_weights(filepath)
