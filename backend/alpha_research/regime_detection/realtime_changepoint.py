import numpy as np
from ruptures import Binseg
from ruptures.costs import CostL2
from ruptures.costs import CostRbf  # Added a common cost function


class OnlineChangeDetector:
    """
    Implements a real-time (online) change point detection mechanism
    using the Binseg algorithm from the 'ruptures' library.
    This is suitable for detecting sudden shifts in financial time series regimes.
    """

    def __init__(self, model_cost="l2", min_size=24, jump=5, penalty=3.0):
        """
        Initializes the change point detector.

        Args:
            model_cost (str): The cost function to use ('l2' for least squares, 'rbf' for kernel-based).
            min_size (int): Minimum size of a segment (number of data points).
            jump (int): Subsample step size (e.g., jump=5 means checking every 5th point).
            penalty (float): Penalty value to control the number of detected change points.
        """
        if model_cost == "rbf":
            self.model = Binseg(
                model="rbf", custom_cost=CostRbf(), jump=jump, min_size=min_size
            )
        else:
            self.model = Binseg(
                model="l2", custom_cost=CostL2(), jump=jump, min_size=min_size
            )

        self.min_size = min_size
        self.jump = jump
        self.penalty = penalty
        self.history = []
        self.last_change_point_idx = 0  # Index in history where the last regime started

    def update(self, new_data_point: float) -> list:
        """
        Updates the internal history with a new data point and checks for a new regime change.

        Args:
            new_data_point (float): The single new data point (e.g., daily return or volatility).

        Returns:
            list: A list of absolute change point indices (relative to the start of history)
                  if a change is detected. Returns an empty list otherwise.
        """
        # 1. Update history
        self.history.append(new_data_point)
        current_len = len(self.history)

        # 2. Check minimum size requirement
        if current_len < self.min_size * 2:
            return []

        # 3. Fit the model to the *relevant* data segment (from the last change point onwards)
        # This speeds up the real-time detection significantly
        detection_segment = self.history[self.last_change_point_idx :]

        # We need a numpy array for the ruptures library
        detection_array = np.array(detection_segment).reshape(-1, 1)

        # 4. Detect change points
        self.model.fit(detection_array)

        # Predict returns the *end indices* of the segments (relative to detection_segment)
        change_points_relative = self.model.predict(pen=self.penalty)

        # The last index returned is always the end of the data, so we ignore it.
        # Check if any new change points were found in the recent data.
        if len(change_points_relative) > 1:
            # New change points found. We only care about the latest one.
            # change_points_relative[-2] is the index of the second to last segment end
            # which marks the start of the current, new regime.

            # Convert the relative index back to the absolute index in self.history
            new_regime_end_index_relative = change_points_relative[-2]

            # The change point index is relative to detection_array
            absolute_change_index = (
                self.last_change_point_idx + new_regime_end_index_relative
            )

            # 5. Filter for recent changes: only trigger if the change is near the end
            # We want to confirm the change within the last `self.min_size` points.
            if absolute_change_index > current_len - self.min_size:
                # Update the last change point index to the start of the new regime
                self.last_change_point_idx = absolute_change_index

                return [absolute_change_index]  # Return the index of the detected shift

        return []

    def get_full_segmentation(self):
        """
        Returns all detected change points across the entire history.
        This is typically for offline analysis or visualization.

        Returns:
            list: Absolute change point indices (segment end indices) from the start of history.
        """
        if len(self.history) < self.min_size:
            return []

        full_array = np.array(self.history).reshape(-1, 1)
        self.model.fit(full_array)
        change_points = self.model.predict(pen=self.penalty)

        # Ruptures returns indices (including the end of the data).
        # It should return a list of indices where a change occurs (end of segments).
        return change_points
