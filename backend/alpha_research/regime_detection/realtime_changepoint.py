from typing import Any
import numpy as np
from ruptures import Binseg
from ruptures.costs import CostL2
from ruptures.costs import CostRbf


class OnlineChangeDetector:
    """
    Implements a real-time (online) change point detection mechanism
    using the Binseg algorithm from the 'ruptures' library.
    This is suitable for detecting sudden shifts in financial time series regimes.
    """

    def __init__(
        self,
        model_cost: Any = "l2",
        min_size: Any = 24,
        jump: Any = 5,
        penalty: Any = 3.0,
    ) -> None:
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
        self.last_change_point_idx = 0

    def update(self, new_data_point: float) -> list:
        """
        Updates the internal history with a new data point and checks for a new regime change.

        Args:
            new_data_point (float): The single new data point (e.g., daily return or volatility).

        Returns:
            list: A list of absolute change point indices (relative to the start of history)
                  if a change is detected. Returns an empty list otherwise.
        """
        self.history.append(new_data_point)
        current_len = len(self.history)
        if current_len < self.min_size * 2:
            return []
        detection_segment = self.history[self.last_change_point_idx :]
        detection_array = np.array(detection_segment).reshape(-1, 1)
        self.model.fit(detection_array)
        change_points_relative = self.model.predict(pen=self.penalty)
        if len(change_points_relative) > 1:
            new_regime_end_index_relative = change_points_relative[-2]
            absolute_change_index = (
                self.last_change_point_idx + new_regime_end_index_relative
            )
            if absolute_change_index > current_len - self.min_size:
                self.last_change_point_idx = absolute_change_index
                return [absolute_change_index]
        return []

    def get_full_segmentation(self) -> Any:
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
        return change_points
