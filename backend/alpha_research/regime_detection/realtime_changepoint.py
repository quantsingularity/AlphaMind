# import numpy as np
# from ruptures import Binseg


# class RealTimeRegimeDetector:
#     def __init__(self, model=Binseg(), min_size=24, jump=5):
#         self.model = model
#         self.min_size = min_size
#         self.jump = jump
#         self.history = []

#     def update(self, new_data):
#         self.history.extend(new_data)
#         if len(self.history) < self.min_size * 2:
#             return []

#         self.model.fit(np.array(self.history))
#         change_points = self.model.predict(pen=3)
#         return [
#             self.history[cp]
#             for cp in change_points
#             if cp > len(self.history) - self.jump
        ]
