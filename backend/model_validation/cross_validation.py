#""""""
## Cross-validation strategies specifically designed for financial time series data.
#
## This module provides specialized cross-validation techniques that account for
## the temporal nature of financial data, including:
## - Time series splits with proper train/validation/test separation
## - Blocking time series cross-validation to handle temporal dependencies
## - Purged k-fold cross-validation to prevent data leakage
#""""""

# from typing import Iterator, List, Optional, Tuple, Union

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit


# class TimeSeriesSplit:
#    """"""
##     Enhanced time series cross-validation with proper train/validation/test separation
##     and embargo periods to prevent data leakage in financial applications.
#
##     This implementation extends scikit-learn's TimeSeriesSplit with additional
##     features for financial time series data.
#
##     Parameters
#    ----------
##     n_splits : int
##         Number of splits. Must be at least 2.
##     max_train_size : int, optional
##         Maximum size for a single training set.
##     test_size : int, optional
##         Number of samples in each test set.
##     gap : int, optional
##         Number of samples to exclude from the end of each train set before the test set.
##     embargo : int, optional
##         Number of samples to exclude from the beginning of each test set after the train set.
#    """"""

#     def __init__(
#         self,
#         n_splits: int = 5,
#         max_train_size: Optional[int] = None,
#         test_size: Optional[int] = None,
#         gap: int = 0,
#         embargo: int = 0,
    ):
#         self.n_splits = n_splits
#         self.max_train_size = max_train_size
#         self.test_size = test_size
#         self.gap = gap
#         self.embargo = embargo
#         self._sklearn_tscv = SklearnTimeSeriesSplit(
#             n_splits=n_splits,
#             max_train_size=max_train_size,
#             test_size=test_size,
#             gap=gap,
        )

#     def split(self, X, y=None, groups=None):
#        """"""
##         Generate indices to split data into training and test set.
#
##         Parameters
#        ----------
##         X : array-like of shape (n_samples, n_features)
##             Training data, where n_samples is the number of samples
##             and n_features is the number of features.
##         y : array-like of shape (n_samples,), optional
##             The target variable for supervised learning problems.
##         groups : array-like of shape (n_samples,), optional
##             Group labels for the samples used while splitting the dataset.
#
##         Yields
#        ------
##         train : ndarray
##             The training set indices for that split.
##         test : ndarray
##             The testing set indices for that split.
#        """"""
#         for train_idx, test_idx in self._sklearn_tscv.split(X, y, groups):
#             if self.embargo > 0 and len(test_idx) > self.embargo:
#                 test_idx = test_idx[self.embargo :]
#             yield train_idx, test_idx

#     def get_n_splits(self, X=None, y=None, groups=None):
#        """"""
##         Returns the number of splitting iterations in the cross-validator.
#
##         Parameters
#        ----------
##         X : object, optional
##             Always ignored, exists for compatibility.
##         y : object, optional
##             Always ignored, exists for compatibility.
##         groups : object, optional
##             Always ignored, exists for compatibility.
#
##         Returns
#        -------
##         n_splits : int
##             Returns the number of splitting iterations in the cross-validator.
#        """"""
#         return self.n_splits


# class BlockingTimeSeriesSplit:
#    """"""
##     Blocking time series cross-validation to handle temporal dependencies.
#
##     This approach divides the time series into blocks and ensures that
##     blocks used for training are separate from those used for testing,
##     which helps prevent data leakage in financial applications.
#
##     Parameters
#    ----------
##     n_splits : int
##         Number of splits. Must be at least 2.
##     block_size : int
##         Size of each block in number of samples.
##     train_blocks : int
##         Number of blocks to use for training in each split.
##     test_blocks : int
##         Number of blocks to use for testing in each split.
##     gap_blocks : int, optional
##         Number of blocks to exclude between train and test sets.
#    """"""

#     def __init__(
#         self,
#         n_splits: int = 5,
#         block_size: int = 50,
#         train_blocks: int = 10,
#         test_blocks: int = 2,
#         gap_blocks: int = 1,
    ):
#         self.n_splits = n_splits
#         self.block_size = block_size
#         self.train_blocks = train_blocks
#         self.test_blocks = test_blocks
#         self.gap_blocks = gap_blocks

#     def split(self, X, y=None, groups=None):
#        """"""
##         Generate indices to split data into training and test set.
#
##         Parameters
#        ----------
##         X : array-like of shape (n_samples, n_features)
##             Training data, where n_samples is the number of samples
##             and n_features is the number of features.
##         y : array-like of shape (n_samples,), optional
##             The target variable for supervised learning problems.
##         groups : array-like of shape (n_samples,), optional
##             Group labels for the samples used while splitting the dataset.
#
##         Yields
#        ------
##         train : ndarray
##             The training set indices for that split.
##         test : ndarray
##             The testing set indices for that split.
#        """"""
#         n_samples = len(X)
#         n_blocks = n_samples // self.block_size

        # Ensure we have enough blocks for the requested splits
#         if n_blocks < self.train_blocks + self.gap_blocks + self.test_blocks:
#             raise ValueError(
#                 f"Not enough blocks ({n_blocks}) for the requested "
#                 f"train_blocks ({self.train_blocks}), gap_blocks ({self.gap_blocks}), "
#                 f"and test_blocks ({self.test_blocks})."
            )

        # Calculate the maximum number of splits possible
#         max_splits = (
#             n_blocks - self.train_blocks - self.gap_blocks - self.test_blocks + 1
        )
#         n_splits = min(self.n_splits, max_splits)

        # Calculate step size to distribute splits evenly
#         if n_splits > 1:
#             step = (max_splits - 1) // (n_splits - 1)
#         else:
#             step = 1

#         for i in range(n_splits):
            # Calculate the starting block for this split
#             start_block = i * step

            # Calculate train and test indices
#             train_start = start_block * self.block_size
#             train_end = (start_block + self.train_blocks) * self.block_size

#             test_start = (
#                 start_block + self.train_blocks + self.gap_blocks
#             ) * self.block_size
#             test_end = (
#                 start_block + self.train_blocks + self.gap_blocks + self.test_blocks
#             ) * self.block_size

            # Ensure we don't exceed the data length
#             test_end = min(test_end, n_samples)

#             train_indices = np.arange(train_start, train_end)
#             test_indices = np.arange(test_start, test_end)

#             yield train_indices, test_indices

#     def get_n_splits(self, X=None, y=None, groups=None):
#        """"""
##         Returns the number of splitting iterations in the cross-validator.
#
##         Parameters
#        ----------
##         X : object, optional
##             Always ignored, exists for compatibility.
##         y : object, optional
##             Always ignored, exists for compatibility.
##         groups : object, optional
##             Always ignored, exists for compatibility.
#
##         Returns
#        -------
##         n_splits : int
##             Returns the number of splitting iterations in the cross-validator.
#        """"""
#         return self.n_splits


# class PurgedKFold:
#    """"""
##     Purged K-Fold cross-validation for financial time series.
#
##     This implementation prevents data leakage by purging samples in the test set
##     that overlap with samples in the training set based on time information.
#
##     Parameters
#    ----------
##     n_splits : int
##         Number of folds. Must be at least 2.
##     purge_overlap : bool, default=True
##         Whether to purge overlapping samples between train and test sets.
##     embargo : float, default=0.0
##         Ratio of test samples to embargo after training samples.
#    """"""

#     def __init__(
#         self, n_splits: int = 5, purge_overlap: bool = True, embargo: float = 0.0
    ):
#         self.n_splits = n_splits
#         self.purge_overlap = purge_overlap
#         self.embargo = embargo
#         self._kf = KFold(n_splits=n_splits, shuffle=False)

#     def split(self, X, y=None, groups=None, times=None):
#        """"""
##         Generate indices to split data into training and test set.
#
##         Parameters
#        ----------
##         X : array-like of shape (n_samples, n_features)
##             Training data, where n_samples is the number of samples
##             and n_features is the number of features.
##         y : array-like of shape (n_samples,), optional
##             The target variable for supervised learning problems.
##         groups : array-like of shape (n_samples,), optional
##             Group labels for the samples used while splitting the dataset.
##         times : array-like of shape (n_samples,), optional
##             Timestamps for each sample. Required for purging and embargo.
#
##         Yields
#        ------
##         train : ndarray
##             The training set indices for that split.
##         test : ndarray
##             The testing set indices for that split.
#        """"""
#         if times is None and (self.purge_overlap or self.embargo > 0):
#             raise ValueError(
                "times must be provided when purge_overlap=True or embargo > 0"
            )

#         if times is not None:
#             times = pd.Series(times)

#         for train_idx, test_idx in self._kf.split(X, y, groups):
#             if self.purge_overlap and times is not None:
#                 train_times = times.iloc[train_idx]
#                 test_times = times.iloc[test_idx]

                # Find test samples that overlap with train samples
#                 train_start, train_end = train_times.min(), train_times.max()
#                 overlapping = (test_times >= train_start) & (test_times <= train_end)

                # Remove overlapping samples from test set
#                 test_idx = test_idx[~overlapping.values]

#             if self.embargo > 0 and times is not None:
#                 train_times = times.iloc[train_idx]
#                 test_times = times.iloc[test_idx]

                # Calculate embargo size
#                 embargo_size = int(len(test_idx) * self.embargo)

#                 if embargo_size > 0:
                    # Sort test indices by time
#                     sorted_test_idx = test_idx[np.argsort(test_times.values)]

                    # Apply embargo by removing samples closest to train set
#                     test_idx = sorted_test_idx[embargo_size:]

#             yield train_idx, test_idx

#     def get_n_splits(self, X=None, y=None, groups=None):
#        """"""
##         Returns the number of splitting iterations in the cross-validator.
#
##         Parameters
#        ----------
##         X : object, optional
##             Always ignored, exists for compatibility.
##         y : object, optional
##             Always ignored, exists for compatibility.
##         groups : object, optional
##             Always ignored, exists for compatibility.
#
##         Returns
#        -------
##         n_splits : int
##             Returns the number of splitting iterations in the cross-validator.
#        """"""
#         return self.n_splits
