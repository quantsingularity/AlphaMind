"""
AlphaMind Model Validation Framework

This package provides comprehensive tools for validating machine learning models
in financial applications, including cross-validation strategies, performance metrics,
model comparison utilities, and explainability tools.
"""

from .comparison import ModelComparison, StatisticalTests
from .cross_validation import BlockingTimeSeriesSplit, PurgedKFold, TimeSeriesSplit
from .explainability import (
    FeatureImportance,
    PartialDependence,
    PermutationImportance,
    ShapExplainer,
)
from .metrics import (
    calculate_metrics,
    calmar_ratio,
    information_coefficient,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from .validation_report import ValidationReport

__all__ = [
    "TimeSeriesSplit",
    "BlockingTimeSeriesSplit",
    "PurgedKFold",
    "calculate_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "maximum_drawdown",
    "information_coefficient",
    "ModelComparison",
    "StatisticalTests",
    "FeatureImportance",
    "PartialDependence",
    "ShapExplainer",
    "PermutationImportance",
    "ValidationReport",
]
