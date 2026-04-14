"""
AlphaMind Model Validation Framework

This package provides comprehensive tools for validating machine learning models
in financial applications, including cross-validation strategies, performance metrics,
model comparison utilities, and explainability tools.
"""

from model_validation.comparison import ModelComparison, StatisticalTests
from model_validation.cross_validation import (
    BlockingTimeSeriesSplit,
    PurgedKFold,
    TimeSeriesSplit,
)
from model_validation.explainability import (
    FeatureImportance,
    PartialDependence,
    PermutationImportance,
    ShapExplainer,
)
from model_validation.metrics import (
    calculate_metrics,
    calmar_ratio,
    information_coefficient,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from model_validation.validation_report import ValidationReport

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
