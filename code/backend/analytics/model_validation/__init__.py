"""
AlphaMind Model Validation Framework

Comprehensive tools for validating machine learning models in financial
applications: cross-validation, metrics, model comparison, and explainability.
"""

from analytics.model_validation.comparison import ModelComparison, StatisticalTests
from analytics.model_validation.cross_validation import (
    BlockingTimeSeriesSplit,
    PurgedKFold,
    TimeSeriesSplit,
)
from analytics.model_validation.explainability import (
    FeatureImportance,
    PartialDependence,
    PermutationImportance,
    ShapExplainer,
)
from analytics.model_validation.metrics import (
    calculate_metrics,
    calmar_ratio,
    information_coefficient,
    maximum_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from analytics.model_validation.validation_report import ValidationReport

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
