"""
AlphaMind Model Validation Framework

This package provides comprehensive tools for validating machine learning models
in financial applications, including cross-validation strategies, performance metrics,
model comparison utilities, and explainability tools.
"""

from .cross_validation import TimeSeriesSplit, BlockingTimeSeriesSplit, PurgedKFold
from .metrics import (
    calculate_metrics, 
    sharpe_ratio, 
    sortino_ratio, 
    calmar_ratio, 
    maximum_drawdown,
    information_coefficient
)
from .comparison import ModelComparison, StatisticalTests
from .explainability import (
    FeatureImportance, 
    PartialDependence, 
    ShapExplainer,
    PermutationImportance
)
from .validation_report import ValidationReport

__all__ = [
    'TimeSeriesSplit', 'BlockingTimeSeriesSplit', 'PurgedKFold',
    'calculate_metrics', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
    'maximum_drawdown', 'information_coefficient',
    'ModelComparison', 'StatisticalTests',
    'FeatureImportance', 'PartialDependence', 'ShapExplainer', 'PermutationImportance',
    'ValidationReport'
]
