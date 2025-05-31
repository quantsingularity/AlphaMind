"""
AlphaMind A/B Testing Framework

This package provides a comprehensive framework for conducting A/B tests
in financial applications, including experiment tracking, statistical analysis,
visualization tools, and experiment configuration management.
"""

from .experiment import Experiment, ExperimentGroup
from .tracking import ExperimentTracker, ExperimentResult
from .statistics import (
    StatisticalTest, 
    TTest, 
    MannWhitneyU, 
    BayesianABTest,
    MultipleTestingCorrection
)
from .visualization import ExperimentVisualizer
from .config import ExperimentConfig

__all__ = [
    'Experiment', 'ExperimentGroup',
    'ExperimentTracker', 'ExperimentResult',
    'StatisticalTest', 'TTest', 'MannWhitneyU', 'BayesianABTest', 'MultipleTestingCorrection',
    'ExperimentVisualizer',
    'ExperimentConfig'
]
