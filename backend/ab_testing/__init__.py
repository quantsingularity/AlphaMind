"""
AlphaMind A/B Testing Framework

This package provides a comprehensive framework for conducting A/B tests
in financial applications, including experiment tracking, statistical analysis,
visualization tools, and experiment configuration management.
"""

from ab_testing.config import ExperimentConfig
from ab_testing.experiment import Experiment, ExperimentGroup
from ab_testing.statistics import (
    BayesianABTest,
    MannWhitneyU,
    MultipleTestingCorrection,
    StatisticalTest,
    TTest,
)
from ab_testing.tracking import ExperimentResult, ExperimentTracker
from ab_testing.visualization import ExperimentVisualizer

__all__ = [
    "Experiment",
    "ExperimentGroup",
    "ExperimentTracker",
    "ExperimentResult",
    "StatisticalTest",
    "TTest",
    "MannWhitneyU",
    "BayesianABTest",
    "MultipleTestingCorrection",
    "ExperimentVisualizer",
    "ExperimentConfig",
]
