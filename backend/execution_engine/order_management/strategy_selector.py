"""
## Execution Strategy Selection Framework.
#
## This module provides functionality for selecting and managing execution strategies
## based on market conditions, order characteristics, and performance metrics.
"""

from dataclasses import dataclass, field
import datetime
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple


# Configure logging
logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Types of execution algorithms available."""

    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"
    DARK_POOL = "dark_pool"
    ICEBERG = "iceberg"
    SNIPER = "sniper"
    PERCENTAGE_OF_VOLUME = "percentage_of_volume"


@dataclass
class MarketCondition:
    """Market condition metrics used for strategy selection."""

    volatility: float
    spread: float
    depth: float
    volume: float
    momentum: float
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Performance metrics for execution strategies."""

    strategy: ExecutionAlgorithm
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    opportunity_cost: float
    total_cost: float
    fill_rate: float
    execution_time: float
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    additional_metrics: Dict[str, float] = field(default_factory=dict)


class StrategySelector:
    """Selects optimal execution strategies based on market conditions and order characteristics."""

    def __init__(self):
        """Initialize strategy selector."""
        self.strategies = {}
        self.historical_performance = {}
        self.market_condition_thresholds = {}
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register default execution strategies."""
        for strategy in ExecutionAlgorithm:
            self.strategies[strategy] = {
                "name": strategy.value,
                "description": f"Default {strategy.value} strategy",
                "enabled": True,
                "parameters": {},
                "suitability_function": self._default_suitability_function,
            }

    def _default_suitability_function(
        self, order_size: float, market_condition: MarketCondition
    ) -> float:
        """
        Default function to calculate strategy suitability score.

        Args:
            order_size: Size of the order
            market_condition: Current market conditions

        Returns:
            Suitability score (0-100)
        """
        # Default implementation returns 50 (neutral score)
        return 50.0

    def register_strategy(
        self,
        strategy: ExecutionAlgorithm,
        description: str,
        suitability_function: Callable,
        parameters: Dict[str, Any] = None,
        enabled: bool = True,
    ) -> None:
        """
        Register a custom execution strategy.

        Args:
            strategy: Strategy type
            description: Strategy description
            suitability_function: Function to calculate suitability score
            parameters: Strategy parameters
            enabled: Whether the strategy is enabled
        """
        self.strategies[strategy] = {
            "name": strategy.value,
            "description": description,
            "enabled": enabled,
            "parameters": parameters or {},
            "suitability_function": suitability_function,
        }
        logger.info(f"Registered execution strategy: {strategy.value}")

    def set_market_condition_thresholds(
        self, thresholds: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Set thresholds for market condition metrics.

        Args:
            thresholds: Dictionary mapping metric names to dictionaries of thresholds
                        (e.g., {"volatility": {"low": 0.1, "medium": 0.3, "high": 0.5}})
        """
        self.market_condition_thresholds = thresholds
        logger.info(f"Set market condition thresholds: {thresholds}")

    def record_strategy_performance(self, metrics: ExecutionMetrics) -> None:
        """
        Record performance metrics for a strategy.

        Args:
            metrics: ExecutionMetrics object
        """
        strategy = metrics.strategy

        if strategy not in self.historical_performance:
            self.historical_performance[strategy] = []

        self.historical_performance[strategy].append(metrics)
        logger.info(f"Recorded performance metrics for {strategy.value} strategy")

    def get_strategy_performance(
        self,
        strategy: ExecutionAlgorithm,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> List[ExecutionMetrics]:
        """
        Get historical performance metrics for a strategy.

        Args:
            strategy: Strategy to get metrics for
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering

        Returns:
            List of ExecutionMetrics objects
        """
        if strategy not in self.historical_performance:
            return []

        metrics = self.historical_performance[strategy]

        if start_time is None and end_time is None:
            return metrics

        filtered = []
        for m in metrics:
            if start_time and m.timestamp < start_time:
                continue
            if end_time and m.timestamp > end_time:
                continue
            filtered.append(m)

        return filtered

    def calculate_strategy_scores(
        self, order_size: float, market_condition: MarketCondition
    ) -> Dict[ExecutionAlgorithm, float]:
        """
        Calculate suitability scores for all strategies.

        Args:
            order_size: Size of the order
            market_condition: Current market conditions

        Returns:
            Dictionary mapping strategies to suitability scores
        """
        scores = {}

        for strategy, config in self.strategies.items():
            if not config["enabled"]:
                continue

            try:
                score = config["suitability_function"](order_size, market_condition)
                scores[strategy] = score
            except Exception as e:
                logger.error(
                    f"Error calculating suitability score for {strategy.value}: {str(e)}"
                )
                scores[strategy] = 0.0

        return scores

    def select_best_strategy(
        self, order_size: float, market_condition: MarketCondition
    ) -> Tuple[ExecutionAlgorithm, float]:
        """
        Select the best execution strategy based on current conditions.

        Args:
            order_size: Size of the order
            market_condition: Current market conditions

        Returns:
            Tuple of (best_strategy, score)
        """
        scores = self.calculate_strategy_scores(order_size, market_condition)

        if not scores:
            # Default to MARKET if no strategies are available
            return ExecutionAlgorithm.MARKET, 0.0

        best_strategy = max(scores.items(), key=lambda x: x[1])
        return best_strategy[0], best_strategy[1]

    def get_strategy_parameters(self, strategy: ExecutionAlgorithm) -> Dict[str, Any]:
        """
        Get parameters for a specific strategy.

        Args:
            strategy: Strategy to get parameters for

        Returns:
            Dictionary of strategy parameters

        Raises:
            KeyError: If the strategy doesn't exist
        """
        if strategy not in self.strategies:
            raise KeyError(f"Strategy {strategy.value} not registered")

        return self.strategies[strategy]["parameters"].copy()

    def update_strategy_parameters(
        self, strategy: ExecutionAlgorithm, parameters: Dict[str, Any]
    ) -> None:
        """
        Update parameters for a specific strategy.

        Args:
            strategy: Strategy to update
            parameters: New parameters

        Raises:
            KeyError: If the strategy doesn't exist
        """
        if strategy not in self.strategies:
            raise KeyError(f"Strategy {strategy.value} not registered")

        self.strategies[strategy]["parameters"].update(parameters)
        logger.info(f"Updated parameters for {strategy.value} strategy")

    def enable_strategy(self, strategy: ExecutionAlgorithm) -> None:
        """
        Enable a strategy.

        Args:
            strategy: Strategy to enable

        Raises:
            KeyError: If the strategy doesn't exist
        """
        if strategy not in self.strategies:
            raise KeyError(f"Strategy {strategy.value} not registered")

        self.strategies[strategy]["enabled"] = True
        logger.info(f"Enabled {strategy.value} strategy")

    def disable_strategy(self, strategy: ExecutionAlgorithm) -> None:
        """
        Disable a strategy.

        Args:
            strategy: Strategy to disable

        Raises:
            KeyError: If the strategy doesn't exist
        """
        if strategy not in self.strategies:
            raise KeyError(f"Strategy {strategy.value} not registered")

        self.strategies[strategy]["enabled"] = False
        logger.info(f"Disabled {strategy.value} strategy")

    def get_market_condition_classification(
        self, market_condition: MarketCondition
    ) -> Dict[str, str]:
        """
        Classify market conditions based on thresholds.

        Args:
            market_condition: Current market conditions

        Returns:
            Dictionary mapping metric names to classifications (e.g., {"volatility": "high"})
        """
        classification = {}

        # Process standard metrics
        for metric in ["volatility", "spread", "depth", "volume", "momentum"]:
            value = getattr(market_condition, metric)
            classification[metric] = self._classify_metric(metric, value)

        # Process additional metrics
        for metric, value in market_condition.additional_metrics.items():
            classification[metric] = self._classify_metric(metric, value)

        return classification

    def _classify_metric(self, metric: str, value: float) -> str:
        """
        Classify a metric value based on thresholds.

        Args:
            metric: Metric name
            value: Metric value

        Returns:
            Classification (e.g., "low", "medium", "high")
        """
        if metric not in self.market_condition_thresholds:
            return "unknown"

        # Get thresholds for the metric
        thresholds = self.market_condition_thresholds[metric]

        # Sort thresholds by value to classify correctly (e.g., low <= 0.1, medium <= 0.3)
        # Note: The expected structure for thresholds should be designed such that
        # higher values correspond to higher classifications. E.g., for volatility,
        # {"low": 0.1, "medium": 0.3} means 'low' if value <= 0.1, 'medium' if value <= 0.3,
        # and 'high' (or the next level) otherwise.

        # We need to sort by the threshold value itself (x[1])
        sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1])

        for classification, threshold in sorted_thresholds:
            if value <= threshold:
                return classification

        # If the value is higher than all defined thresholds,
        # return the classification corresponding to the highest threshold value.
        # This assumes the highest key in the sorted list represents the "boundary" for the next level.
        # To be precise, if the structure is {"low": 0.1, "medium": 0.3, "high": 0.5}, and value is 0.6,
        # it should be considered "high". The implementation below handles this by:
        # 1. If value is 0.05, it hits "low" (<=0.1) and returns.
        # 2. If value is 0.2, it misses "low", hits "medium" (<=0.3) and returns.
        # 3. If value is 0.6, it misses all. The final return should be the classification
        #    corresponding to the highest band (e.g., "high" if 0.5 was the boundary to "high").

        # Given the logic above, if the loop finishes, the value is greater than all thresholds.
        # We should return the highest classification label available (which is the last one in the sorted list).
        return sorted_thresholds[-1][0] if sorted_thresholds else "above_all_thresholds"
