"""Risk Aggregation Module for AlphaMind.

This module provides comprehensive risk aggregation functionality across portfolios,
including position risk calculation, portfolio-level risk metrics, and
risk limit monitoring.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class RiskAggregator:
    """
    Manages and aggregates various risk metrics from different sources
    (e.g., market risk, credit risk, liquidity risk).
    """

    def __init__(self):
        self.aggregated_data: Dict[str, Any] = {}
        logger.info("RiskAggregator initialized.")

    def aggregate_metrics(
        self, risk_reports: List[Dict[str, Union[float, str]]]
    ) -> Dict[str, Any]:
        """
        Combines risk metrics from a list of risk reports.

        Args:
            risk_reports: A list of dictionaries, where each dictionary
                          represents a risk report (e.g., from a single asset or desk).

        Returns:
            A dictionary containing the aggregated risk metrics.
        """
        if not risk_reports:
            return {"total_risk": 0.0, "count": 0}

        # Simple aggregation: sum up a 'VaR' metric if present
        total_var = sum(report.get("VaR", 0.0) for report in risk_reports)

        self.aggregated_data = {
            "total_var": total_var,
            "number_of_reports": len(risk_reports),
            "last_aggregated_time": pd.Timestamp.now().isoformat(),
        }
        logger.info(
            f"Aggregated {len(risk_reports)} reports. Total VaR: {total_var:.2f}"
        )
        return self.aggregated_data


def calculate_portfolio_var(
    portfolio_returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.99
) -> float:
    """
    Calculates the historical Value-at-Risk (VaR) for a portfolio.

    Args:
        portfolio_returns: A pandas Series or numpy array of historical portfolio returns.
        confidence_level: The confidence level for the VaR calculation (e.g., 0.99 for 99% VaR).

    Returns:
        The calculated VaR value (a positive number representing the loss).
    """
    if (
        not isinstance(portfolio_returns, (pd.Series, np.ndarray))
        or len(portfolio_returns) == 0
    ):
        logger.error("Invalid or empty portfolio returns provided for VaR calculation.")
        return 0.0

    # Ensure returns are a numpy array for quantile calculation
    returns_array = np.asarray(portfolio_returns)

    # Calculate the percentile corresponding to the confidence level
    # For 99% VaR, we look at the 1st percentile (1 - 0.99 = 0.01)
    percentile = 1.0 - confidence_level

    # VaR is the negative of the percentile of the returns distribution
    # We use interpolation='lower' to be conservative
    var_value = -np.quantile(returns_array, percentile, interpolation="lower")

    # VaR is typically reported as a positive loss value
    return max(0.0, var_value)
