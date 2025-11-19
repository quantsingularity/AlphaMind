#""""""
## Advanced Risk Controls Module for AlphaMind
#
## This module provides comprehensive risk management tools for trading strategies,
## including position sizing, stop-loss mechanisms, exposure limits, and risk metrics.
#""""""

# import asyncio
# from datetime import datetime, timedelta
# from enum import Enum
# import json
# import logging
# import math
# import os
# import time
# from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# import numpy as np
# import pandas as pd

# Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# logger = logging.getLogger(__name__)


# class RiskLevel(Enum):
#    """Risk level for trading strategies."""
#
##     LOW = "low"
##     MEDIUM = "medium"
##     HIGH = "high"
##     CUSTOM = "custom"
#
#
## class StopLossType(Enum):
#    """Types of stop-loss mechanisms."""

#     FIXED = "fixed"  # Fixed price
#     PERCENT = "percent"  # Percentage from entry
#     ATR = "atr"  # Average True Range multiple
#     TRAILING = "trailing"  # Trailing stop
#     VOLATILITY = "volatility"  # Volatility-based
#     TIME = "time"  # Time-based stop
#     CUSTOM = "custom"  # Custom stop-loss logic


# class TakeProfitType(Enum):
#    """Types of take-profit mechanisms."""
#
##     FIXED = "fixed"  # Fixed price
##     PERCENT = "percent"  # Percentage from entry
##     RISK_REWARD = "risk_reward"  # Risk-reward ratio
##     TRAILING = "trailing"  # Trailing take-profit
##     VOLATILITY = "volatility"  # Volatility-based
##     CUSTOM = "custom"  # Custom take-profit logic
#
#
## class PositionSizingMethod(Enum):
#    """Methods for position sizing."""

#     FIXED = "fixed"  # Fixed position size
#     PERCENT_EQUITY = "percent_equity"  # Percentage of equity
#     VOLATILITY = "volatility"  # Volatility-based sizing
#     KELLY = "kelly"  # Kelly criterion
#     OPTIMAL_F = "optimal_f"  # Optimal f
#     RISK_PARITY = "risk_parity"  # Risk parity
#     FIXED_RISK = "fixed_risk"  # Fixed risk per trade
#     CUSTOM = "custom"  # Custom position sizing logic


# class RiskMetric(Enum):
#    """Risk metrics for portfolio and strategy evaluation."""
#
##     VOLATILITY = "volatility"  # Standard deviation of returns
##     VAR = "var"  # Value at Risk
##     CVAR = "cvar"  # Conditional Value at Risk (Expected Shortfall)
##     DRAWDOWN = "drawdown"  # Maximum drawdown
##     SHARPE = "sharpe"  # Sharpe ratio
##     SORTINO = "sortino"  # Sortino ratio
##     CALMAR = "calmar"  # Calmar ratio
##     BETA = "beta"  # Beta to benchmark
##     CORRELATION = "correlation"  # Correlation to benchmark
##     ALPHA = "alpha"  # Alpha to benchmark
##     OMEGA = "omega"  # Omega ratio
##     TAIL_RATIO = "tail_ratio"  # Tail ratio
##     CUSTOM = "custom"  # Custom risk metric
#
#
## class RiskLimitType(Enum):
#    """Types of risk limits."""

#     POSITION_SIZE = "position_size"  # Maximum position size
#     EXPOSURE = "exposure"  # Maximum exposure
#     CONCENTRATION = "concentration"  # Maximum concentration in single asset
#     DRAWDOWN = "drawdown"  # Maximum drawdown
#     DAILY_LOSS = "daily_loss"  # Maximum daily loss
#     VAR = "var"  # Value at Risk limit
#     VOLATILITY = "volatility"  # Volatility limit
#     LEVERAGE = "leverage"  # Maximum leverage
#     CUSTOM = "custom"  # Custom risk limit


# class RiskLimit:
#    """Risk limit for trading strategies."""
#
##     def __init__(
##         self,
##         limit_type: RiskLimitType,
##         value: float,
##         action: str = "alert",
##         custom_action: Optional[Callable] = None,
#    ):
#        """"""
#         Initialize risk limit.

#         Args:
#             limit_type: Type of risk limit
#             value: Limit value
#             action: Action to take when limit is breached ("alert", "reduce", "close", "custom")
#             custom_action: Custom action function to call when limit is breached
#        """"""
##         self.limit_type = limit_type
##         self.value = value
##         self.action = action
##         self.custom_action = custom_action
##         self.is_breached = False
##         self.breach_time = None
##         self.breach_value = None
#
##     def check(self, current_value: float) -> bool:
#        """"""
#         Check if limit is breached.

#         Args:
#             current_value: Current value to check against limit

#         Returns:
#             True if limit is breached, False otherwise
#        """"""
#        # Different limit types have different comparison logic
##         if self.limit_type in [
##             RiskLimitType.POSITION_SIZE,
##             RiskLimitType.EXPOSURE,
##             RiskLimitType.CONCENTRATION,
##             RiskLimitType.DRAWDOWN,
##             RiskLimitType.DAILY_LOSS,
##             RiskLimitType.VAR,
##             RiskLimitType.VOLATILITY,
##             RiskLimitType.LEVERAGE,
#        ]:
#            # For these limits, breach occurs when current value exceeds limit
##             is_breached = abs(current_value) > self.value
##         else:
#            # For custom limits, use custom logic
##             is_breached = current_value > self.value
#
#        # Update breach status
##         if is_breached and not self.is_breached:
##             self.is_breached = True
##             self.breach_time = datetime.now()
##             self.breach_value = current_value
##         elif not is_breached and self.is_breached:
##             self.is_breached = False
##             self.breach_time = None
##             self.breach_value = None
#
##         return self.is_breached
#
##     def get_action(self) -> Tuple[str, Optional[Callable]]:
#        """"""
#         Get action to take when limit is breached.

#         Returns:
#             Tuple of action type and custom action function (if any)
#        """"""
##         return self.action, self.custom_action
#
#
## class StopLoss:
#    """Stop-loss mechanism for trading positions."""

#     def __init__(
#         self,
#         stop_type: StopLossType,
#         value: float,
#         is_trailing: bool = False,
#         time_window: Optional[timedelta] = None,
    ):
#        """"""
##         Initialize stop-loss.
#
##         Args:
##             stop_type: Type of stop-loss
##             value: Stop-loss value (interpretation depends on stop_type)
##             is_trailing: Whether stop-loss is trailing
##             time_window: Time window for time-based stop-loss
#        """"""
#         self.stop_type = stop_type
#         self.value = value
#         self.is_trailing = is_trailing
#         self.time_window = time_window
#         self.stop_price = None
#         self.highest_price = None
#         self.lowest_price = None
#         self.entry_time = None
#         self.entry_price = None
#         self.atr = None

#     def initialize(
#         self,
#         entry_price: float,
#         entry_time: datetime,
#         is_long: bool,
#         atr: Optional[float] = None,
    ):
#        """"""
##         Initialize stop-loss with entry information.
#
##         Args:
##             entry_price: Entry price
##             entry_time: Entry time
##             is_long: Whether position is long
##             atr: Average True Range (for ATR-based stop-loss)
#        """"""
#         self.entry_price = entry_price
#         self.entry_time = entry_time
#         self.highest_price = entry_price
#         self.lowest_price = entry_price
#         self.atr = atr

        # Calculate initial stop price
#         if self.stop_type == StopLossType.FIXED:
#             self.stop_price = self.value
#         elif self.stop_type == StopLossType.PERCENT:
#             if is_long:
#                 self.stop_price = entry_price * (1 - self.value)
#             else:
#                 self.stop_price = entry_price * (1 + self.value)
#         elif self.stop_type == StopLossType.ATR:
#             if atr is None:
#                 raise ValueError("ATR is required for ATR-based stop-loss")

#             if is_long:
#                 self.stop_price = entry_price - (atr * self.value)
#             else:
#                 self.stop_price = entry_price + (atr * self.value)
#         elif self.stop_type == StopLossType.TRAILING:
#             if is_long:
#                 self.stop_price = entry_price * (1 - self.value)
#             else:
#                 self.stop_price = entry_price * (1 + self.value)
#         elif self.stop_type == StopLossType.VOLATILITY:
            # Volatility-based stop-loss requires historical volatility
            # For now, use a simple approximation based on ATR
#             if atr is None:
#                 raise ValueError("ATR is required for volatility-based stop-loss")

#             if is_long:
#                 self.stop_price = entry_price - (atr * self.value)
#             else:
#                 self.stop_price = entry_price + (atr * self.value)
#         elif self.stop_type == StopLossType.TIME:
            # Time-based stop-loss doesn't have a price level
#             self.stop_price = None
#         elif self.stop_type == StopLossType.CUSTOM:
            # Custom stop-loss requires external logic
#             self.stop_price = None

#         logger.info(
#             f"Initialized stop-loss: type={self.stop_type.value}, price={self.stop_price}"
        )

#     def update(
#         self,
#         current_price: float,
#         current_time: datetime,
#         is_long: bool,
#         atr: Optional[float] = None,
#     ) -> bool:
#        """"""
##         Update stop-loss with current price information.
#
##         Args:
##             current_price: Current price
##             current_time: Current time
##             is_long: Whether position is long
##             atr: Current Average True Range (for ATR-based stop-loss)
#
##         Returns:
##             True if stop-loss is triggered, False otherwise
#        """"""
#         if self.entry_price is None:
#             raise ValueError("Stop-loss not initialized")

        # Update highest and lowest prices
#         self.highest_price = max(self.highest_price, current_price)
#         self.lowest_price = min(self.lowest_price, current_price)

        # Update ATR if provided
#         if atr is not None:
#             self.atr = atr

        # Update trailing stop if applicable
#         if self.is_trailing:
#             if is_long:
#                 new_stop = self.highest_price * (1 - self.value)
#                 if new_stop > self.stop_price:
#                     self.stop_price = new_stop
#             else:
#                 new_stop = self.lowest_price * (1 + self.value)
#                 if new_stop < self.stop_price:
#                     self.stop_price = new_stop

        # Check if stop-loss is triggered
#         if self.stop_type == StopLossType.TIME:
#             if self.time_window is None:
#                 raise ValueError("Time window is required for time-based stop-loss")

#             return current_time - self.entry_time >= self.time_window
#         elif self.stop_type in [
#             StopLossType.FIXED,
#             StopLossType.PERCENT,
#             StopLossType.ATR,
#             StopLossType.TRAILING,
#             StopLossType.VOLATILITY,
        ]:
#             if is_long:
#                 return current_price <= self.stop_price
#             else:
#                 return current_price >= self.stop_price
#         elif self.stop_type == StopLossType.CUSTOM:
            # Custom stop-loss requires external logic
#             return False

#         return False


# class TakeProfit:
#    """Take-profit mechanism for trading positions."""
#
##     def __init__(
##         self,
##         profit_type: TakeProfitType,
##         value: float,
##         is_trailing: bool = False,
##         risk_reward_ratio: Optional[float] = None,
#    ):
#        """"""
#         Initialize take-profit.

#         Args:
#             profit_type: Type of take-profit
#             value: Take-profit value (interpretation depends on profit_type)
#             is_trailing: Whether take-profit is trailing
#             risk_reward_ratio: Risk-reward ratio for RISK_REWARD type
#        """"""
##         self.profit_type = profit_type
##         self.value = value
##         self.is_trailing = is_trailing
##         self.risk_reward_ratio = risk_reward_ratio
##         self.profit_price = None
##         self.highest_price = None
##         self.lowest_price = None
##         self.entry_price = None
##         self.stop_loss = None
#
##     def initialize(
##         self,
##         entry_price: float,
##         is_long: bool,
##         stop_loss: Optional[StopLoss] = None,
##         volatility: Optional[float] = None,
#    ):
#        """"""
#         Initialize take-profit with entry information.

#         Args:
#             entry_price: Entry price
#             is_long: Whether position is long
#             stop_loss: Associated stop-loss (for RISK_REWARD type)
#             volatility: Volatility measure (for VOLATILITY type)
#        """"""
##         self.entry_price = entry_price
##         self.highest_price = entry_price
##         self.lowest_price = entry_price
##         self.stop_loss = stop_loss
#
#        # Calculate initial profit price
##         if self.profit_type == TakeProfitType.FIXED:
##             self.profit_price = self.value
##         elif self.profit_type == TakeProfitType.PERCENT:
##             if is_long:
##                 self.profit_price = entry_price * (1 + self.value)
##             else:
##                 self.profit_price = entry_price * (1 - self.value)
##         elif self.profit_type == TakeProfitType.RISK_REWARD:
##             if stop_loss is None or stop_loss.stop_price is None:
##                 raise ValueError("Stop-loss is required for risk-reward take-profit")
#
##             risk = abs(entry_price - stop_loss.stop_price)
##             reward = risk * self.value  # value is the risk-reward ratio
#
##             if is_long:
##                 self.profit_price = entry_price + reward
##             else:
##                 self.profit_price = entry_price - reward
##         elif self.profit_type == TakeProfitType.TRAILING:
##             if is_long:
##                 self.profit_price = entry_price * (1 + self.value)
##             else:
##                 self.profit_price = entry_price * (1 - self.value)
##         elif self.profit_type == TakeProfitType.VOLATILITY:
##             if volatility is None:
##                 raise ValueError(
#                    "Volatility is required for volatility-based take-profit"
#                )
#
##             if is_long:
##                 self.profit_price = entry_price + (volatility * self.value)
##             else:
##                 self.profit_price = entry_price - (volatility * self.value)
##         elif self.profit_type == TakeProfitType.CUSTOM:
#            # Custom take-profit requires external logic
##             self.profit_price = None
#
##         logger.info(
##             f"Initialized take-profit: type={self.profit_type.value}, price={self.profit_price}"
#        )
#
##     def update(
##         self, current_price: float, is_long: bool, volatility: Optional[float] = None
##     ) -> bool:
#        """"""
#         Update take-profit with current price information.

#         Args:
#             current_price: Current price
#             is_long: Whether position is long
#             volatility: Current volatility measure (for VOLATILITY type)

#         Returns:
#             True if take-profit is triggered, False otherwise
#        """"""
##         if self.entry_price is None:
##             raise ValueError("Take-profit not initialized")
#
#        # Update highest and lowest prices
##         self.highest_price = max(self.highest_price, current_price)
##         self.lowest_price = min(self.lowest_price, current_price)
#
#        # Update trailing take-profit if applicable
##         if self.is_trailing:
##             if is_long:
##                 new_profit = self.highest_price * (1 - self.value)
##                 if new_profit < self.profit_price:
##                     self.profit_price = new_profit
##             else:
##                 new_profit = self.lowest_price * (1 + self.value)
##                 if new_profit > self.profit_price:
##                     self.profit_price = new_profit
#
#        # Check if take-profit is triggered
##         if self.profit_type in [
##             TakeProfitType.FIXED,
##             TakeProfitType.PERCENT,
##             TakeProfitType.RISK_REWARD,
##             TakeProfitType.TRAILING,
##             TakeProfitType.VOLATILITY,
#        ]:
##             if is_long:
##                 return current_price >= self.profit_price
##             else:
##                 return current_price <= self.profit_price
##         elif self.profit_type == TakeProfitType.CUSTOM:
#            # Custom take-profit requires external logic
##             return False
#
##         return False
#
#
## class PositionSizer:
#    """Position sizer for trading strategies."""

#     def __init__(
#         self,
#         sizing_method: PositionSizingMethod,
#         value: float,
#         max_position_size: Optional[float] = None,
#         max_risk_per_trade: Optional[float] = None,
    ):
#        """"""
##         Initialize position sizer.
#
##         Args:
##             sizing_method: Method for position sizing
##             value: Sizing value (interpretation depends on sizing_method)
##             max_position_size: Maximum position size
##             max_risk_per_trade: Maximum risk per trade
#        """"""
#         self.sizing_method = sizing_method
#         self.value = value
#         self.max_position_size = max_position_size
#         self.max_risk_per_trade = max_risk_per_trade

#     def calculate_position_size(
#         self,
#         equity: float,
#         entry_price: float,
#         stop_price: Optional[float] = None,
#         volatility: Optional[float] = None,
#         win_rate: Optional[float] = None,
#         avg_win: Optional[float] = None,
#         avg_loss: Optional[float] = None,
#         correlation: Optional[float] = None,
#     ) -> float:
#        """"""
##         Calculate position size.
#
##         Args:
##             equity: Current equity
##             entry_price: Entry price
##             stop_price: Stop-loss price
##             volatility: Volatility measure
##             win_rate: Win rate
##             avg_win: Average win
##             avg_loss: Average loss
##             correlation: Correlation with other positions
#
##         Returns:
##             Position size
#        """"""
#         position_size = 0.0

#         if self.sizing_method == PositionSizingMethod.FIXED:
            # Fixed position size
#             position_size = self.value

#         elif self.sizing_method == PositionSizingMethod.PERCENT_EQUITY:
            # Percentage of equity
#             position_size = equity * self.value / entry_price

#         elif self.sizing_method == PositionSizingMethod.VOLATILITY:
            # Volatility-based sizing
#             if volatility is None:
#                 raise ValueError(
                    "Volatility is required for volatility-based position sizing"
                )

            # Calculate position size based on target volatility
#             target_volatility = self.value  # e.g., 0.01 for 1% target volatility
#             position_size = (target_volatility * equity) / (volatility * entry_price)

#         elif self.sizing_method == PositionSizingMethod.KELLY:
            # Kelly criterion
#             if win_rate is None or avg_win is None or avg_loss is None:
#                 raise ValueError(
                    "Win rate, average win, and average loss are required for Kelly criterion"
                )

            # Kelly formula: f* = (p * b - (1 - p)) / b
            # where p is win rate, b is win/loss ratio
#             win_loss_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 1.0
#             kelly_fraction = (
#                 win_rate * win_loss_ratio - (1 - win_rate)
#             ) / win_loss_ratio

            # Apply Kelly fraction adjustment (value is the fraction of Kelly to use)
#             kelly_fraction = kelly_fraction * self.value

            # Ensure Kelly fraction is positive and not too large
#             kelly_fraction = max(0.0, min(kelly_fraction, 1.0))

            # Calculate position size
#             position_size = (kelly_fraction * equity) / entry_price

#         elif self.sizing_method == PositionSizingMethod.OPTIMAL_F:
            # Optimal f
#             if avg_win is None or avg_loss is None:
#                 raise ValueError(
                    "Average win and average loss are required for Optimal f"
                )

            # Optimal f formula: f* = ((avg_win / abs(avg_loss)) - 1) / (avg_win / abs(avg_loss))
#             win_loss_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 1.0
#             optimal_f = (
#                 (win_loss_ratio - 1) / win_loss_ratio if win_loss_ratio > 1 else 0.0
            )

            # Apply adjustment (value is the fraction of Optimal f to use)
#             optimal_f = optimal_f * self.value

            # Ensure Optimal f is positive and not too large
#             optimal_f = max(0.0, min(optimal_f, 1.0))

            # Calculate position size
#             position_size = (optimal_f * equity) / entry_price

#         elif self.sizing_method == PositionSizingMethod.RISK_PARITY:
            # Risk parity
#             if volatility is None or correlation is None:
#                 raise ValueError(
                    "Volatility and correlation are required for risk parity"
                )

            # Risk contribution formula: RC_i = w_i * sigma_i * (w * Sigma * w)^(1/2)
            # For a single asset, this simplifies to w_i * sigma_i
            # We want RC_i = target_risk, so w_i = target_risk / sigma_i
#             target_risk = self.value  # e.g., 0.01 for 1% target risk
#             position_size = (target_risk * equity) / (volatility * entry_price)

            # Adjust for correlation (simplified)
#             position_size = position_size * (1 - correlation)

#         elif self.sizing_method == PositionSizingMethod.FIXED_RISK:
            # Fixed risk per trade
#             if stop_price is None:
#                 raise ValueError(
                    "Stop price is required for fixed risk position sizing"
                )

            # Calculate risk per share
#             risk_per_share = abs(entry_price - stop_price)

#             if risk_per_share > 0:
                # Calculate position size based on risk amount
#                 risk_amount = equity * self.value  # e.g., 0.01 for 1% risk
#                 position_size = risk_amount / risk_per_share
#             else:
#                 position_size = 0.0

#         elif self.sizing_method == PositionSizingMethod.CUSTOM:
            # Custom position sizing requires external logic
#             position_size = 0.0

        # Apply maximum position size if specified
#         if self.max_position_size is not None:
#             position_size = min(position_size, self.max_position_size)

        # Apply maximum risk per trade if specified
#         if self.max_risk_per_trade is not None and stop_price is not None:
#             risk_per_share = abs(entry_price - stop_price)
#             max_position_by_risk = (
#                 (equity * self.max_risk_per_trade) / risk_per_share
#                 if risk_per_share > 0
#                 else float("inf")
            )
#             position_size = min(position_size, max_position_by_risk)

#         return position_size


# class RiskMetrics:
#    """Risk metrics calculator for portfolio and strategy evaluation."""
#
##     def __init__(
##         self,
##         returns: Optional[pd.Series] = None,
##         benchmark_returns: Optional[pd.Series] = None,
#    ):
#        """"""
#         Initialize risk metrics calculator.

#         Args:
#             returns: Series of returns
#             benchmark_returns: Series of benchmark returns
#        """"""
##         self.returns = returns
##         self.benchmark_returns = benchmark_returns
#
##     def set_returns(
##         self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
#    ):
#        """"""
#         Set returns data.

#         Args:
#             returns: Series of returns
#             benchmark_returns: Series of benchmark returns
#        """"""
##         self.returns = returns
##         self.benchmark_returns = benchmark_returns
#
##     def calculate_metric(self, metric: RiskMetric, **kwargs) -> float:
#        """"""
#         Calculate risk metric.

#         Args:
#             metric: Risk metric to calculate
#             **kwargs: Additional parameters for specific metrics

#         Returns:
#             Calculated risk metric
#        """"""
##         if self.returns is None:
##             raise ValueError("Returns data not set")
#
##         if metric == RiskMetric.VOLATILITY:
##             return self._calculate_volatility(**kwargs)
##         elif metric == RiskMetric.VAR:
##             return self._calculate_var(**kwargs)
##         elif metric == RiskMetric.CVAR:
##             return self._calculate_cvar(**kwargs)
##         elif metric == RiskMetric.DRAWDOWN:
##             return self._calculate_drawdown(**kwargs)
##         elif metric == RiskMetric.SHARPE:
##             return self._calculate_sharpe(**kwargs)
##         elif metric == RiskMetric.SORTINO:
##             return self._calculate_sortino(**kwargs)
##         elif metric == RiskMetric.CALMAR:
##             return self._calculate_calmar(**kwargs)
##         elif metric == RiskMetric.BETA:
##             return self._calculate_beta(**kwargs)
##         elif metric == RiskMetric.CORRELATION:
##             return self._calculate_correlation(**kwargs)
##         elif metric == RiskMetric.ALPHA:
##             return self._calculate_alpha(**kwargs)
##         elif metric == RiskMetric.OMEGA:
##             return self._calculate_omega(**kwargs)
##         elif metric == RiskMetric.TAIL_RATIO:
##             return self._calculate_tail_ratio(**kwargs)
##         elif metric == RiskMetric.CUSTOM:
##             if "custom_function" not in kwargs:
##                 raise ValueError("Custom function is required for custom risk metric")
#
##             return kwargs["custom_function"](self.returns, **kwargs)
#
##         raise ValueError(f"Unknown risk metric: {metric}")
#
##     def calculate_all_metrics(self, **kwargs) -> Dict[str, float]:
#        """"""
#         Calculate all risk metrics.

#         Args:
#             **kwargs: Additional parameters for specific metrics

#         Returns:
#             Dictionary of risk metrics
#        """"""
##         metrics = {}
#
##         for metric in RiskMetric:
##             if metric != RiskMetric.CUSTOM:
##                 try:
##                     metrics[metric.value] = self.calculate_metric(metric, **kwargs)
##                 except Exception as e:
##                     logger.warning(f"Failed to calculate {metric.value}: {e}")
##                     metrics[metric.value] = None
#
##         return metrics
#
##     def _calculate_volatility(
##         self, annualize: bool = True, trading_days: int = 252
##     ) -> float:
#        """"""
#         Calculate volatility (standard deviation of returns).

#         Args:
#             annualize: Whether to annualize the result
#             trading_days: Number of trading days in a year

#         Returns:
#             Volatility
#        """"""
##         volatility = self.returns.std()
#
##         if annualize:
##             volatility = volatility * np.sqrt(trading_days)
#
##         return volatility
#
##     def _calculate_var(
##         self, confidence: float = 0.95, window: Optional[int] = None
##     ) -> float:
#        """"""
#         Calculate Value at Risk.

#         Args:
#             confidence: Confidence level
#             window: Rolling window size (if None, use all data)

#         Returns:
#             Value at Risk
#        """"""
##         if window is not None:
#            # Use rolling window
##             return self.returns.rolling(window=window).quantile(1 - confidence).min()
#
#        # Use all data
##         return self.returns.quantile(1 - confidence)
#
##     def _calculate_cvar(
##         self, confidence: float = 0.95, window: Optional[int] = None
##     ) -> float:
#        """"""
#         Calculate Conditional Value at Risk (Expected Shortfall).

#         Args:
#             confidence: Confidence level
#             window: Rolling window size (if None, use all data)

#         Returns:
#             Conditional Value at Risk
#        """"""
##         if window is not None:
#            # Use rolling window
##             var = self.returns.rolling(window=window).quantile(1 - confidence)
##             cvar_values = []
#
##             for i in range(window, len(self.returns) + 1):
##                 window_returns = self.returns.iloc[i - window : i]
##                 window_var = var.iloc[i - 1]
##                 cvar_values.append(window_returns[window_returns <= window_var].mean())
#
##             return min(cvar_values)
#
#        # Use all data
##         var = self.returns.quantile(1 - confidence)
##         return self.returns[self.returns <= var].mean()
#
##     def _calculate_drawdown(self) -> float:
#        """"""
#         Calculate maximum drawdown.

#         Returns:
#             Maximum drawdown
#        """"""
#        # Calculate cumulative returns
##         cum_returns = (1 + self.returns).cumprod()
#
#        # Calculate running maximum
##         running_max = cum_returns.cummax()
#
#        # Calculate drawdown
##         drawdown = (cum_returns - running_max) / running_max
#
#        # Return maximum drawdown
##         return drawdown.min()
#
##     def _calculate_sharpe(
##         self,
##         risk_free_rate: float = 0.0,
##         annualize: bool = True,
##         trading_days: int = 252,
##     ) -> float:
#        """"""
#         Calculate Sharpe ratio.

#         Args:
#             risk_free_rate: Risk-free rate
#             annualize: Whether to annualize the result
#             trading_days: Number of trading days in a year

#         Returns:
#             Sharpe ratio
#        """"""
#        # Calculate excess returns
##         excess_returns = (
##             self.returns - risk_free_rate / trading_days
##             if annualize
##             else self.returns - risk_free_rate
#        )
#
#        # Calculate mean and standard deviation
##         mean_excess_returns = excess_returns.mean()
##         std_excess_returns = excess_returns.std()
#
##         if std_excess_returns == 0:
##             return 0.0
#
#        # Calculate Sharpe ratio
##         sharpe = mean_excess_returns / std_excess_returns
#
##         if annualize:
##             sharpe = sharpe * np.sqrt(trading_days)
#
##         return sharpe
#
##     def _calculate_sortino(
##         self,
##         risk_free_rate: float = 0.0,
##         annualize: bool = True,
##         trading_days: int = 252,
##     ) -> float:
#        """"""
#         Calculate Sortino ratio.

#         Args:
#             risk_free_rate: Risk-free rate
#             annualize: Whether to annualize the result
#             trading_days: Number of trading days in a year

#         Returns:
#             Sortino ratio
#        """"""
#        # Calculate excess returns
##         excess_returns = (
##             self.returns - risk_free_rate / trading_days
##             if annualize
##             else self.returns - risk_free_rate
#        )
#
#        # Calculate mean and downside deviation
##         mean_excess_returns = excess_returns.mean()
#
#        # Calculate downside returns
##         downside_returns = excess_returns[excess_returns < 0]
#
##         if len(downside_returns) == 0:
##             return float("inf")  # No downside returns
#
##         downside_deviation = np.sqrt(np.mean(downside_returns**2))
#
##         if downside_deviation == 0:
##             return 0.0
#
#        # Calculate Sortino ratio
##         sortino = mean_excess_returns / downside_deviation
#
##         if annualize:
##             sortino = sortino * np.sqrt(trading_days)
#
##         return sortino
#
##     def _calculate_calmar(
##         self, annualize: bool = True, trading_days: int = 252
##     ) -> float:
#        """"""
#         Calculate Calmar ratio.

#         Args:
#             annualize: Whether to annualize the result
#             trading_days: Number of trading days in a year

#         Returns:
#             Calmar ratio
#        """"""
#        # Calculate annualized return
##         mean_return = self.returns.mean()
##         annualized_return = (
##             (1 + mean_return) ** trading_days - 1 if annualize else mean_return
#        )
#
#        # Calculate maximum drawdown
##         max_drawdown = abs(self._calculate_drawdown())
#
##         if max_drawdown == 0:
##             return float("inf")  # No drawdown
#
#        # Calculate Calmar ratio
##         return annualized_return / max_drawdown
#
##     def _calculate_beta(self) -> float:
#        """"""
#         Calculate beta to benchmark.

#         Returns:
#             Beta
#        """"""
##         if self.benchmark_returns is None:
##             raise ValueError("Benchmark returns not set")
#
#        # Align returns and benchmark returns
##         aligned_returns = pd.concat(
##             [self.returns, self.benchmark_returns], axis=1
##         ).dropna()
#
##         if len(aligned_returns) == 0:
##             return 0.0
#
#        # Calculate covariance and variance
##         covariance = aligned_returns.iloc[:, 0].cov(aligned_returns.iloc[:, 1])
##         variance = aligned_returns.iloc[:, 1].var()
#
##         if variance == 0:
##             return 0.0
#
#        # Calculate beta
##         return covariance / variance
#
##     def _calculate_correlation(self) -> float:
#        """"""
#         Calculate correlation to benchmark.

#         Returns:
#             Correlation
#        """"""
##         if self.benchmark_returns is None:
##             raise ValueError("Benchmark returns not set")
#
#        # Align returns and benchmark returns
##         aligned_returns = pd.concat(
##             [self.returns, self.benchmark_returns], axis=1
##         ).dropna()
#
##         if len(aligned_returns) == 0:
##             return 0.0
#
#        # Calculate correlation
##         return aligned_returns.iloc[:, 0].corr(aligned_returns.iloc[:, 1])
#
##     def _calculate_alpha(
##         self,
##         risk_free_rate: float = 0.0,
##         annualize: bool = True,
##         trading_days: int = 252,
##     ) -> float:
#        """"""
#         Calculate alpha to benchmark.

#         Args:
#             risk_free_rate: Risk-free rate
#             annualize: Whether to annualize the result
#             trading_days: Number of trading days in a year

#         Returns:
#             Alpha
#        """"""
##         if self.benchmark_returns is None:
##             raise ValueError("Benchmark returns not set")
#
#        # Calculate beta
##         beta = self._calculate_beta()
#
#        # Calculate mean returns
##         mean_return = self.returns.mean()
##         mean_benchmark_return = self.benchmark_returns.mean()
#
#        # Calculate daily risk-free rate
##         daily_risk_free_rate = (
##             risk_free_rate / trading_days if annualize else risk_free_rate
#        )
#
#        # Calculate alpha
##         alpha = (
##             mean_return
##             - daily_risk_free_rate
##             - beta * (mean_benchmark_return - daily_risk_free_rate)
#        )
#
##         if annualize:
##             alpha = alpha * trading_days
#
##         return alpha
#
##     def _calculate_omega(self, threshold: float = 0.0) -> float:
#        """"""
#         Calculate Omega ratio.

#         Args:
#             threshold: Return threshold

#         Returns:
#             Omega ratio
#        """"""
#        # Calculate returns above and below threshold
##         returns_above = self.returns[self.returns > threshold]
##         returns_below = self.returns[self.returns <= threshold]
#
##         if len(returns_below) == 0:
##             return float("inf")  # No returns below threshold
#
#        # Calculate Omega ratio
##         return (returns_above - threshold).sum() / abs(
##             (returns_below - threshold).sum()
#        )
#
##     def _calculate_tail_ratio(self, quantile: float = 0.05) -> float:
#        """"""
#         Calculate tail ratio.

#         Args:
#             quantile: Quantile for tail calculation

#         Returns:
#             Tail ratio
#        """"""
#        # Calculate upper and lower tails
##         upper_tail = self.returns.quantile(1 - quantile)
##         lower_tail = abs(self.returns.quantile(quantile))
#
##         if lower_tail == 0:
##             return float("inf")  # No lower tail
#
#        # Calculate tail ratio
##         return upper_tail / lower_tail
#
#
## class RiskManager:
#    """Risk manager for trading strategies."""

#     def __init__(
#         self,
#         risk_level: RiskLevel = RiskLevel.MEDIUM,
#         position_sizer: Optional[PositionSizer] = None,
#         risk_limits: Optional[List[RiskLimit]] = None,
#         metrics_calculator: Optional[RiskMetrics] = None,
    ):
#        """"""
##         Initialize risk manager.
#
##         Args:
##             risk_level: Risk level for trading strategy
##             position_sizer: Position sizer
##             risk_limits: List of risk limits
##             metrics_calculator: Risk metrics calculator
#        """"""
#         self.risk_level = risk_level
#         self.position_sizer = position_sizer
#         self.risk_limits = risk_limits or []
#         self.metrics_calculator = metrics_calculator or RiskMetrics()

#         self.positions = {}
#         self.orders = {}
#         self.portfolio_value = 0.0
#         self.cash = 0.0
#         self.equity = 0.0
#         self.returns = pd.Series()
#         self.benchmark_returns = pd.Series()

#         self.stop_losses = {}
#         self.take_profits = {}

#         self.risk_events = []
#         self.limit_breaches = []

        # Set default position sizer based on risk level
#         if position_sizer is None:
#             if risk_level == RiskLevel.LOW:
#                 self.position_sizer = PositionSizer(
#                     sizing_method=PositionSizingMethod.PERCENT_EQUITY,
#                     value=0.01,  # 1% of equity
#                     max_position_size=None,
#                     max_risk_per_trade=0.005,  # 0.5% max risk per trade
                )
#             elif risk_level == RiskLevel.MEDIUM:
#                 self.position_sizer = PositionSizer(
#                     sizing_method=PositionSizingMethod.PERCENT_EQUITY,
#                     value=0.02,  # 2% of equity
#                     max_position_size=None,
#                     max_risk_per_trade=0.01,  # 1% max risk per trade
                )
#             elif risk_level == RiskLevel.HIGH:
#                 self.position_sizer = PositionSizer(
#                     sizing_method=PositionSizingMethod.PERCENT_EQUITY,
#                     value=0.05,  # 5% of equity
#                     max_position_size=None,
#                     max_risk_per_trade=0.02,  # 2% max risk per trade
                )
#             else:  # CUSTOM
#                 self.position_sizer = PositionSizer(
#                     sizing_method=PositionSizingMethod.FIXED,
#                     value=1.0,
#                     max_position_size=None,
#                     max_risk_per_trade=None,
                )

        # Set default risk limits based on risk level
#         if not risk_limits:
#             if risk_level == RiskLevel.LOW:
#                 self.risk_limits = [
#                     RiskLimit(
#                         RiskLimitType.POSITION_SIZE, 0.05, "alert"
#                     ),  # 5% max position size
#                     RiskLimit(RiskLimitType.EXPOSURE, 0.3, "alert"),  # 30% max exposure
#                     RiskLimit(
#                         RiskLimitType.CONCENTRATION, 0.1, "alert"
#                     ),  # 10% max concentration
#                     RiskLimit(RiskLimitType.DRAWDOWN, 0.05, "alert"),  # 5% max drawdown
#                     RiskLimit(
#                         RiskLimitType.DAILY_LOSS, 0.01, "alert"
#                     ),  # 1% max daily loss
#                     RiskLimit(RiskLimitType.VAR, 0.02, "alert"),  # 2% max VaR
#                     RiskLimit(
#                         RiskLimitType.VOLATILITY, 0.1, "alert"
#                     ),  # 10% max volatility
#                     RiskLimit(RiskLimitType.LEVERAGE, 1.0, "alert"),  # 1.0 max leverage
                ]
#             elif risk_level == RiskLevel.MEDIUM:
#                 self.risk_limits = [
#                     RiskLimit(
#                         RiskLimitType.POSITION_SIZE, 0.1, "alert"
#                     ),  # 10% max position size
#                     RiskLimit(RiskLimitType.EXPOSURE, 0.5, "alert"),  # 50% max exposure
#                     RiskLimit(
#                         RiskLimitType.CONCENTRATION, 0.2, "alert"
#                     ),  # 20% max concentration
#                     RiskLimit(RiskLimitType.DRAWDOWN, 0.1, "alert"),  # 10% max drawdown
#                     RiskLimit(
#                         RiskLimitType.DAILY_LOSS, 0.02, "alert"
#                     ),  # 2% max daily loss
#                     RiskLimit(RiskLimitType.VAR, 0.05, "alert"),  # 5% max VaR
#                     RiskLimit(
#                         RiskLimitType.VOLATILITY, 0.15, "alert"
#                     ),  # 15% max volatility
#                     RiskLimit(RiskLimitType.LEVERAGE, 1.5, "alert"),  # 1.5 max leverage
                ]
#             elif risk_level == RiskLevel.HIGH:
#                 self.risk_limits = [
#                     RiskLimit(
#                         RiskLimitType.POSITION_SIZE, 0.2, "alert"
#                     ),  # 20% max position size
#                     RiskLimit(RiskLimitType.EXPOSURE, 0.8, "alert"),  # 80% max exposure
#                     RiskLimit(
#                         RiskLimitType.CONCENTRATION, 0.3, "alert"
#                     ),  # 30% max concentration
#                     RiskLimit(RiskLimitType.DRAWDOWN, 0.2, "alert"),  # 20% max drawdown
#                     RiskLimit(
#                         RiskLimitType.DAILY_LOSS, 0.05, "alert"
#                     ),  # 5% max daily loss
#                     RiskLimit(RiskLimitType.VAR, 0.1, "alert"),  # 10% max VaR
#                     RiskLimit(
#                         RiskLimitType.VOLATILITY, 0.25, "alert"
#                     ),  # 25% max volatility
#                     RiskLimit(RiskLimitType.LEVERAGE, 2.0, "alert"),  # 2.0 max leverage
                ]

#     def set_portfolio_value(self, portfolio_value: float, cash: float):
#        """"""
##         Set portfolio value and cash.
#
##         Args:
##             portfolio_value: Portfolio value
##             cash: Cash
#        """"""
#         self.portfolio_value = portfolio_value
#         self.cash = cash
#         self.equity = portfolio_value - cash

#     def add_position(
#         self,
#         symbol: str,
#         quantity: float,
#         entry_price: float,
#         is_long: bool,
#         stop_loss: Optional[StopLoss] = None,
#         take_profit: Optional[TakeProfit] = None,
    ):
#        """"""
##         Add position.
#
##         Args:
##             symbol: Symbol
##             quantity: Quantity
##             entry_price: Entry price
##             is_long: Whether position is long
##             stop_loss: Stop-loss
##             take_profit: Take-profit
#        """"""
#         self.positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": entry_price,
            "is_long": is_long,
            "entry_time": datetime.now(),
            "market_value": quantity * entry_price,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_percent": 0.0,
        }

        # Initialize stop-loss if provided
#         if stop_loss:
#             stop_loss.initialize(
#                 entry_price=entry_price, entry_time=datetime.now(), is_long=is_long
            )
#             self.stop_losses[symbol] = stop_loss

        # Initialize take-profit if provided
#         if take_profit:
#             take_profit.initialize(
#                 entry_price=entry_price, is_long=is_long, stop_loss=stop_loss
            )
#             self.take_profits[symbol] = take_profit

#         logger.info(
#             f"Added position: {symbol}, quantity={quantity}, entry_price={entry_price}, is_long={is_long}"
        )

#     def update_position(
#         self, symbol: str, current_price: float, quantity: Optional[float] = None
    ):
#        """"""
##         Update position.
#
##         Args:
##             symbol: Symbol
##             current_price: Current price
##             quantity: New quantity (if changed)
#        """"""
#         if symbol not in self.positions:
#             logger.warning(f"Position not found: {symbol}")
#             return

#         position = self.positions[symbol]

        # Update quantity if provided
#         if quantity is not None:
#             position["quantity"] = quantity

        # Update current price and market value
#         position["current_price"] = current_price
#         position["market_value"] = position["quantity"] * current_price

        # Calculate unrealized P&L
#         if position["is_long"]:
#             position["unrealized_pnl"] = position["quantity"] * (
#                 current_price - position["entry_price"]
            )
#         else:
#             position["unrealized_pnl"] = position["quantity"] * (
#                 position["entry_price"] - current_price
            )

        # Calculate unrealized P&L percent
#         if position["entry_price"] > 0:
#             position["unrealized_pnl_percent"] = position["unrealized_pnl"] / (
#                 position["quantity"] * position["entry_price"]
            )

        # Update stop-loss if exists
#         if symbol in self.stop_losses:
#             stop_loss = self.stop_losses[symbol]
#             is_triggered = stop_loss.update(
#                 current_price=current_price,
#                 current_time=datetime.now(),
#                 is_long=position["is_long"],
            )

#             if is_triggered:
#                 logger.info(f"Stop-loss triggered for {symbol} at {current_price}")
#                 self._add_risk_event("stop_loss_triggered", symbol, current_price)
#                 return True

        # Update take-profit if exists
#         if symbol in self.take_profits:
#             take_profit = self.take_profits[symbol]
#             is_triggered = take_profit.update(
#                 current_price=current_price, is_long=position["is_long"]
            )

#             if is_triggered:
#                 logger.info(f"Take-profit triggered for {symbol} at {current_price}")
#                 self._add_risk_event("take_profit_triggered", symbol, current_price)
#                 return True

#         return False

#     def remove_position(self, symbol: str):
#        """"""
##         Remove position.
#
##         Args:
##             symbol: Symbol
#        """"""
#         if symbol in self.positions:
#             self.positions.pop(symbol)

#         if symbol in self.stop_losses:
#             self.stop_losses.pop(symbol)

#         if symbol in self.take_profits:
#             self.take_profits.pop(symbol)

#         logger.info(f"Removed position: {symbol}")

#     def calculate_position_size(
#         self,
#         symbol: str,
#         entry_price: float,
#         stop_price: Optional[float] = None,
#         **kwargs,
#     ) -> float:
#        """"""
##         Calculate position size.
#
##         Args:
##             symbol: Symbol
##             entry_price: Entry price
##             stop_price: Stop-loss price
##             **kwargs: Additional parameters for position sizer
#
##         Returns:
##             Position size
#        """"""
#         if self.position_sizer is None:
#             logger.warning("Position sizer not set")
#             return 0.0

#         return self.position_sizer.calculate_position_size(
#             equity=self.equity, entry_price=entry_price, stop_price=stop_price, **kwargs
        )

#     def check_risk_limits(self) -> List[RiskLimit]:
#        """"""
##         Check risk limits.
#
##         Returns:
##             List of breached risk limits
#        """"""
#         breached_limits = []

#         for limit in self.risk_limits:
#             current_value = self._get_risk_limit_value(limit.limit_type)

#             if limit.check(current_value):
#                 breached_limits.append(limit)

                # Record limit breach
#                 self._add_limit_breach(limit, current_value)

                # Take action based on limit breach
#                 self._handle_limit_breach(limit, current_value)

#         return breached_limits

#     def update_returns(
#         self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
    ):
#        """"""
##         Update returns data.
#
##         Args:
##             returns: Series of returns
##             benchmark_returns: Series of benchmark returns
#        """"""
#         self.returns = returns

#         if benchmark_returns is not None:
#             self.benchmark_returns = benchmark_returns

        # Update risk metrics calculator
#         self.metrics_calculator.set_returns(returns, benchmark_returns)

#     def calculate_risk_metrics(
#         self, metrics: Optional[List[RiskMetric]] = None, **kwargs
#     ) -> Dict[str, float]:
#        """"""
##         Calculate risk metrics.
#
##         Args:
##             metrics: List of risk metrics to calculate (if None, calculate all)
##             **kwargs: Additional parameters for specific metrics
#
##         Returns:
##             Dictionary of risk metrics
#        """"""
#         if metrics is None:
#             return self.metrics_calculator.calculate_all_metrics(**kwargs)

#         result = {}
#         for metric in metrics:
#             result[metric.value] = self.metrics_calculator.calculate_metric(
#                 metric, **kwargs
            )

#         return result

#     def get_risk_events(
#         self, event_type: Optional[str] = None, limit: Optional[int] = None
#     ) -> List[Dict[str, Any]]:
#        """"""
##         Get risk events.
#
##         Args:
##             event_type: Type of events to get (if None, get all)
##             limit: Maximum number of events to return (if None, return all)
#
##         Returns:
##             List of risk events
#        """"""
#         if event_type is None:
#             events = self.risk_events
#         else:
#             events = [
#                 event for event in self.risk_events if event["type"] == event_type
            ]

#         if limit is not None:
#             events = events[-limit:]

#         return events

#     def get_limit_breaches(
#         self, limit_type: Optional[RiskLimitType] = None, limit: Optional[int] = None
#     ) -> List[Dict[str, Any]]:
#        """"""
##         Get limit breaches.
#
##         Args:
##             limit_type: Type of limit breaches to get (if None, get all)
##             limit: Maximum number of breaches to return (if None, return all)
#
##         Returns:
##             List of limit breaches
#        """"""
#         if limit_type is None:
#             breaches = self.limit_breaches
#         else:
#             breaches = [
#                 breach
#                 for breach in self.limit_breaches
#                 if breach["limit_type"] == limit_type.value
            ]

#         if limit is not None:
#             breaches = breaches[-limit:]

#         return breaches

#     def _get_risk_limit_value(self, limit_type: RiskLimitType) -> float:
#        """"""
##         Get current value for risk limit.
#
##         Args:
##             limit_type: Risk limit type
#
##         Returns:
##             Current value
#        """"""
#         if limit_type == RiskLimitType.POSITION_SIZE:
            # Maximum position size as percentage of portfolio
#             if not self.positions:
#                 return 0.0

#             max_position_value = max(
#                 position["market_value"] for position in self.positions.values()
            )
#             return (
#                 max_position_value / self.portfolio_value
#                 if self.portfolio_value > 0
#                 else 0.0
            )

#         elif limit_type == RiskLimitType.EXPOSURE:
            # Total exposure as percentage of portfolio
#             total_exposure = sum(
#                 position["market_value"] for position in self.positions.values()
            )
#             return (
#                 total_exposure / self.portfolio_value
#                 if self.portfolio_value > 0
#                 else 0.0
            )

#         elif limit_type == RiskLimitType.CONCENTRATION:
            # Maximum concentration in single asset
#             if not self.positions:
#                 return 0.0

#             total_exposure = sum(
#                 position["market_value"] for position in self.positions.values()
            )
#             max_position_value = max(
#                 position["market_value"] for position in self.positions.values()
            )
#             return max_position_value / total_exposure if total_exposure > 0 else 0.0

#         elif limit_type == RiskLimitType.DRAWDOWN:
            # Maximum drawdown
#             if self.returns.empty:
#                 return 0.0

#             return abs(self.metrics_calculator.calculate_metric(RiskMetric.DRAWDOWN))

#         elif limit_type == RiskLimitType.DAILY_LOSS:
            # Daily loss as percentage of portfolio
#             if self.returns.empty:
#                 return 0.0

#             daily_return = self.returns.iloc[-1] if len(self.returns) > 0 else 0.0
#             return abs(daily_return) if daily_return < 0 else 0.0

#         elif limit_type == RiskLimitType.VAR:
            # Value at Risk
#             if self.returns.empty:
#                 return 0.0

#             return abs(self.metrics_calculator.calculate_metric(RiskMetric.VAR))

#         elif limit_type == RiskLimitType.VOLATILITY:
            # Volatility
#             if self.returns.empty:
#                 return 0.0

#             return self.metrics_calculator.calculate_metric(RiskMetric.VOLATILITY)

#         elif limit_type == RiskLimitType.LEVERAGE:
            # Leverage
#             total_exposure = sum(
#                 position["market_value"] for position in self.positions.values()
            )
#             return total_exposure / self.equity if self.equity > 0 else 0.0

#         elif limit_type == RiskLimitType.CUSTOM:
            # Custom limit requires external logic
#             return 0.0

#         return 0.0

#     def _add_risk_event(self, event_type: str, symbol: str, price: float, **kwargs):
#        """"""
##         Add risk event.
#
##         Args:
##             event_type: Event type
##             symbol: Symbol
##             price: Price
##             **kwargs: Additional event data
#        """"""
#         event = {
            "type": event_type,
            "symbol": symbol,
            "price": price,
            "time": datetime.now(),
#             **kwargs,
        }

#         self.risk_events.append(event)
#         logger.info(f"Risk event: {event_type} for {symbol} at {price}")

#     def _add_limit_breach(self, limit: RiskLimit, current_value: float):
#        """"""
##         Add limit breach.
#
##         Args:
##             limit: Risk limit
##             current_value: Current value
#        """"""
#         breach = {
            "limit_type": limit.limit_type.value,
            "limit_value": limit.value,
            "current_value": current_value,
            "time": datetime.now(),
            "action": limit.action,
        }

#         self.limit_breaches.append(breach)
#         logger.warning(
#             f"Risk limit breach: {limit.limit_type.value}, limit={limit.value}, current={current_value}"
        )

#     def _handle_limit_breach(self, limit: RiskLimit, current_value: float):
#        """"""
##         Handle limit breach.
#
##         Args:
##             limit: Risk limit
##             current_value: Current value
#        """"""
#         action, custom_action = limit.get_action()

#         if action == "alert":
            # Just log the breach (already done in _add_limit_breach)
#             pass

#         elif action == "reduce":
            # Reduce exposure
#             self._reduce_exposure(limit.limit_type, limit.value, current_value)

#         elif action == "close":
            # Close positions
#             self._close_positions(limit.limit_type)

#         elif action == "custom" and custom_action is not None:
            # Execute custom action
#             custom_action(limit, current_value, self)

#     def _reduce_exposure(
#         self, limit_type: RiskLimitType, limit_value: float, current_value: float
    ):
#        """"""
##         Reduce exposure to comply with risk limit.
#
##         Args:
##             limit_type: Risk limit type
##             limit_value: Limit value
##             current_value: Current value
#        """"""
#         if limit_type == RiskLimitType.POSITION_SIZE:
            # Reduce largest position
#             if not self.positions:
#                 return

            # Find largest position
#             largest_position = max(
#                 self.positions.items(), key=lambda x: x[1]["market_value"]
            )
#             symbol, position = largest_position

            # Calculate reduction needed
#             max_position_value = limit_value * self.portfolio_value
#             current_position_value = position["market_value"]
#             reduction_ratio = (
#                 max_position_value / current_position_value
#                 if current_position_value > 0
#                 else 0.0
            )

#             if reduction_ratio < 1.0:
                # Reduce position size
#                 new_quantity = position["quantity"] * reduction_ratio
#                 self._add_risk_event(
                    "position_reduced",
#                     symbol,
#                     position["current_price"],
#                     old_quantity=position["quantity"],
#                     new_quantity=new_quantity,
#                     reason=f"Position size limit breach: {limit_value:.2%}",
                )

                # Update position
#                 self.update_position(symbol, position["current_price"], new_quantity)

#         elif limit_type == RiskLimitType.EXPOSURE:
            # Reduce overall exposure
#             if not self.positions:
#                 return

            # Calculate total exposure
#             total_exposure = sum(
#                 position["market_value"] for position in self.positions.values()
            )

            # Calculate reduction needed
#             max_exposure = limit_value * self.portfolio_value
#             reduction_ratio = (
#                 max_exposure / total_exposure if total_exposure > 0 else 0.0
            )

#             if reduction_ratio < 1.0:
                # Reduce all positions proportionally
#                 for symbol, position in self.positions.items():
#                     new_quantity = position["quantity"] * reduction_ratio
#                     self._add_risk_event(
                        "position_reduced",
#                         symbol,
#                         position["current_price"],
#                         old_quantity=position["quantity"],
#                         new_quantity=new_quantity,
#                         reason=f"Exposure limit breach: {limit_value:.2%}",
                    )

                    # Update position
#                     self.update_position(
#                         symbol, position["current_price"], new_quantity
                    )

#         elif limit_type == RiskLimitType.CONCENTRATION:
            # Reduce concentration in largest position
#             if not self.positions:
#                 return

            # Find largest position
#             largest_position = max(
#                 self.positions.items(), key=lambda x: x[1]["market_value"]
            )
#             symbol, position = largest_position

            # Calculate total exposure
#             total_exposure = sum(
#                 position["market_value"] for position in self.positions.values()
            )

            # Calculate concentration
#             concentration = (
#                 position["market_value"] / total_exposure if total_exposure > 0 else 0.0
            )

#             if concentration > limit_value:
                # Calculate reduction needed
#                 new_position_value = limit_value * total_exposure
#                 reduction_ratio = (
#                     new_position_value / position["market_value"]
#                     if position["market_value"] > 0
#                     else 0.0
                )

                # Reduce position size
#                 new_quantity = position["quantity"] * reduction_ratio
#                 self._add_risk_event(
                    "position_reduced",
#                     symbol,
#                     position["current_price"],
#                     old_quantity=position["quantity"],
#                     new_quantity=new_quantity,
#                     reason=f"Concentration limit breach: {limit_value:.2%}",
                )

                # Update position
#                 self.update_position(symbol, position["current_price"], new_quantity)

#     def _close_positions(self, limit_type: RiskLimitType):
#        """"""
##         Close positions due to risk limit breach.
#
##         Args:
##             limit_type: Risk limit type
#        """"""
#         if (
#             limit_type == RiskLimitType.DRAWDOWN
#             or limit_type == RiskLimitType.DAILY_LOSS
        ):
            # Close all positions
#             for symbol, position in list(self.positions.items()):
#                 self._add_risk_event(
                    "position_closed",
#                     symbol,
#                     position["current_price"],
#                     quantity=position["quantity"],
#                     reason=f"{limit_type.value} limit breach",
                )

                # Remove position
#                 self.remove_position(symbol)

#         elif limit_type == RiskLimitType.POSITION_SIZE:
            # Close largest position
#             if not self.positions:
#                 return

            # Find largest position
#             largest_position = max(
#                 self.positions.items(), key=lambda x: x[1]["market_value"]
            )
#             symbol, position = largest_position

#             self._add_risk_event(
                "position_closed",
#                 symbol,
#                 position["current_price"],
#                 quantity=position["quantity"],
#                 reason=f"Position size limit breach",
            )

            # Remove position
#             self.remove_position(symbol)

#         elif limit_type == RiskLimitType.LEVERAGE:
            # Close positions to reduce leverage
#             if not self.positions:
#                 return

            # Sort positions by unrealized P&L (close worst performers first)
#             sorted_positions = sorted(
#                 self.positions.items(), key=lambda x: x[1]["unrealized_pnl_percent"]
            )

            # Close positions until leverage is below limit
#             for symbol, position in sorted_positions:
#                 self._add_risk_event(
                    "position_closed",
#                     symbol,
#                     position["current_price"],
#                     quantity=position["quantity"],
#                     reason=f"Leverage limit breach",
                )

                # Remove position
#                 self.remove_position(symbol)

                # Recalculate leverage
#                 total_exposure = sum(p["market_value"] for p in self.positions.values())
#                 leverage = total_exposure / self.equity if self.equity > 0 else 0.0

                # Check if leverage is now below limit
#                 limit = next(
                    (
                        l
#                         for l in self.risk_limits
#                         if l.limit_type == RiskLimitType.LEVERAGE
                    ),
#                     None,
                )
#                 if limit is None or leverage <= limit.value:
#                     break


# Example usage
# def example_usage():
#    """Example of how to use the risk management module."""
#    # Create risk manager with medium risk level
##     risk_manager = RiskManager(risk_level=RiskLevel.MEDIUM)
#
#    # Set portfolio value
##     risk_manager.set_portfolio_value(portfolio_value=100000.0, cash=50000.0)
#
#    # Create stop-loss
##     stop_loss = StopLoss(
##         stop_type=StopLossType.PERCENT, value=0.05, is_trailing=True  # 5% stop-loss
#    )
#
#    # Create take-profit
##     take_profit = TakeProfit(
##         profit_type=TakeProfitType.PERCENT,
##         value=0.1,  # 10% take-profit
##         is_trailing=False,
#    )
#
#    # Add position
##     risk_manager.add_position(
##         symbol="AAPL",
##         quantity=100,
##         entry_price=150.0,
##         is_long=True,
##         stop_loss=stop_loss,
##         take_profit=take_profit,
#    )
#
#    # Update position with new price
##     triggered = risk_manager.update_position(symbol="AAPL", current_price=160.0)
#
##     if triggered:
##         print("Stop-loss or take-profit triggered")
#
#    # Calculate position size for new trade
##     position_size = risk_manager.calculate_position_size(
##         symbol="MSFT", entry_price=250.0, stop_price=240.0
#    )
#
##     print(f"Calculated position size: {position_size}")
#
#    # Check risk limits
##     breached_limits = risk_manager.check_risk_limits()
#
##     if breached_limits:
##         print(
##             f"Breached limits: {[limit.limit_type.value for limit in breached_limits]}"
#        )
#
#    # Calculate risk metrics
##     metrics = risk_manager.calculate_risk_metrics(
##         [RiskMetric.VOLATILITY, RiskMetric.SHARPE, RiskMetric.DRAWDOWN]
#    )
#
##     print(f"Risk metrics: {metrics}")
#
#
## if __name__ == "__main__":
##     example_usage()
