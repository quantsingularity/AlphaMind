"""
## Advanced Risk Controls Module for AlphaMind

## This module provides comprehensive risk management tools for trading strategies,
## including position sizing, stop-loss mechanisms, exposure limits, and risk metrics.
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level for trading strategies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CUSTOM = "custom"


class StopLossType(Enum):
    """Types of stop-loss mechanisms."""

    FIXED = "fixed"  # Fixed price
    PERCENT = "percent"  # Percentage from entry
    ATR = "atr"  # Average True Range multiple
    TRAILING = "trailing"  # Trailing stop
    VOLATILITY = "volatility"  # Volatility-based
    TIME = "time"  # Time-based stop
    CUSTOM = "custom"  # Custom stop-loss logic


class TakeProfitType(Enum):
    """Types of take-profit mechanisms."""

    FIXED = "fixed"  # Fixed price
    PERCENT = "percent"  # Percentage from entry
    RISK_REWARD = "risk_reward"  # Risk-reward ratio
    TRAILING = "trailing"  # Trailing take-profit
    VOLATILITY = "volatility"  # Volatility-based
    CUSTOM = "custom"  # Custom take-profit logic


class PositionSizingMethod(Enum):
    """Methods for position sizing."""

    FIXED = "fixed"  # Fixed position size
    PERCENT_EQUITY = "percent_equity"  # Percentage of equity
    VOLATILITY = "volatility"  # Volatility-based sizing
    KELLY = "kelly"  # Kelly criterion
    OPTIMAL_F = "optimal_f"  # Optimal f
    RISK_PARITY = "risk_parity"  # Risk parity
    FIXED_RISK = "fixed_risk"  # Fixed risk per trade
    CUSTOM = "custom"  # Custom position sizing logic


class RiskMetric(Enum):
    """Risk metrics for portfolio and strategy evaluation."""

    VOLATILITY = "volatility"  # Standard deviation of returns
    VAR = "var"  # Value at Risk
    CVAR = "cvar"  # Conditional Value at Risk (Expected Shortfall)
    DRAWDOWN = "drawdown"  # Maximum drawdown
    SHARPE = "sharpe"  # Sharpe ratio
    SORTINO = "sortino"  # Sortino ratio
    CALMAR = "calmar"  # Calmar ratio
    BETA = "beta"  # Beta to benchmark
    CORRELATION = "correlation"  # Correlation to benchmark
    ALPHA = "alpha"  # Alpha to benchmark
    OMEGA = "omega"  # Omega ratio
    TAIL_RATIO = "tail_ratio"  # Tail ratio
    CUSTOM = "custom"  # Custom risk metric


class RiskLimitType(Enum):
    """Types of risk limits."""

    POSITION_SIZE = "position_size"  # Maximum position size
    EXPOSURE = "exposure"  # Maximum exposure
    CONCENTRATION = "concentration"  # Maximum concentration in single asset
    DRAWDOWN = "drawdown"  # Maximum drawdown
    DAILY_LOSS = "daily_loss"  # Maximum daily loss
    VAR = "var"  # Value at Risk limit
    VOLATILITY = "volatility"  # Volatility limit
    LEVERAGE = "leverage"  # Maximum leverage
    CUSTOM = "custom"  # Custom risk limit


class RiskLimit:
    """Risk limit for trading strategies."""

    def __init__(
        self,
        limit_type: RiskLimitType,
        value: float,
        action: str = "alert",
        custom_action: Optional[Callable] = None,
    ):
        """
        Initialize risk limit.

        Args:
            limit_type: Type of risk limit
            value: Limit value
            action: Action to take when limit is breached ("alert", "reduce", "close", "custom")
            custom_action: Custom action function to call when limit is breached
        """
        self.limit_type = limit_type
        self.value = value
        self.action = action
        self.custom_action = custom_action
        self.is_breached = False
        self.breach_time = None
        self.breach_value = None

    def check(self, current_value: float) -> bool:
        """
        Check if limit is breached.

        Args:
            current_value: Current value to check against limit

        Returns:
            True if limit is breached, False otherwise
        """
        # Different limit types have different comparison logic
        if self.limit_type in [
            RiskLimitType.POSITION_SIZE,
            RiskLimitType.EXPOSURE,
            RiskLimitType.CONCENTRATION,
            RiskLimitType.DRAWDOWN,
            RiskLimitType.DAILY_LOSS,
            RiskLimitType.VAR,
            RiskLimitType.VOLATILITY,
            RiskLimitType.LEVERAGE,
        ]:
            # For these limits, breach occurs when current value exceeds limit
            is_breached = abs(current_value) > self.value
        else:
            # For custom limits, use custom logic
            is_breached = current_value > self.value

        # Update breach status
        if is_breached and not self.is_breached:
            self.is_breached = True
            self.breach_time = datetime.now()
            self.breach_value = current_value
        elif not is_breached and self.is_breached:
            self.is_breached = False
            self.breach_time = None
            self.breach_value = None

        return self.is_breached

    def get_action(self) -> Tuple[str, Optional[Callable]]:
        """
        Get action to take when limit is breached.

        Returns:
            Tuple of action type and custom action function (if any)
        """
        return self.action, self.custom_action


class StopLoss:
    """Stop-loss mechanism for trading positions."""

    def __init__(
        self,
        stop_type: StopLossType,
        value: float,
        is_trailing: bool = False,
        time_window: Optional[timedelta] = None,
    ):
        """
        Initialize stop-loss.

        Args:
            stop_type: Type of stop-loss
            value: Stop-loss value (interpretation depends on stop_type)
            is_trailing: Whether stop-loss is trailing
            time_window: Time window for time-based stop-loss
        """
        self.stop_type = stop_type
        self.value = value
        self.is_trailing = is_trailing
        self.time_window = time_window
        self.stop_price = None
        self.highest_price = None
        self.lowest_price = None
        self.entry_time = None
        self.entry_price = None
        self.atr = None

    def initialize(
        self,
        entry_price: float,
        entry_time: datetime,
        is_long: bool,
        atr: Optional[float] = None,
    ):
        """
        Initialize stop-loss with entry information.

        Args:
            entry_price: Entry price
            entry_time: Entry time
            is_long: Whether position is long
            atr: Average True Range (for ATR-based stop-loss)
        """
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.atr = atr

        # Calculate initial stop price
        if self.stop_type == StopLossType.FIXED:
            self.stop_price = self.value
        elif self.stop_type == StopLossType.PERCENT:
            if is_long:
                self.stop_price = entry_price * (1 - self.value)
            else:
                self.stop_price = entry_price * (1 + self.value)
        elif self.stop_type == StopLossType.ATR:
            if atr is None:
                raise ValueError("ATR is required for ATR-based stop-loss")

            if is_long:
                self.stop_price = entry_price - (atr * self.value)
            else:
                self.stop_price = entry_price + (atr * self.value)
        elif self.stop_type == StopLossType.TRAILING:
            if is_long:
                self.stop_price = entry_price * (1 - self.value)
            else:
                self.stop_price = entry_price * (1 + self.value)
        elif self.stop_type == StopLossType.VOLATILITY:
            # Volatility-based stop-loss requires historical volatility
            # For now, use a simple approximation based on ATR
            if atr is None:
                raise ValueError("ATR is required for volatility-based stop-loss")

            if is_long:
                self.stop_price = entry_price - (atr * self.value)
            else:
                self.stop_price = entry_price + (atr * self.value)
        elif self.stop_type == StopLossType.TIME:
            # Time-based stop-loss doesn't have a price level
            self.stop_price = None
        elif self.stop_type == StopLossType.CUSTOM:
            # Custom stop-loss requires external logic
            self.stop_price = None

        logger.info(
            f"Initialized stop-loss: type={self.stop_type.value}, price={self.stop_price}"
        )

    def update(
        self,
        current_price: float,
        current_time: datetime,
        is_long: bool,
        atr: Optional[float] = None,
    ) -> bool:
        """
        Update stop-loss with current price information.

        Args:
            current_price: Current price
            current_time: Current time
            is_long: Whether position is long
            atr: Current Average True Range (for ATR-based stop-loss)

        Returns:
            True if stop-loss is triggered, False otherwise
        """
        if self.entry_price is None:
            raise ValueError("Stop-loss not initialized")

        # Update highest and lowest prices
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)

        # Update ATR if provided
        if atr is not None:
            self.atr = atr

        # Update trailing stop if applicable
        if self.is_trailing:
            if is_long:
                new_stop = self.highest_price * (1 - self.value)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
            else:
                new_stop = self.lowest_price * (1 + self.value)
                if new_stop < self.stop_price:
                    self.stop_price = new_stop

        # Check if stop-loss is triggered
        if self.stop_type == StopLossType.TIME:
            if self.time_window is None:
                raise ValueError("Time window is required for time-based stop-loss")

            return current_time - self.entry_time >= self.time_window
        elif self.stop_type in [
            StopLossType.FIXED,
            StopLossType.PERCENT,
            StopLossType.ATR,
            StopLossType.TRAILING,
            StopLossType.VOLATILITY,
        ]:
            if is_long:
                return current_price <= self.stop_price
            else:
                return current_price >= self.stop_price
        elif self.stop_type == StopLossType.CUSTOM:
            # Custom stop-loss requires external logic
            return False

        return False


class TakeProfit:
    """Take-profit mechanism for trading positions."""

    def __init__(
        self,
        profit_type: TakeProfitType,
        value: float,
        is_trailing: bool = False,
        risk_reward_ratio: Optional[float] = None,
    ):
        """
        Initialize take-profit.

        Args:
            profit_type: Type of take-profit
            value: Take-profit value (interpretation depends on profit_type)
            is_trailing: Whether take-profit is trailing
            risk_reward_ratio: Risk-reward ratio for RISK_REWARD type
        """
        self.profit_type = profit_type
        self.value = value
        self.is_trailing = is_trailing
        self.risk_reward_ratio = risk_reward_ratio
        self.profit_price = None
        self.highest_price = None
        self.lowest_price = None
        self.entry_price = None
        self.stop_loss = None

    def initialize(
        self,
        entry_price: float,
        is_long: bool,
        stop_loss: Optional[StopLoss] = None,
        volatility: Optional[float] = None,
    ):
        """
        Initialize take-profit with entry information.

        Args:
            entry_price: Entry price
            is_long: Whether position is long
            stop_loss: Associated stop-loss (for RISK_REWARD type)
            volatility: Volatility measure (for VOLATILITY type)
        """
        self.entry_price = entry_price
        self.highest_price = entry_price
        self.lowest_price = entry_price
        self.stop_loss = stop_loss

        # Calculate initial profit price
        if self.profit_type == TakeProfitType.FIXED:
            self.profit_price = self.value
        elif self.profit_type == TakeProfitType.PERCENT:
            if is_long:
                self.profit_price = entry_price * (1 + self.value)
            else:
                self.profit_price = entry_price * (1 - self.value)
        elif self.profit_type == TakeProfitType.RISK_REWARD:
            if stop_loss is None or stop_loss.stop_price is None:
                raise ValueError("Stop-loss is required for risk-reward take-profit")

            risk = abs(entry_price - stop_loss.stop_price)
            reward = risk * self.value  # value is the risk-reward ratio

            if is_long:
                self.profit_price = entry_price + reward
            else:
                self.profit_price = entry_price - reward
        elif self.profit_type == TakeProfitType.TRAILING:
            if is_long:
                self.profit_price = entry_price * (1 + self.value)
            else:
                self.profit_price = entry_price * (1 - self.value)
        elif self.profit_type == TakeProfitType.VOLATILITY:
            if volatility is None:
                raise ValueError(
                    "Volatility is required for volatility-based take-profit"
                )

            if is_long:
                self.profit_price = entry_price + (volatility * self.value)
            else:
                self.profit_price = entry_price - (volatility * self.value)
        elif self.profit_type == TakeProfitType.CUSTOM:
            # Custom take-profit requires external logic
            self.profit_price = None

        logger.info(
            f"Initialized take-profit: type={self.profit_type.value}, price={self.profit_price}"
        )

    def update(
        self, current_price: float, is_long: bool, volatility: Optional[float] = None
    ) -> bool:
        """
        Update take-profit with current price information.

        Args:
            current_price: Current price
            is_long: Whether position is long
            volatility: Current volatility measure (for VOLATILITY type)

        Returns:
            True if take-profit is triggered, False otherwise
        """
        if self.entry_price is None:
            raise ValueError("Take-profit not initialized")

        # Update highest and lowest prices
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)

        # Update trailing take-profit if applicable
        if self.is_trailing:
            if is_long:
                new_profit = self.highest_price * (1 - self.value)
                if new_profit < self.profit_price:
                    self.profit_price = new_profit
            else:
                new_profit = self.lowest_price * (1 + self.value)
                if new_profit > self.profit_price:
                    self.profit_price = new_profit

        # Check if take-profit is triggered
        if self.profit_type in [
            TakeProfitType.FIXED,
            TakeProfitType.PERCENT,
            TakeProfitType.RISK_REWARD,
            TakeProfitType.TRAILING,
            TakeProfitType.VOLATILITY,
        ]:
            if is_long:
                return current_price >= self.profit_price
            else:
                return current_price <= self.profit_price
        elif self.profit_type == TakeProfitType.CUSTOM:
            # Custom take-profit requires external logic
            return False

        return False


class PositionSizer:
    """Position sizer for trading strategies."""

    def __init__(
        self,
        sizing_method: PositionSizingMethod,
        value: float,
        max_position_size: Optional[float] = None,
        max_risk_per_trade: Optional[float] = None,
    ):
        """
        Initialize position sizer.

        Args:
            sizing_method: Method for position sizing
            value: Sizing value (interpretation depends on sizing_method)
            max_position_size: Maximum position size
            max_risk_per_trade: Maximum risk per trade
        """
        self.sizing_method = sizing_method
        self.value = value
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        correlation: Optional[float] = None,
    ) -> float:
        """
        Calculate position size.

        Args:
            equity: Current equity
            entry_price: Entry price
            stop_price: Stop-loss price
            volatility: Volatility measure
            win_rate: Win rate
            avg_win: Average win
            avg_loss: Average loss
            correlation: Correlation with other positions

        Returns:
            Position size
        """
        position_size = 0.0

        if self.sizing_method == PositionSizingMethod.FIXED:
            # Fixed position size
            position_size = self.value

        elif self.sizing_method == PositionSizingMethod.PERCENT_EQUITY:
            # Percentage of equity
            position_size = equity * self.value / entry_price

        elif self.sizing_method == PositionSizingMethod.VOLATILITY:
            # Volatility-based sizing
            if volatility is None:
                raise ValueError(
                    "Volatility is required for volatility-based position sizing"
                )

            # Calculate position size based on target volatility
            target_volatility = self.value  # e.g., 0.01 for 1% target volatility
            position_size = (target_volatility * equity) / (volatility * entry_price)

        elif self.sizing_method == PositionSizingMethod.KELLY:
            # Kelly criterion
            if win_rate is None or avg_win is None or avg_loss is None:
                raise ValueError(
                    "Win rate, average win, and average loss are required for Kelly criterion"
                )

            # Kelly formula: f* = (p * b - (1 - p)) / b
            # where p is win rate, b is win/loss ratio
            win_loss_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 1.0
            kelly_fraction = (
                win_rate * win_loss_ratio - (1 - win_rate)
            ) / win_loss_ratio

            # Apply Kelly fraction adjustment (value is the fraction of Kelly to use)
            kelly_fraction = kelly_fraction * self.value

            # Ensure Kelly fraction is positive and not too large
            kelly_fraction = max(0.0, min(kelly_fraction, 1.0))

            # Calculate position size
            position_size = (kelly_fraction * equity) / entry_price

        elif self.sizing_method == PositionSizingMethod.OPTIMAL_F:
            # Optimal f
            if avg_win is None or avg_loss is None:
                raise ValueError(
                    "Average win and average loss are required for Optimal f"
                )

            # Optimal f formula: f* = ((avg_win / abs(avg_loss)) - 1) / (avg_win / abs(avg_loss))
            win_loss_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 1.0
            optimal_f = (
                (win_loss_ratio - 1) / win_loss_ratio if win_loss_ratio > 1 else 0.0
            )

            # Apply adjustment (value is the fraction of Optimal f to use)
            optimal_f = optimal_f * self.value

            # Ensure Optimal f is positive and not too large
            optimal_f = max(0.0, min(optimal_f, 1.0))

            # Calculate position size
            position_size = (optimal_f * equity) / entry_price

        elif self.sizing_method == PositionSizingMethod.RISK_PARITY:
            # Risk parity
            if volatility is None or correlation is None:
                raise ValueError(
                    "Volatility and correlation are required for risk parity"
                )

            # Risk contribution formula: RC_i = w_i * sigma_i * (w * Sigma * w)^(1/2)
            # For a single asset, this simplifies to w_i * sigma_i
            # We want RC_i = target_risk, so w_i = target_risk / sigma_i
            target_risk = self.value  # e.g., 0.01 for 1% target risk
            position_size = (target_risk * equity) / (volatility * entry_price)

            # Adjust for correlation (simplified)
            position_size = position_size * (1 - correlation)

        elif self.sizing_method == PositionSizingMethod.FIXED_RISK:
            # Fixed risk per trade
            if stop_price is None:
                raise ValueError(
                    "Stop price is required for fixed risk position sizing"
                )

            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_price)

            if risk_per_share > 0:
                # Calculate position size based on risk amount
                risk_amount = equity * self.value  # e.g., 0.01 for 1% risk
                position_size = risk_amount / risk_per_share
            else:
                position_size = 0.0

        elif self.sizing_method == PositionSizingMethod.CUSTOM:
            # Custom position sizing requires external logic
            position_size = 0.0

        # Apply maximum position size if specified
        if self.max_position_size is not None:
            position_size = min(position_size, self.max_position_size)

        # Apply maximum risk per trade if specified
        if self.max_risk_per_trade is not None and stop_price is not None:
            risk_per_share = abs(entry_price - stop_price)
            max_position_by_risk = (
                (equity * self.max_risk_per_trade) / risk_per_share
                if risk_per_share > 0
                else float("inf")
            )
            position_size = min(position_size, max_position_by_risk)

        return position_size


class RiskMetrics:
    """Risk metrics calculator for portfolio and strategy evaluation."""

    def __init__(
        self,
        returns: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        """
        Initialize risk metrics calculator.

        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns

    def set_returns(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
    ):
        """
        Set returns data.

        Args:
            returns: Series of returns
            benchmark_returns: Series of benchmark returns
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns

    def calculate_metric(self, metric: RiskMetric, **kwargs) -> float:
        """
        Calculate risk metric.

        Args:
            metric: Risk metric to calculate
            **kwargs: Additional parameters for specific metrics

        Returns:
            Calculated risk metric
        """
        if self.returns is None:
            raise ValueError("Returns data not set")

        if metric == RiskMetric.VOLATILITY:
            return self._calculate_volatility(**kwargs)
        elif metric == RiskMetric.VAR:
            return self._calculate_var(**kwargs)
        elif metric == RiskMetric.CVAR:
            return self._calculate_cvar(**kwargs)
        elif metric == RiskMetric.DRAWDOWN:
            return self._calculate_drawdown(**kwargs)
        elif metric == RiskMetric.SHARPE:
            return self._calculate_sharpe(**kwargs)
        elif metric == RiskMetric.SORTINO:
            return self._calculate_sortino(**kwargs)
        elif metric == RiskMetric.CALMAR:
            return self._calculate_calmar(**kwargs)
        elif metric == RiskMetric.BETA:
            return self._calculate_beta(**kwargs)
        elif metric == RiskMetric.CORRELATION:
            return self._calculate_correlation(**kwargs)
        elif metric == RiskMetric.ALPHA:
            return self._calculate_alpha(**kwargs)
        elif metric == RiskMetric.OMEGA:
            return self._calculate_omega(**kwargs)
        elif metric == RiskMetric.TAIL_RATIO:
            return self._calculate_tail_ratio(**kwargs)
        elif metric == RiskMetric.CUSTOM:
            if "custom_function" not in kwargs:
                raise ValueError("Custom function is required for custom risk metric")

            return kwargs["custom_function"](self.returns, **kwargs)

        raise ValueError(f"Unknown risk metric: {metric}")

    def calculate_all_metrics(self, **kwargs) -> Dict[str, float]:
        """
        Calculate all risk metrics.

        Args:
            **kwargs: Additional parameters for specific metrics

        Returns:
            Dictionary of risk metrics
        """
        metrics = {}

        for metric in RiskMetric:
            if metric != RiskMetric.CUSTOM:
                try:
                    metrics[metric.value] = self.calculate_metric(metric, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric.value}: {e}")
                    metrics[metric.value] = None

        return metrics

    def _calculate_volatility(
        self, annualize: bool = True, trading_days: int = 252
    ) -> float:
        """
        Calculate volatility (standard deviation of returns).

        Args:
            annualize: Whether to annualize the result
            trading_days: Number of trading days in a year

        Returns:
            Volatility
        """
        volatility = self.returns.std()

        if annualize:
            volatility = volatility * np.sqrt(trading_days)

        return volatility

    def _calculate_var(
        self, confidence: float = 0.95, window: Optional[int] = None
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            confidence: Confidence level
            window: Rolling window size (if None, use all data)

        Returns:
            Value at Risk
        """
        if window is not None:
            # Use rolling window
            return self.returns.rolling(window=window).quantile(1 - confidence).min()

        # Use all data
        return self.returns.quantile(1 - confidence)

    def _calculate_cvar(
        self, confidence: float = 0.95, window: Optional[int] = None
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            confidence: Confidence level
            window: Rolling window size (if None, use all data)

        Returns:
            Conditional Value at Risk
        """
        if window is not None:
            # Use rolling window
            var = self.returns.rolling(window=window).quantile(1 - confidence)
            cvar_values = []

            for i in range(window, len(self.returns) + 1):
                window_returns = self.returns.iloc[i - window : i]
                window_var = var.iloc[i - 1]
                cvar_values.append(window_returns[window_returns <= window_var].mean())

            return min(cvar_values)

        # Use all data
        var = self.returns.quantile(1 - confidence)
        return self.returns[self.returns <= var].mean()

    def _calculate_drawdown(self) -> float:
        """
        Calculate maximum drawdown.

        Returns:
            Maximum drawdown
        """
        # Calculate cumulative returns
        cum_returns = (1 + self.returns).cumprod()

        # Calculate running maximum
        running_max = cum_returns.cummax()

        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max

        # Return maximum drawdown
        return drawdown.min()

    def _calculate_sharpe(
        self,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            risk_free_rate: Risk-free rate
            annualize: Whether to annualize the result
            trading_days: Number of trading days in a year

        Returns:
            Sharpe ratio
        """
        # Calculate excess returns
        excess_returns = (
            self.returns - risk_free_rate / trading_days
            if annualize
            else self.returns - risk_free_rate
        )

        # Calculate mean and standard deviation
        mean_excess_returns = excess_returns.mean()
        std_excess_returns = excess_returns.std()

        if std_excess_returns == 0:
            return 0.0

        # Calculate Sharpe ratio
        sharpe = mean_excess_returns / std_excess_returns

        if annualize:
            sharpe = sharpe * np.sqrt(trading_days)

        return sharpe

    def _calculate_sortino(
        self,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> float:
        """
        Calculate Sortino ratio.

        Args:
            risk_free_rate: Risk-free rate
            annualize: Whether to annualize the result
            trading_days: Number of trading days in a year

        Returns:
            Sortino ratio
        """
        # Calculate excess returns
        excess_returns = (
            self.returns - risk_free_rate / trading_days
            if annualize
            else self.returns - risk_free_rate
        )

        # Calculate mean of excess returns
        mean_excess_returns = excess_returns.mean()

        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(
            (downside_returns**2).mean()
        )  # Mean of squared negative returns

        if downside_deviation == 0:
            return 0.0

        # Calculate Sortino ratio
        sortino = mean_excess_returns / downside_deviation

        if annualize:
            sortino = sortino * np.sqrt(trading_days)

        return sortino
