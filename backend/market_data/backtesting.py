"""
Backtesting Module for AlphaMind

This module provides comprehensive backtesting capabilities for trading strategies,
including historical data processing, strategy execution simulation, performance
analysis, and visualization tools.
"""

from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported in backtesting."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides supported in backtesting."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses in backtesting."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class BacktestOrder:
    """Represents an order in the backtesting system."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        order_id: Optional[str] = None,
    ):
        """
        Initialize a backtest order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            order_type: Order type (market, limit, etc.)
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            order_id: Order ID (generated if not provided)
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.order_id = order_id or f"order_{datetime.now().timestamp():.6f}"

        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.filled_price = 0.0
        self.filled_time = None
        self.created_time = datetime.now()
        self.updated_time = self.created_time
        self.commission = 0.0
        self.slippage = 0.0

        # Validate order
        self._validate()

    def _validate(self):
        """Validate order parameters."""
        if (
            self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]
            and self.price is None
        ):
            raise ValueError(f"Price is required for {self.order_type.value} orders")

        if (
            self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]
            and self.stop_price is None
        ):
            raise ValueError(
                f"Stop price is required for {self.order_type.value} orders"
            )

    def fill(
        self,
        price: float,
        quantity: Optional[float] = None,
        time: Optional[datetime] = None,
    ):
        """
        Fill the order (partially or completely).

        Args:
            price: Fill price
            quantity: Fill quantity (defaults to remaining quantity)
            time: Fill time (defaults to current time)
        """
        fill_qty = quantity or (self.quantity - self.filled_quantity)
        fill_qty = min(fill_qty, self.quantity - self.filled_quantity)

        if fill_qty <= 0:
            return

        # Update filled information
        self.filled_quantity += fill_qty

        # Calculate average filled price
        if self.filled_price > 0:
            self.filled_price = (
                self.filled_price * (self.filled_quantity - fill_qty) + price * fill_qty
            ) / self.filled_quantity
        else:
            self.filled_price = price

        self.filled_time = time or datetime.now()
        self.updated_time = self.filled_time

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self):
        """Cancel the order if not filled."""
        if self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
            self.status = OrderStatus.CANCELED
            self.updated_time = datetime.now()

    def reject(self, reason: str = ""):
        """
        Reject the order.

        Args:
            reason: Rejection reason
        """
        self.status = OrderStatus.REJECTED
        self.updated_time = datetime.now()
        logger.warning(f"Order {self.order_id} rejected: {reason}")

    def to_dict(self) -> Dict:
        """
        Convert order to dictionary.

        Returns:
            Order as dictionary
        """
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "filled_time": self.filled_time.isoformat() if self.filled_time else None,
            "created_time": self.created_time.isoformat(),
            "updated_time": self.updated_time.isoformat(),
            "commission": self.commission,
            "slippage": self.slippage,
        }


class Position:
    """Represents a trading position in the backtesting system."""

    def __init__(self, symbol: str, quantity: float = 0.0, avg_price: float = 0.0):
        """
        Initialize a position.

        Args:
            symbol: Trading symbol
            quantity: Position quantity (positive for long, negative for short)
            avg_price: Average entry price
        """
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.last_price = avg_price
        self.trades = []

    def update(self, price: float):
        """
        Update position with current market price.

        Args:
            price: Current market price
        """
        self.last_price = price
        self.unrealized_pnl = self.quantity * (price - self.avg_price)

    def apply_trade(
        self, side: OrderSide, quantity: float, price: float, commission: float = 0.0
    ):
        """
        Apply a trade to the position.

        Args:
            side: Trade side (buy/sell)
            quantity: Trade quantity
            price: Trade price
            commission: Trade commission

        Returns:
            Realized PnL from this trade
        """
        trade_qty = quantity
        if side == OrderSide.SELL:
            trade_qty = -quantity

        # Calculate realized PnL for reducing positions
        realized_pnl = 0.0
        if (self.quantity > 0 and trade_qty < 0) or (
            self.quantity < 0 and trade_qty > 0
        ):
            # Reducing position
            reduction_qty = min(abs(self.quantity), abs(trade_qty))
            if self.quantity > 0:
                realized_pnl = reduction_qty * (price - self.avg_price) - commission
            else:
                realized_pnl = reduction_qty * (self.avg_price - price) - commission

        # Update position
        new_quantity = self.quantity + trade_qty

        # Update average price for increasing positions
        if (self.quantity >= 0 and trade_qty > 0) or (
            self.quantity <= 0 and trade_qty < 0
        ):
            # Increasing position
            self.avg_price = (
                abs(self.quantity) * self.avg_price + abs(trade_qty) * price
            ) / (abs(self.quantity) + abs(trade_qty))

        # Handle position flipping
        if self.quantity * new_quantity < 0:  # Sign change
            self.avg_price = price

        self.quantity = new_quantity
        self.realized_pnl += realized_pnl

        # Record trade
        self.trades.append(
            {
                "time": datetime.now(),
                "side": side.value,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "realized_pnl": realized_pnl,
            }
        )

        return realized_pnl

    def to_dict(self) -> Dict:
        """
        Convert position to dictionary.

        Returns:
            Position as dictionary
        """
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "last_price": self.last_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "trades_count": len(self.trades),
        }


class Portfolio:
    """Represents a portfolio in the backtesting system."""

    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize a portfolio.

        Args:
            initial_cash: Initial cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.equity = initial_cash
        self.history = []

    def get_position(self, symbol: str) -> Position:
        """
        Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]

    def update(self, prices: Dict[str, float]):
        """
        Update portfolio with current market prices.

        Args:
            prices: Dictionary of symbol -> price
        """
        portfolio_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update(prices[symbol])
                portfolio_value += position.quantity * position.last_price

        self.equity = portfolio_value

        # Record history
        self.history.append(
            {
                "time": datetime.now(),
                "cash": self.cash,
                "equity": self.equity,
                "positions": {s: p.to_dict() for s, p in self.positions.items()},
            }
        )

    def apply_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        commission: float = 0.0,
    ):
        """
        Apply a trade to the portfolio.

        Args:
            symbol: Trading symbol
            side: Trade side (buy/sell)
            quantity: Trade quantity
            price: Trade price
            commission: Trade commission

        Returns:
            True if trade was successful, False otherwise
        """
        position = self.get_position(symbol)

        # Check if we have enough cash for buys
        if side == OrderSide.BUY:
            cost = quantity * price + commission
            if cost > self.cash:
                logger.warning(f"Insufficient cash for trade: {cost} > {self.cash}")
                return False

            # Update cash
            self.cash -= cost

        # Apply trade to position
        realized_pnl = position.apply_trade(side, quantity, price, commission)

        # Update cash for sells
        if side == OrderSide.SELL:
            self.cash += quantity * price - commission

        # Remove position if quantity is zero
        if position.quantity == 0:
            self.positions.pop(symbol, None)

        return True

    def to_dict(self) -> Dict:
        """
        Convert portfolio to dictionary.

        Returns:
            Portfolio as dictionary
        """
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "equity": self.equity,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "positions_count": len(self.positions),
        }


class MarketSimulator:
    """Simulates market behavior for backtesting."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        commission_rate: float = 0.001,
        slippage_model: str = "fixed",
        slippage_params: Dict = None,
    ):
        """
        Initialize market simulator.

        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV data
            commission_rate: Commission rate as percentage of trade value
            slippage_model: Slippage model type (fixed, normal, proportional)
            slippage_params: Parameters for slippage model
        """
        self.data = data
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_params = slippage_params or {}

        # Default slippage parameters
        if self.slippage_model == "fixed" and "value" not in self.slippage_params:
            self.slippage_params["value"] = 0.0001  # 1 basis point
        elif self.slippage_model == "normal" and "std" not in self.slippage_params:
            self.slippage_params["std"] = 0.0005  # 5 basis points std dev
        elif (
            self.slippage_model == "proportional"
            and "factor" not in self.slippage_params
        ):
            self.slippage_params["factor"] = 0.1  # 10% of volatility

    def get_price_for_order(self, order: BacktestOrder, bar: pd.Series) -> float:
        """
        Get execution price for an order based on the current bar.

        Args:
            order: Order to execute
            bar: Current price bar (OHLCV)

        Returns:
            Execution price
        """
        # Base price depends on order type
        if order.order_type == OrderType.MARKET:
            # For market orders, use the next bar's open or current bar's close
            base_price = bar["close"]
        elif order.order_type == OrderType.LIMIT:
            # For limit orders, use the limit price
            if order.side == OrderSide.BUY and bar["low"] <= order.price:
                # Buy limit order executed if low price <= limit price
                base_price = min(
                    bar["open"], order.price
                )  # Can't get better than limit price
            elif order.side == OrderSide.SELL and bar["high"] >= order.price:
                # Sell limit order executed if high price >= limit price
                base_price = max(
                    bar["open"], order.price
                )  # Can't get better than limit price
            else:
                # Limit order not executed
                return None
        elif order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            # For stop orders, check if stop price is triggered
            if order.side == OrderSide.BUY and bar["high"] >= order.stop_price:
                # Buy stop triggered if high price >= stop price
                if order.order_type == OrderType.STOP:
                    base_price = max(
                        bar["open"], order.stop_price
                    )  # Can't get better than stop price
                else:  # STOP_LIMIT
                    if bar["low"] <= order.price:
                        base_price = min(
                            max(bar["open"], order.stop_price), order.price
                        )
                    else:
                        # Stop triggered but limit not reached
                        return None
            elif order.side == OrderSide.SELL and bar["low"] <= order.stop_price:
                # Sell stop triggered if low price <= stop price
                if order.order_type == OrderType.STOP:
                    base_price = min(
                        bar["open"], order.stop_price
                    )  # Can't get better than stop price
                else:  # STOP_LIMIT
                    if bar["high"] >= order.price:
                        base_price = max(
                            min(bar["open"], order.stop_price), order.price
                        )
                    else:
                        # Stop triggered but limit not reached
                        return None
            else:
                # Stop not triggered
                return None
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        # Apply slippage
        price_with_slippage = self._apply_slippage(base_price, order.side, bar)

        return price_with_slippage

    def _apply_slippage(self, price: float, side: OrderSide, bar: pd.Series) -> float:
        """
        Apply slippage to the execution price.

        Args:
            price: Base execution price
            side: Order side
            bar: Current price bar

        Returns:
            Price with slippage applied
        """
        if self.slippage_model == "fixed":
            # Fixed basis point slippage
            slippage_value = price * self.slippage_params["value"]
            if side == OrderSide.BUY:
                return price * (1 + self.slippage_params["value"])
            else:
                return price * (1 - self.slippage_params["value"])

        elif self.slippage_model == "normal":
            # Normal distribution slippage
            std = self.slippage_params["std"]
            slippage_value = price * np.random.normal(0, std)
            if side == OrderSide.BUY:
                return price + abs(slippage_value)
            else:
                return price - abs(slippage_value)

        elif self.slippage_model == "proportional":
            # Proportional to volatility
            factor = self.slippage_params["factor"]
            volatility = (bar["high"] - bar["low"]) / bar["close"]
            slippage_value = price * volatility * factor
            if side == OrderSide.BUY:
                return price + slippage_value
            else:
                return price - slippage_value

        else:
            return price  # No slippage

    def calculate_commission(self, price: float, quantity: float) -> float:
        """
        Calculate commission for a trade.

        Args:
            price: Execution price
            quantity: Execution quantity

        Returns:
            Commission amount
        """
        return price * quantity * self.commission_rate


class BacktestEngine:
    """Main backtesting engine."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_model: str = "fixed",
        slippage_params: Dict = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Initialize backtesting engine.

        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV data
            initial_cash: Initial cash balance
            commission_rate: Commission rate as percentage of trade value
            slippage_model: Slippage model type (fixed, normal, proportional)
            slippage_params: Parameters for slippage model
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
        """
        self.data = data
        self.portfolio = Portfolio(initial_cash)
        self.market_simulator = MarketSimulator(
            data, commission_rate, slippage_model, slippage_params
        )

        # Prepare data
        self._prepare_data(start_date, end_date)

        # Initialize state
        self.current_time = None
        self.current_bar = {}
        self.pending_orders = []
        self.filled_orders = []
        self.canceled_orders = []
        self.rejected_orders = []

        # Performance tracking
        self.performance = {
            "equity_curve": [],
            "returns": [],
            "drawdowns": [],
            "trades": [],
        }

    def _prepare_data(self, start_date: Optional[str], end_date: Optional[str]):
        """
        Prepare data for backtesting.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
        """
        # Ensure all dataframes have datetime index
        for symbol, df in self.data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df.set_index("timestamp", inplace=True)
                else:
                    raise ValueError(
                        f"DataFrame for {symbol} must have datetime index or timestamp column"
                    )

        # Filter by date range
        if start_date or end_date:
            for symbol, df in self.data.items():
                if start_date:
                    self.data[symbol] = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    self.data[symbol] = df[df.index <= pd.to_datetime(end_date)]

        # Get common date range
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)

        self.dates = sorted(all_dates)

        if not self.dates:
            raise ValueError("No data available for the specified date range")

    def place_order(self, order: BacktestOrder) -> str:
        """
        Place an order in the backtesting system.

        Args:
            order: Order to place

        Returns:
            Order ID
        """
        self.pending_orders.append(order)
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if order was canceled, False otherwise
        """
        for i, order in enumerate(self.pending_orders):
            if order.order_id == order_id:
                order.cancel()
                self.canceled_orders.append(order)
                self.pending_orders.pop(i)
                return True

        return False

    def _process_orders(self):
        """Process pending orders with current bar data."""
        remaining_orders = []

        for order in self.pending_orders:
            # Skip if symbol not in current bar
            if order.symbol not in self.current_bar:
                remaining_orders.append(order)
                continue

            # Get execution price
            bar = self.current_bar[order.symbol]
            execution_price = self.market_simulator.get_price_for_order(order, bar)

            if execution_price is not None:
                # Calculate commission
                commission = self.market_simulator.calculate_commission(
                    execution_price, order.quantity
                )

                # Apply trade to portfolio
                success = self.portfolio.apply_trade(
                    order.symbol,
                    order.side,
                    order.quantity,
                    execution_price,
                    commission,
                )

                if success:
                    # Fill order
                    order.fill(execution_price, time=self.current_time)
                    order.commission = commission
                    self.filled_orders.append(order)

                    # Record trade
                    self.performance["trades"].append(
                        {
                            "time": self.current_time,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "quantity": order.quantity,
                            "price": execution_price,
                            "commission": commission,
                            "order_type": order.order_type.value,
                        }
                    )
                else:
                    # Reject order
                    order.reject("Insufficient funds")
                    self.rejected_orders.append(order)
            else:
                # Order not executed this bar
                remaining_orders.append(order)

        self.pending_orders = remaining_orders

    def _update_performance(self):
        """Update performance metrics."""
        self.performance["equity_curve"].append(
            {
                "time": self.current_time,
                "equity": self.portfolio.equity,
                "cash": self.portfolio.cash,
            }
        )

        # Calculate returns
        if len(self.performance["equity_curve"]) > 1:
            prev_equity = self.performance["equity_curve"][-2]["equity"]
            curr_equity = self.performance["equity_curve"][-1]["equity"]
            returns = (curr_equity / prev_equity) - 1
            self.performance["returns"].append(
                {"time": self.current_time, "returns": returns}
            )

        # Calculate drawdowns
        equity_series = [p["equity"] for p in self.performance["equity_curve"]]
        if equity_series:
            max_equity = max(equity_series)
            current_equity = equity_series[-1]
            drawdown = (
                (max_equity - current_equity) / max_equity if max_equity > 0 else 0
            )
            self.performance["drawdowns"].append(
                {"time": self.current_time, "drawdown": drawdown}
            )

    def run(self, strategy_fn: Callable[[Dict, Portfolio, BacktestEngine], None]):
        """
        Run backtest with the provided strategy function.

        Args:
            strategy_fn: Strategy function that receives current bar data, portfolio, and engine
        """
        for date in self.dates:
            self.current_time = date

            # Get current bar data for all symbols
            self.current_bar = {}
            for symbol, df in self.data.items():
                if date in df.index:
                    self.current_bar[symbol] = df.loc[date]

            # Update portfolio with current prices
            current_prices = {s: bar["close"] for s, bar in self.current_bar.items()}
            self.portfolio.update(current_prices)

            # Execute strategy
            strategy_fn(self.current_bar, self.portfolio, self)

            # Process orders
            self._process_orders()

            # Update performance metrics
            self._update_performance()

        # Calculate final performance statistics
        self._calculate_performance_stats()

    def _calculate_performance_stats(self):
        """Calculate performance statistics after backtest completion."""
        # Convert lists to DataFrames for easier analysis
        equity_df = pd.DataFrame(self.performance["equity_curve"])
        if not equity_df.empty:
            equity_df.set_index("time", inplace=True)

            returns_df = pd.DataFrame(self.performance["returns"])
            if not returns_df.empty:
                returns_df.set_index("time", inplace=True)

                # Calculate statistics
                total_return = (
                    equity_df["equity"].iloc[-1] / self.portfolio.initial_cash
                ) - 1
                annual_return = (
                    (1 + total_return) ** (252 / len(returns_df)) - 1
                    if len(returns_df) > 0
                    else 0
                )

                daily_returns = returns_df["returns"]
                volatility = (
                    daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
                )

                sharpe_ratio = annual_return / volatility if volatility > 0 else 0

                drawdown_df = pd.DataFrame(self.performance["drawdowns"])
                max_drawdown = (
                    drawdown_df["drawdown"].max() if not drawdown_df.empty else 0
                )

                # Calculate win rate and profit factor
                trades_df = pd.DataFrame(self.performance["trades"])
                if not trades_df.empty:
                    # Add position and realized PnL to trades
                    trades_df["position"] = (
                        trades_df["side"].apply(lambda x: 1 if x == "buy" else -1)
                        * trades_df["quantity"]
                    )

                    # Group by symbol and calculate PnL
                    trades_by_symbol = trades_df.groupby("symbol")

                    all_pnls = []
                    for symbol, group in trades_by_symbol:
                        # Calculate PnL for each trade
                        position = 0
                        avg_price = 0
                        trade_pnls = []

                        for _, trade in group.iterrows():
                            trade_qty = trade["quantity"]
                            trade_price = trade["price"]
                            trade_side = trade["side"]
                            commission = trade["commission"]

                            if trade_side == "buy":
                                # Buying
                                if position >= 0:
                                    # Increasing long position
                                    avg_price = (
                                        position * avg_price + trade_qty * trade_price
                                    ) / (position + trade_qty)
                                    position += trade_qty
                                    trade_pnls.append(-commission)
                                else:
                                    # Covering short position
                                    cover_qty = min(trade_qty, abs(position))
                                    pnl = (
                                        cover_qty * (avg_price - trade_price)
                                        - commission
                                    )
                                    trade_pnls.append(pnl)

                                    position += trade_qty
                                    if position > 0:
                                        # Remaining quantity becomes long position
                                        avg_price = trade_price
                            else:
                                # Selling
                                if position <= 0:
                                    # Increasing short position
                                    avg_price = (
                                        abs(position) * avg_price
                                        + trade_qty * trade_price
                                    ) / (abs(position) + trade_qty)
                                    position -= trade_qty
                                    trade_pnls.append(-commission)
                                else:
                                    # Closing long position
                                    close_qty = min(trade_qty, position)
                                    pnl = (
                                        close_qty * (trade_price - avg_price)
                                        - commission
                                    )
                                    trade_pnls.append(pnl)

                                    position -= trade_qty
                                    if position < 0:
                                        # Remaining quantity becomes short position
                                        avg_price = trade_price

                        all_pnls.extend(trade_pnls)

                    # Calculate win rate and profit factor
                    winning_trades = sum(1 for pnl in all_pnls if pnl > 0)
                    losing_trades = sum(1 for pnl in all_pnls if pnl < 0)

                    win_rate = winning_trades / len(all_pnls) if all_pnls else 0

                    gross_profit = sum(pnl for pnl in all_pnls if pnl > 0)
                    gross_loss = abs(sum(pnl for pnl in all_pnls if pnl < 0))

                    profit_factor = (
                        gross_profit / gross_loss if gross_loss > 0 else float("inf")
                    )

                    # Store statistics
                    self.performance["statistics"] = {
                        "total_return": total_return,
                        "annual_return": annual_return,
                        "volatility": volatility,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "win_rate": win_rate,
                        "profit_factor": profit_factor,
                        "total_trades": len(all_pnls),
                        "winning_trades": winning_trades,
                        "losing_trades": losing_trades,
                    }

    def get_results(self) -> Dict:
        """
        Get backtest results.

        Returns:
            Dictionary with backtest results
        """
        return {
            "portfolio": self.portfolio.to_dict(),
            "performance": self.performance,
            "orders": {
                "filled": [order.to_dict() for order in self.filled_orders],
                "pending": [order.to_dict() for order in self.pending_orders],
                "canceled": [order.to_dict() for order in self.canceled_orders],
                "rejected": [order.to_dict() for order in self.rejected_orders],
            },
        }

    def plot_results(
        self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None
    ):
        """
        Plot backtest results.

        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.performance["equity_curve"]:
            logger.warning("No data to plot")
            return

        # Convert to DataFrames
        equity_df = pd.DataFrame(self.performance["equity_curve"])
        equity_df.set_index("time", inplace=True)

        drawdown_df = pd.DataFrame(self.performance["drawdowns"])
        drawdown_df.set_index("time", inplace=True)

        # Create figure
        fig, axes = plt.subplots(
            3, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
        )

        # Plot equity curve
        equity_df["equity"].plot(ax=axes[0], label="Portfolio Value")
        equity_df["cash"].plot(ax=axes[0], label="Cash", alpha=0.7)
        axes[0].set_title("Backtest Results")
        axes[0].set_ylabel("Value")
        axes[0].legend()
        axes[0].grid(True)

        # Plot returns
        if "returns" in self.performance and self.performance["returns"]:
            returns_df = pd.DataFrame(self.performance["returns"])
            returns_df.set_index("time", inplace=True)
            returns_df["returns"].plot(ax=axes[1], label="Daily Returns")
            axes[1].set_ylabel("Returns")
            axes[1].legend()
            axes[1].grid(True)

        # Plot drawdowns
        drawdown_df["drawdown"].plot(ax=axes[2], label="Drawdown", color="red")
        axes[2].set_ylabel("Drawdown")
        axes[2].set_xlabel("Date")
        axes[2].legend()
        axes[2].grid(True)

        # Add statistics as text
        if "statistics" in self.performance:
            stats = self.performance["statistics"]
            stats_text = (
                f"Total Return: {stats['total_return']:.2%}\n"
                f"Annual Return: {stats['annual_return']:.2%}\n"
                f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {stats['max_drawdown']:.2%}\n"
                f"Win Rate: {stats['win_rate']:.2%}\n"
                f"Profit Factor: {stats['profit_factor']:.2f}\n"
                f"Total Trades: {stats['total_trades']}"
            )

            # Add text box
            axes[0].text(
                0.02,
                0.95,
                stats_text,
                transform=axes[0].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")

        plt.show()


class BacktestDataLoader:
    """Utility for loading and preparing data for backtesting."""

    @staticmethod
    def load_from_csv(
        file_paths: Dict[str, str],
        date_column: str = "timestamp",
        date_format: Optional[str] = None,
        columns_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from CSV files.

        Args:
            file_paths: Dictionary of symbol -> file path
            date_column: Name of the date/timestamp column
            date_format: Date format string for parsing
            columns_map: Dictionary to map CSV columns to OHLCV format

        Returns:
            Dictionary of symbol -> DataFrame with OHLCV data
        """
        data = {}

        for symbol, file_path in file_paths.items():
            try:
                df = pd.read_csv(file_path)

                # Parse date column
                if date_format:
                    df[date_column] = pd.to_datetime(
                        df[date_column], format=date_format
                    )
                else:
                    df[date_column] = pd.to_datetime(df[date_column])

                # Rename columns if mapping provided
                if columns_map:
                    df.rename(columns=columns_map, inplace=True)

                # Ensure required columns exist
                required_columns = ["open", "high", "low", "close", "volume"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]

                if missing_columns:
                    logger.warning(f"Missing columns for {symbol}: {missing_columns}")

                    # Fill missing columns with defaults
                    for col in missing_columns:
                        if col == "volume":
                            df[col] = 0
                        elif col in ["open", "high", "low"] and "close" in df.columns:
                            df[col] = df["close"]

                # Set index
                df.set_index(date_column, inplace=True)

                # Sort by date
                df.sort_index(inplace=True)

                data[symbol] = df

                logger.info(f"Loaded {len(df)} rows for {symbol} from {file_path}")

            except Exception as e:
                logger.error(f"Error loading data for {symbol} from {file_path}: {e}")

        return data

    @staticmethod
    def load_from_api(
        symbols: List[str],
        api_client: Any,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data from API.

        Args:
            symbols: List of symbols to load
            api_client: API client object with get_historical_data method
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval

        Returns:
            Dictionary of symbol -> DataFrame with OHLCV data
        """
        data = {}

        for symbol in symbols:
            try:
                df = api_client.get_historical_data(
                    symbol, interval=interval, start_date=start_date, end_date=end_date
                )

                # Ensure required columns exist
                required_columns = ["open", "high", "low", "close", "volume"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]

                if missing_columns:
                    logger.warning(f"Missing columns for {symbol}: {missing_columns}")

                    # Fill missing columns with defaults
                    for col in missing_columns:
                        if col == "volume":
                            df[col] = 0
                        elif col in ["open", "high", "low"] and "close" in df.columns:
                            df[col] = df["close"]

                data[symbol] = df

                logger.info(f"Loaded {len(df)} rows for {symbol} from API")

            except Exception as e:
                logger.error(f"Error loading data for {symbol} from API: {e}")

        return data

    @staticmethod
    def resample_data(
        data: Dict[str, pd.DataFrame], interval: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample data to a different time interval.

        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV data
            interval: Target interval (e.g., '1h', '1d', '1w')

        Returns:
            Dictionary of symbol -> resampled DataFrame
        """
        resampled_data = {}

        for symbol, df in data.items():
            try:
                # Ensure DataFrame has datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(f"DataFrame for {symbol} must have datetime index")

                # Resample
                resampled = df.resample(interval).agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )

                # Drop rows with NaN values
                resampled.dropna(inplace=True)

                resampled_data[symbol] = resampled

                logger.info(
                    f"Resampled {symbol} data to {interval} interval, resulting in {len(resampled)} bars"
                )

            except Exception as e:
                logger.error(f"Error resampling data for {symbol}: {e}")

        return resampled_data

    @staticmethod
    def align_data(
        data: Dict[str, pd.DataFrame], fill_method: str = "ffill"
    ) -> Dict[str, pd.DataFrame]:
        """
        Align data to ensure all symbols have the same dates.

        Args:
            data: Dictionary of symbol -> DataFrame with OHLCV data
            fill_method: Method to fill missing values (ffill, bfill, none)

        Returns:
            Dictionary of symbol -> aligned DataFrame
        """
        if not data:
            return {}

        # Get all unique dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)

        all_dates = sorted(all_dates)

        # Reindex all dataframes
        aligned_data = {}

        for symbol, df in data.items():
            reindexed = df.reindex(all_dates)

            if fill_method == "ffill":
                reindexed.fillna(method="ffill", inplace=True)
            elif fill_method == "bfill":
                reindexed.fillna(method="bfill", inplace=True)

            # Drop rows that still have NaN values
            reindexed.dropna(inplace=True)

            aligned_data[symbol] = reindexed

            logger.info(f"Aligned {symbol} data, resulting in {len(reindexed)} bars")

        return aligned_data


# Example usage
def example_strategy(bar_data, portfolio, engine):
    """
    Example moving average crossover strategy.

    Args:
        bar_data: Current bar data
        portfolio: Portfolio object
        engine: Backtest engine
    """
    symbol = "AAPL"  # Example symbol

    if symbol not in bar_data:
        return

    # Get position
    position = portfolio.get_position(symbol)

    # Calculate moving averages
    if not hasattr(example_strategy, "history"):
        example_strategy.history = []

    example_strategy.history.append(bar_data[symbol]["close"])

    if len(example_strategy.history) < 50:
        return

    # Calculate moving averages
    short_ma = sum(example_strategy.history[-20:]) / 20
    long_ma = sum(example_strategy.history[-50:]) / 50

    # Trading logic
    if short_ma > long_ma and position.quantity <= 0:
        # Buy signal
        price = bar_data[symbol]["close"]
        cash = portfolio.cash
        quantity = int(cash * 0.95 / price)  # Use 95% of available cash

        if quantity > 0:
            order = BacktestOrder(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
            engine.place_order(order)

    elif short_ma < long_ma and position.quantity > 0:
        # Sell signal
        order = BacktestOrder(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
        )
        engine.place_order(order)


def run_example_backtest():
    """Run an example backtest."""
    # Load data
    data_loader = BacktestDataLoader()

    # Example: Load from CSV
    data = data_loader.load_from_csv({"AAPL": "data/AAPL.csv", "MSFT": "data/MSFT.csv"})

    # Initialize backtest engine
    engine = BacktestEngine(
        data=data,
        initial_cash=100000.0,
        commission_rate=0.001,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    # Run backtest
    engine.run(example_strategy)

    # Get results
    results = engine.get_results()

    # Print performance statistics
    if "statistics" in results["performance"]:
        stats = results["performance"]["statistics"]
        print(f"Total Return: {stats['total_return']:.2%}")
        print(f"Annual Return: {stats['annual_return']:.2%}")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Total Trades: {stats['total_trades']}")

    # Plot results
    engine.plot_results(save_path="backtest_results.png")


if __name__ == "__main__":
    run_example_backtest()
