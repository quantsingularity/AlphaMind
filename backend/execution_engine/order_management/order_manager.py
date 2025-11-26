"""
## Order Management System Module.
#
## This module provides comprehensive order management functionality including
## order creation, validation, routing, execution, and lifecycle management.
"""

from dataclasses import dataclass, field
import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
import uuid


# Configure logging
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders that can be placed."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Status of an order throughout its lifecycle."""

    CREATED = "created"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ERROR = "error"


class OrderTimeInForce(Enum):
    """Time in force options for orders."""

    DAY = "day"
    GTC = "good_till_cancelled"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    GTD = "good_till_date"


@dataclass
class OrderValidationError:
    """Error information for order validation failures."""

    error_code: str
    message: str
    field: Optional[str] = None
    severity: str = "error"


@dataclass
class OrderFill:
    """Information about an order fill (execution)."""

    fill_id: str
    order_id: str
    timestamp: datetime.datetime
    quantity: float
    price: float
    venue: str
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Comprehensive order information."""

    order_id: str
    instrument_id: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    status: OrderStatus = OrderStatus.CREATED
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.DAY
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    expires_at: Optional[datetime.datetime] = None
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fills: List[OrderFill] = field(default_factory=list)
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    client_order_id: Optional[str] = None
    validation_errors: List[OrderValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if the order is valid."""

        # Returns:
        #     True if the order has no validation errors, False otherwise
        # """"""
        return len(self.validation_errors) == 0

    def remaining_quantity(self) -> float:
        """Calculate the remaining quantity to be filled."""

        # Returns:
        #     Remaining quantity
        # """"""
        return self.quantity - self.filled_quantity

    def is_complete(self) -> bool:
        """Check if the order is in a terminal state."""

        # Returns:
        #     True if the order is complete (filled, cancelled, rejected, expired, or error)
        # """"""
        terminal_states = [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        ]
        return self.status in terminal_states

    def update_status(self, new_status: OrderStatus) -> None:
        """Update the order status."""

        # Args:
        #     new_status: New status to set
        # """"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.datetime.now()
        logger.info(
            f"Order {self.order_id} status changed from {old_status.value} to {new_status.value}"
        )

    def add_fill(self, fill: OrderFill) -> None:
        """Add a fill to the order."""

        # Args:
        #     fill: OrderFill object to add
        # """"""
        self.fills.append(fill)
        self.filled_quantity += fill.quantity

        # Update average fill price
        total_value = sum(f.quantity * f.price for f in self.fills)
        self.average_fill_price = (
            total_value / self.filled_quantity if self.filled_quantity > 0 else None
        )

        # Update status based on fill
        if (
            abs(self.filled_quantity - self.quantity) < 1e-6
        ):  # Account for floating point precision
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)

        self.updated_at = datetime.datetime.now()
        logger.info(
            f"Added fill {fill.fill_id} to order {self.order_id}, filled quantity: {self.filled_quantity}/{self.quantity}"
        )


class OrderValidator:
    """Validates orders before submission."""

    def __init__(self):
        """Initialize order validator."""
        self.validation_rules = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        # Basic field validation
        self.validation_rules.append(self._validate_required_fields)
        self.validation_rules.append(self._validate_price_fields)
        self.validation_rules.append(self._validate_time_in_force)

    def add_validation_rule(self, rule_func) -> None:
        """Add a custom validation rule."""

        # Args:
        #     rule_func: Function that takes an Order and returns a list of OrderValidationError
        # """"""
        self.validation_rules.append(rule_func)

    def validate(self, order: Order) -> List[OrderValidationError]:
        """Validate an order against all rules."""

        # Args:
        #     order: Order to validate
        #
        # Returns:
        #     List of validation errors (empty if order is valid)
        # """"""
        all_errors = []

        for rule in self.validation_rules:
            try:
                errors = rule(order)
                if errors:
                    all_errors.extend(errors)
            except Exception as e:
                logger.error(f"Error in validation rule {rule.__name__}: {str(e)}")
                all_errors.append(
                    OrderValidationError(
                        error_code="VALIDATION_RULE_ERROR",
                        message=f"Internal validation error: {str(e)}",
                        severity="error",
                    )
                )

        # Update order with validation results
        order.validation_errors = all_errors

        if all_errors:
            order.update_status(OrderStatus.REJECTED)
        else:
            order.update_status(OrderStatus.VALIDATED)

        return all_errors

    def _validate_required_fields(self, order: Order) -> List[OrderValidationError]:
        """Validate that all required fields are present."""

        # Args:
        #     order: Order to validate
        #
        # Returns:
        #     List of validation errors
        # """"""
        errors = []

        if not order.instrument_id:
            errors.append(
                OrderValidationError(
                    error_code="MISSING_INSTRUMENT",
                    message="Instrument ID is required",
                    field="instrument_id",
                )
            )

        if order.quantity <= 0:
            errors.append(
                OrderValidationError(
                    error_code="INVALID_QUANTITY",
                    message="Quantity must be positive",
                    field="quantity",
                )
            )

        return errors

    def _validate_price_fields(self, order: Order) -> List[OrderValidationError]:
        """Validate price fields based on order type."""

        # Args:
        #     order: Order to validate
        #
        # Returns:
        #     List of validation errors
        # """"""
        errors = []

        # Limit price required for limit orders
        if order.order_type in [
            OrderType.LIMIT,
            OrderType.STOP_LIMIT,
            OrderType.ICEBERG,
        ]:
            if order.limit_price is None or order.limit_price <= 0:
                errors.append(
                    OrderValidationError(
                        error_code="MISSING_LIMIT_PRICE",
                        message=f"Limit price is required for {order.order_type.value} orders",
                        field="limit_price",
                    )
                )

        # Stop price required for stop orders
        if order.order_type in [
            OrderType.STOP,
            OrderType.STOP_LIMIT,
            OrderType.TRAILING_STOP,
        ]:
            if order.stop_price is None or order.stop_price <= 0:
                errors.append(
                    OrderValidationError(
                        error_code="MISSING_STOP_PRICE",
                        message=f"Stop price is required for {order.order_type.value} orders",
                        field="stop_price",
                    )
                )

        return errors

    def _validate_time_in_force(self, order: Order) -> List[OrderValidationError]:
        """Validate time in force settings."""

        # Args:
        #     order: Order to validate
        #
        # Returns:
        #     List of validation errors
        # """""
        errors = []

        if order.time_in_force == OrderTimeInForce.GTD and order.expires_at is None:
            errors.append(
                OrderValidationError(
                    error_code="MISSING_EXPIRY",
                    message="Expiry date is required for GTD orders",
                    field="expires_at",
                )
            )

        return errors


class OrderManager:
    """Manages the lifecycle of orders."""

    def __init__(self):
        """Initialize order manager."""
        self.orders: Dict[str, Order] = {}
        self.validator = OrderValidator()

    def create_order(self, **kwargs) -> Order:
        """Create a new order."""

        # Args:
        #     **kwargs: Order attributes
        #
        # Returns:
        #     Newly created Order object
        # """"""
        # Generate order ID if not provided
        if "order_id" not in kwargs:
            kwargs["order_id"] = str(uuid.uuid4())

        # Convert enum strings to enum values
        if "order_type" in kwargs and isinstance(kwargs["order_type"], str):
            kwargs["order_type"] = OrderType(kwargs["order_type"])

        if "side" in kwargs and isinstance(kwargs["side"], str):
            kwargs["side"] = OrderSide(kwargs["side"])

        if "time_in_force" in kwargs and isinstance(kwargs["time_in_force"], str):
            kwargs["time_in_force"] = OrderTimeInForce(kwargs["time_in_force"])

        # Create order object
        order = Order(**kwargs)

        # Store order
        self.orders[order.order_id] = order
        logger.info(
            f"Created order {order.order_id} for {order.quantity} {order.instrument_id}"
        )

        return order

    def validate_order(self, order_id: str) -> bool:
        """Validate an order."""

        # Args:
        #     order_id: ID of the order to validate
        #
        # Returns:
        #     True if the order is valid, False otherwise
        #
        # Raises:
        #     KeyError: If the order_id doesn't exist
        # """"""
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found")

        order = self.orders[order_id]
        errors = self.validator.validate(order)

        return len(errors) == 0

    def submit_order(self, order_id: str) -> bool:
        """Submit an order for execution."""

        # Args:
        #     order_id: ID of the order to submit
        #
        # Returns:
        #     True if the order was submitted successfully, False otherwise
        #
        # Raises:
        #     KeyError: If the order_id doesn't exist
        # """"""
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found")

        order = self.orders[order_id]

        # Validate order if not already validated
        if order.status == OrderStatus.CREATED:
            if not self.validate_order(order_id):
                return False

        # Check if order is in a valid state for submission
        if order.status not in [OrderStatus.VALIDATED, OrderStatus.CREATED]:
            logger.warning(
                f"Cannot submit order {order_id} with status {order.status.value}"
            )
            return False

        # Update status to pending
        order.update_status(OrderStatus.PENDING)
        logger.info(f"Submitted order {order_id} for execution")

        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""

        # Args:
        #     order_id: ID of the order to cancel
        #
        # Returns:
        #     True if the order was cancelled successfully, False otherwise
        #
        # Raises:
        #     KeyError: If the order_id doesn't exist
        # """"""
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found")

        order = self.orders[order_id]

        # Check if order can be cancelled
        if order.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        ]:
            logger.warning(
                f"Cannot cancel order {order_id} with status {order.status.value}"
            )
            return False

        # Update status to cancelled
        order.update_status(OrderStatus.CANCELLED)
        logger.info(f"Cancelled order {order_id}")

        return True

    def add_fill(self, order_id: str, fill: OrderFill) -> bool:
        """Add a fill to an order."""

        # Args:
        #     order_id: ID of the order
        #     fill: OrderFill object to add
        #
        # Returns:
        #     True if the fill was added successfully, False otherwise
        #
        # Raises:
        #     KeyError: If the order_id doesn't exist
        # """"""
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found")

        order = self.orders[order_id]

        # Check if order can be filled
        if order.status in [
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        ]:
            logger.warning(
                f"Cannot add fill to order {order_id} with status {order.status.value}"
            )
            return False

        # Check if fill would exceed order quantity
        if (
            order.filled_quantity + fill.quantity > order.quantity + 1e-6
        ):  # Account for floating point precision
            logger.warning(f"Fill would exceed order quantity for order {order_id}")
            return False

        # Add fill to order
        order.add_fill(fill)

        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID."""

        # Args:
        #     order_id: ID of the order
        #
        # Returns:
        #     Order object or None if not found
        # """"""
        return self.orders.get(order_id)

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with a specific status."""

        # Args:
        #     status: Status to filter by
        #
        # Returns:
        #     List of matching Order objects
        # """"""
        return [order for order in self.orders.values() if order.status == status]

    def get_orders_by_instrument(self, instrument_id: str) -> List[Order]:
        """Get all orders for a specific instrument."""

        # Args:
        #     instrument_id: Instrument ID to filter by
        #
        # Returns:
        #     List of matching Order objects
        # """"""
        return [
            order
            for order in self.orders.values()
            if order.instrument_id == instrument_id
        ]

    def get_active_orders(self) -> List[Order]:
        """Get all active (non-terminal) orders."""

        # Returns:
        #     List of active Order objects
        # """"""
        active_statuses = [
            OrderStatus.CREATED,
            OrderStatus.VALIDATED,
            OrderStatus.PENDING,
            OrderStatus.PARTIALLY_FILLED,
        ]
        return [
            order for order in self.orders.values() if order.status in active_statuses
        ]

    def get_order_history(self, order_id: str) -> Dict:
        """Get the history of an order including all fills."""

        # Args:
        #     order_id: ID of the order
        #
        # Returns:
        #     Dictionary containing order history
        #
        # Raises:
        #     KeyError: If the order_id doesn't exist
        # """"""
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found")

        order = self.orders[order_id]

        history = {
            "order_id": order.order_id,
            "instrument_id": order.instrument_id,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "status": order.status.value,
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat(),
            "filled_quantity": order.filled_quantity,
            "average_fill_price": order.average_fill_price,
            "fills": [],
        }

        for fill in order.fills:
            history["fills"].append(
                {
                    "fill_id": fill.fill_id,
                    "timestamp": fill.timestamp.isoformat(),
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "venue": fill.venue,
                    "fees": fill.fees,
                }
            )

        return history
