"""
Exchange API Integration Module for AlphaMind

This module provides integration with various cryptocurrency and traditional exchanges,
enabling real-time trading, order management, and account operations.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Callable, Any, Tuple

import aiohttp
import pandas as pd
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by exchange API."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides supported by exchange API."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses in exchange API."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options for orders."""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    DAY = "day"  # Day Order


class ExchangeCredentials:
    """Credentials for exchange API authentication."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
        additional_params: Optional[Dict[str, str]] = None
    ):
        """
        Initialize exchange credentials.
        
        Args:
            api_key: API key
            api_secret: API secret
            passphrase: API passphrase (required for some exchanges)
            additional_params: Additional parameters required by specific exchanges
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.additional_params = additional_params or {}


class ExchangeConfig:
    """Configuration for exchange API connections."""
    
    def __init__(self, exchange_id: str, credentials: Optional[ExchangeCredentials] = None):
        """
        Initialize exchange configuration.
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase', 'alpaca')
            credentials: Exchange API credentials
        """
        self.exchange_id = exchange_id.lower()
        self.credentials = credentials
        self.endpoints = self._get_default_endpoints()
        self.rate_limits = self._get_default_rate_limits()
        self.options = self._get_default_options()
    
    def _get_default_endpoints(self) -> Dict[str, str]:
        """
        Get default API endpoints for the exchange.
        
        Returns:
            Dictionary of endpoint types to URLs
        """
        endpoints = {
            # Default endpoints for common exchanges
            "binance": {
                "rest": "https://api.binance.com",
                "websocket": "wss://stream.binance.com:9443/ws",
                "test": "https://testnet.binance.vision"
            },
            "coinbase": {
                "rest": "https://api.exchange.coinbase.com",
                "websocket": "wss://ws-feed.exchange.coinbase.com",
                "test": "https://api-public.sandbox.exchange.coinbase.com"
            },
            "alpaca": {
                "rest": "https://api.alpaca.markets",
                "websocket": "wss://stream.data.alpaca.markets/v2",
                "test": "https://paper-api.alpaca.markets"
            },
            "kraken": {
                "rest": "https://api.kraken.com",
                "websocket": "wss://ws.kraken.com",
                "test": "https://demo-futures.kraken.com"
            },
            "ftx": {
                "rest": "https://ftx.com/api",
                "websocket": "wss://ftx.com/ws",
                "test": "https://ftx.com/api"  # FTX doesn't have a separate test endpoint
            },
            "kucoin": {
                "rest": "https://api.kucoin.com",
                "websocket": "wss://push.kucoin.com",  # Requires token from REST API
                "test": "https://openapi-sandbox.kucoin.com"
            },
            "interactive_brokers": {
                "rest": "http://localhost:5000",  # IB Gateway/TWS with REST API bridge
                "websocket": "ws://localhost:5000/ws",
                "test": "http://localhost:5000"  # Use paper trading account in TWS
            }
        }
        
        return endpoints.get(self.exchange_id, {
            "rest": "",
            "websocket": "",
            "test": ""
        })
    
    def _get_default_rate_limits(self) -> Dict[str, Any]:
        """
        Get default rate limits for the exchange.
        
        Returns:
            Dictionary of rate limit settings
        """
        rate_limits = {
            # Default rate limits for common exchanges
            "binance": {
                "requests_per_minute": 1200,
                "orders_per_second": 10,
                "orders_per_day": 100000
            },
            "coinbase": {
                "requests_per_minute": 300,
                "orders_per_second": 5,
                "orders_per_day": 50000
            },
            "alpaca": {
                "requests_per_minute": 200,
                "orders_per_second": 5,
                "orders_per_day": 10000
            },
            "kraken": {
                "requests_per_minute": 60,
                "orders_per_second": 1,
                "orders_per_day": 5000
            },
            "ftx": {
                "requests_per_minute": 300,
                "orders_per_second": 5,
                "orders_per_day": 50000
            },
            "kucoin": {
                "requests_per_minute": 180,
                "orders_per_second": 3,
                "orders_per_day": 30000
            },
            "interactive_brokers": {
                "requests_per_minute": 100,
                "orders_per_second": 5,
                "orders_per_day": 10000
            }
        }
        
        return rate_limits.get(self.exchange_id, {
            "requests_per_minute": 100,
            "orders_per_second": 1,
            "orders_per_day": 1000
        })
    
    def _get_default_options(self) -> Dict[str, Any]:
        """
        Get default options for the exchange.
        
        Returns:
            Dictionary of exchange-specific options
        """
        options = {
            # Default options for common exchanges
            "binance": {
                "recv_window": 5000,
                "use_test_net": False,
                "default_type": "spot"  # spot, margin, futures
            },
            "coinbase": {
                "sandbox_mode": False
            },
            "alpaca": {
                "paper_trading": True
            },
            "kraken": {
                "validate_only": False
            },
            "ftx": {
                "subaccount": None
            },
            "kucoin": {
                "passphrase_as_header": True
            },
            "interactive_brokers": {
                "client_id": 0,
                "timeout": 60
            }
        }
        
        return options.get(self.exchange_id, {})
    
    def use_testnet(self, enabled: bool = True):
        """
        Configure to use test network instead of production.
        
        Args:
            enabled: Whether to enable test network
        """
        if self.exchange_id == "binance":
            self.options["use_test_net"] = enabled
        elif self.exchange_id == "coinbase":
            self.options["sandbox_mode"] = enabled
        elif self.exchange_id == "alpaca":
            self.options["paper_trading"] = enabled
        elif self.exchange_id == "kraken":
            self.options["validate_only"] = enabled
        elif self.exchange_id == "kucoin":
            self.options["use_sandbox"] = enabled
        
        logger.info(f"Set test network mode to {enabled} for {self.exchange_id}")
    
    def get_api_url(self, endpoint_type: str = "rest") -> str:
        """
        Get API URL for the specified endpoint type.
        
        Args:
            endpoint_type: Endpoint type (rest, websocket, test)
            
        Returns:
            API URL
        """
        if endpoint_type not in self.endpoints:
            raise ValueError(f"Unknown endpoint type: {endpoint_type}")
        
        # For some exchanges, use test endpoint if test mode is enabled
        if endpoint_type == "rest" and any([
            self.exchange_id == "binance" and self.options.get("use_test_net"),
            self.exchange_id == "coinbase" and self.options.get("sandbox_mode"),
            self.exchange_id == "alpaca" and self.options.get("paper_trading"),
            self.exchange_id == "kucoin" and self.options.get("use_sandbox")
        ]):
            return self.endpoints["test"]
        
        return self.endpoints[endpoint_type]


class ExchangeAPIError(Exception):
    """Exception raised for exchange API errors."""
    
    def __init__(self, message: str, code: Optional[int] = None, response: Optional[Any] = None):
        """
        Initialize exchange API error.
        
        Args:
            message: Error message
            code: Error code
            response: Raw API response
        """
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self.message)


class Order:
    """Represents an order in the exchange API."""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        exchange_id: Optional[str] = None,
        exchange_order_id: Optional[str] = None
    ):
        """
        Initialize an order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type (market, limit, etc.)
            quantity: Order quantity
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            client_order_id: Client-assigned order ID
            time_in_force: Time in force
            exchange_id: Exchange identifier
            exchange_order_id: Exchange-assigned order ID
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.client_order_id = client_order_id or f"order_{int(time.time() * 1000)}"
        self.time_in_force = time_in_force
        self.exchange_id = exchange_id
        self.exchange_order_id = exchange_order_id
        
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.remaining_quantity = quantity
        self.average_price = 0.0
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.fills = []
        
        # Validate order
        self._validate()
    
    def _validate(self):
        """Validate order parameters."""
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"Price is required for {self.order_type.value} orders")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and self.stop_price is None:
            raise ValueError(f"Stop price is required for {self.order_type.value} orders")
    
    def update_from_exchange(self, exchange_data: Dict[str, Any]):
        """
        Update order with data from exchange.
        
        Args:
            exchange_data: Order data from exchange
        """
        # Common fields across exchanges
        if "status" in exchange_data:
            status_str = exchange_data["status"].lower()
            for status in OrderStatus:
                if status.value == status_str:
                    self.status = status
                    break
        
        if "filled_quantity" in exchange_data:
            self.filled_quantity = float(exchange_data["filled_quantity"])
            self.remaining_quantity = self.quantity - self.filled_quantity
        
        if "average_price" in exchange_data:
            self.average_price = float(exchange_data["average_price"]) if exchange_data["average_price"] else 0.0
        
        if "exchange_order_id" in exchange_data:
            self.exchange_order_id = exchange_data["exchange_order_id"]
        
        if "updated_at" in exchange_data:
            self.updated_at = exchange_data["updated_at"]
        
        if "fills" in exchange_data:
            self.fills = exchange_data["fills"]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary.
        
        Returns:
            Order as dictionary
        """
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "client_order_id": self.client_order_id,
            "time_in_force": self.time_in_force.value,
            "exchange_id": self.exchange_id,
            "exchange_order_id": self.exchange_order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "fills": self.fills
        }


class ExchangeAPI:
    """Base class for exchange API implementations."""
    
    def __init__(self, config: ExchangeConfig):
        """
        Initialize exchange API.
        
        Args:
            config: Exchange configuration
        """
        self.config = config
        self.session = None
        self.ws_connections = {}
        self.order_updates_callbacks = []
        self.trade_updates_callbacks = []
        self.balance_updates_callbacks = []
        self.market_data_callbacks = {}
    
    async def initialize(self):
        """Initialize HTTP session and connections."""
        self.session = aiohttp.ClientSession()
        logger.info(f"Initialized {self.config.exchange_id} API")
    
    async def close(self):
        """Close all connections and resources."""
        if self.session:
            await self.session.close()
        
        for ws in self.ws_connections.values():
            if ws and not ws.closed:
                await ws.close()
        
        logger.info(f"Closed {self.config.exchange_id} API")
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Exchange information
        """
        raise NotImplementedError("Subclasses must implement get_exchange_info")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information
        """
        raise NotImplementedError("Subclasses must implement get_account_info")
    
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get account balances.
        
        Returns:
            Dictionary of asset -> balance information
        """
        raise NotImplementedError("Subclasses must implement get_balances")
    
    async def get_order(self, order_id: str, client_order_id: Optional[str] = None) -> Order:
        """
        Get order by ID.
        
        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID
            
        Returns:
            Order object
        """
        raise NotImplementedError("Subclasses must implement get_order")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        raise NotImplementedError("Subclasses must implement get_open_orders")
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Order]:
        """
        Get order history.
        
        Args:
            symbol: Trading symbol (optional)
            limit: Maximum number of orders to return
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of historical orders
        """
        raise NotImplementedError("Subclasses must implement get_order_history")
    
    async def create_order(self, order: Order) -> Order:
        """
        Create a new order.
        
        Args:
            order: Order to create
            
        Returns:
            Created order with exchange information
        """
        raise NotImplementedError("Subclasses must implement create_order")
    
    async def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None
    ) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID
            
        Returns:
            True if order was canceled, False otherwise
        """
        raise NotImplementedError("Subclasses must implement cancel_order")
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            Number of orders canceled
        """
        raise NotImplementedError("Subclasses must implement cancel_all_orders")
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker information
        """
        raise NotImplementedError("Subclasses must implement get_ticker")
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get orderbook for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of bids/asks to return
            
        Returns:
            Orderbook information
        """
        raise NotImplementedError("Subclasses must implement get_orderbook")
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades
        """
        raise NotImplementedError("Subclasses must implement get_recent_trades")
    
    async def subscribe_to_order_updates(self, callback: Callable[[Order], None]):
        """
        Subscribe to order updates.
        
        Args:
            callback: Callback function for order updates
        """
        self.order_updates_callbacks.append(callback)
    
    async def subscribe_to_trade_updates(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to trade updates.
        
        Args:
            callback: Callback function for trade updates
        """
        self.trade_updates_callbacks.append(callback)
    
    async def subscribe_to_balance_updates(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to balance updates.
        
        Args:
            callback: Callback function for balance updates
        """
        self.balance_updates_callbacks.append(callback)
    
    async def subscribe_to_market_data(
        self,
        symbol: str,
        channels: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to market data.
        
        Args:
            symbol: Trading symbol
            channels: List of channels to subscribe to (e.g., trades, orderbook, ticker)
            callback: Callback function for market data
        """
        key = f"{symbol}:{','.join(channels)}"
        if key not in self.market_data_callbacks:
            self.market_data_callbacks[key] = []
        self.market_data_callbacks[key].append(callback)
    
    def _handle_order_update(self, order_data: Dict[str, Any]):
        """
        Handle order update from exchange.
        
        Args:
            order_data: Order data from exchange
        """
        # Convert exchange-specific order data to Order object
        order = self._parse_order_from_exchange(order_data)
        
        # Notify callbacks
        for callback in self.order_updates_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")
    
    def _handle_trade_update(self, trade_data: Dict[str, Any]):
        """
        Handle trade update from exchange.
        
        Args:
            trade_data: Trade data from exchange
        """
        # Notify callbacks
        for callback in self.trade_updates_callbacks:
            try:
                callback(trade_data)
            except Exception as e:
                logger.error(f"Error in trade update callback: {e}")
    
    def _handle_balance_update(self, balance_data: Dict[str, Any]):
        """
        Handle balance update from exchange.
        
        Args:
            balance_data: Balance data from exchange
        """
        # Notify callbacks
        for callback in self.balance_updates_callbacks:
            try:
                callback(balance_data)
            except Exception as e:
                logger.error(f"Error in balance update callback: {e}")
    
    def _handle_market_data(self, symbol: str, channel: str, data: Dict[str, Any]):
        """
        Handle market data from exchange.
        
        Args:
            symbol: Trading symbol
            channel: Data channel
            data: Market data
        """
        # Find matching callbacks
        for key, callbacks in self.market_data_callbacks.items():
            key_symbol, key_channels = key.split(":", 1)
            key_channels = key_channels.split(",")
            
            if key_symbol == symbol and channel in key_channels:
                for callback in callbacks:
                    try:
                        callback({
                            "symbol": symbol,
                            "channel": channel,
                            "data": data
                        })
                    except Exception as e:
                        logger.error(f"Error in market data callback: {e}")
    
    def _parse_order_from_exchange(self, order_data: Dict[str, Any]) -> Order:
        """
        Parse order data from exchange.
        
        Args:
            order_data: Order data from exchange
            
        Returns:
            Order object
        """
        raise NotImplementedError("Subclasses must implement _parse_order_from_exchange")


class BinanceAPI(ExchangeAPI):
    """Binance exchange API implementation."""
    
    def __init__(self, config: ExchangeConfig):
        """
        Initialize Binance API.
        
        Args:
            config: Exchange configuration
        """
        super().__init__(config)
        self.ws_listen_key = None
        self.ws_listen_key_timer = None
    
    async def initialize(self):
        """Initialize HTTP session and connections."""
        await super().initialize()
        
        # Get listen key for user data stream
        if self.config.credentials:
            await self._get_listen_key()
            await self._start_user_data_stream()
    
    async def close(self):
        """Close all connections and resources."""
        # Close user data stream
        if self.ws_listen_key:
            await self._close_listen_key()
        
        await super().close()
    
    async def _get_listen_key(self):
        """Get listen key for user data stream."""
        if not self.config.credentials:
            return
        
        endpoint = "/api/v3/userDataStream"
        headers = {"X-MBX-APIKEY": self.config.credentials.api_key}
        
        try:
            async with self.session.post(
                f"{self.config.get_api_url()}{endpoint}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.ws_listen_key = data["listenKey"]
                    logger.info("Obtained Binance listen key")
                    
                    # Start timer to keep listen key alive
                    if self.ws_listen_key_timer:
                        self.ws_listen_key_timer.cancel()
                    
                    self.ws_listen_key_timer = asyncio.create_task(self._keep_listen_key_alive())
                else:
                    logger.error(f"Failed to get listen key: {response.status}")
        except Exception as e:
            logger.error(f"Error getting listen key: {e}")
    
    async def _keep_listen_key_alive(self):
        """Keep listen key alive by sending periodic requests."""
        while True:
            try:
                await asyncio.sleep(30 * 60)  # Ping every 30 minutes
                await self._ping_listen_key()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error keeping listen key alive: {e}")
    
    async def _ping_listen_key(self):
        """Ping listen key to keep it alive."""
        if not self.ws_listen_key:
            return
        
        endpoint = "/api/v3/userDataStream"
        headers = {"X-MBX-APIKEY": self.config.credentials.api_key}
        params = {"listenKey": self.ws_listen_key}
        
        try:
            async with self.session.put(
                f"{self.config.get_api_url()}{endpoint}",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    logger.debug("Pinged Binance listen key")
                else:
                    logger.error(f"Failed to ping listen key: {response.status}")
                    # Try to get a new listen key
                    await self._get_listen_key()
        except Exception as e:
            logger.error(f"Error pinging listen key: {e}")
    
    async def _close_listen_key(self):
        """Close listen key."""
        if not self.ws_listen_key:
            return
        
        endpoint = "/api/v3/userDataStream"
        headers = {"X-MBX-APIKEY": self.config.credentials.api_key}
        params = {"listenKey": self.ws_listen_key}
        
        try:
            async with self.session.delete(
                f"{self.config.get_api_url()}{endpoint}",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    logger.info("Closed Binance listen key")
                else:
                    logger.error(f"Failed to close listen key: {response.status}")
        except Exception as e:
            logger.error(f"Error closing listen key: {e}")
        
        # Cancel timer
        if self.ws_listen_key_timer:
            self.ws_listen_key_timer.cancel()
            self.ws_listen_key_timer = None
        
        self.ws_listen_key = None
    
    async def _start_user_data_stream(self):
        """Start user data stream for real-time updates."""
        if not self.ws_listen_key:
            return
        
        ws_url = f"{self.config.get_api_url('websocket')}/{self.ws_listen_key}"
        
        try:
            self.ws_connections["user_data"] = await websockets.connect(ws_url)
            logger.info("Connected to Binance user data stream")
            
            # Start listening for messages
            asyncio.create_task(self._listen_user_data_stream())
        except Exception as e:
            logger.error(f"Error connecting to user data stream: {e}")
    
    async def _listen_user_data_stream(self):
        """Listen for messages from user data stream."""
        ws = self.ws_connections.get("user_data")
        if not ws:
            return
        
        try:
            while not ws.closed:
                message = await ws.recv()
                data = json.loads(message)
                
                event_type = data.get("e")
                if event_type == "executionReport":
                    # Order update
                    self._handle_order_update(data)
                elif event_type == "outboundAccountPosition":
                    # Balance update
                    self._handle_balance_update(data)
                elif event_type == "balanceUpdate":
                    # Balance update from deposit or withdrawal
                    self._handle_balance_update(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Binance user data stream connection closed")
        except Exception as e:
            logger.error(f"Error in user data stream: {e}")
        
        # Try to reconnect
        await asyncio.sleep(5)
        await self._start_user_data_stream()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = False
    ) -> Dict[str, Any]:
        """
        Send request to Binance API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            auth: Whether authentication is required
            
        Returns:
            API response
        """
        url = f"{self.config.get_api_url()}{endpoint}"
        headers = {}
        
        if auth:
            if not self.config.credentials:
                raise ValueError("Authentication required but no credentials provided")
            
            headers["X-MBX-APIKEY"] = self.config.credentials.api_key
            
            # Add timestamp and signature
            params = params or {}
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.config.options.get("recv_window", 5000)
            
            # Create signature
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.credentials.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            params["signature"] = signature
        
        try:
            if method == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    response_data = await response.json()
                    if response.status != 200:
                        raise ExchangeAPIError(
                            f"Binance API error: {response_data.get('msg', 'Unknown error')}",
                            response.status,
                            response_data
                        )
                    return response_data
            elif method == "POST":
                async with self.session.post(url, params=params, json=data, headers=headers) as response:
                    response_data = await response.json()
                    if response.status != 200:
                        raise ExchangeAPIError(
                            f"Binance API error: {response_data.get('msg', 'Unknown error')}",
                            response.status,
                            response_data
                        )
                    return response_data
            elif method == "DELETE":
                async with self.session.delete(url, params=params, headers=headers) as response:
                    response_data = await response.json()
                    if response.status != 200:
                        raise ExchangeAPIError(
                            f"Binance API error: {response_data.get('msg', 'Unknown error')}",
                            response.status,
                            response_data
                        )
                    return response_data
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except aiohttp.ClientError as e:
            raise ExchangeAPIError(f"Binance API request failed: {e}")
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Exchange information
        """
        return await self._request("GET", "/api/v3/exchangeInfo")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Account information
        """
        return await self._request("GET", "/api/v3/account", auth=True)
    
    async def get_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Get account balances.
        
        Returns:
            Dictionary of asset -> balance information
        """
        account_info = await self.get_account_info()
        balances = {}
        
        for balance in account_info["balances"]:
            asset = balance["asset"]
            free = float(balance["free"])
            locked = float(balance["locked"])
            
            if free > 0 or locked > 0:
                balances[asset] = {
                    "free": free,
                    "locked": locked,
                    "total": free + locked
                }
        
        return balances
    
    async def get_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Order:
        """
        Get order by ID.
        
        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID
            symbol: Trading symbol (required for Binance)
            
        Returns:
            Order object
        """
        if not symbol:
            raise ValueError("Symbol is required for Binance get_order")
        
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id is required")
        
        order_data = await self._request("GET", "/api/v3/order", params=params, auth=True)
        return self._parse_order_from_exchange(order_data)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        orders_data = await self._request("GET", "/api/v3/openOrders", params=params, auth=True)
        return [self._parse_order_from_exchange(order_data) for order_data in orders_data]
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Order]:
        """
        Get order history.
        
        Args:
            symbol: Trading symbol (required for Binance)
            limit: Maximum number of orders to return
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of historical orders
        """
        if not symbol:
            raise ValueError("Symbol is required for Binance get_order_history")
        
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        
        if end_time:
            params["endTime"] = end_time
        
        orders_data = await self._request("GET", "/api/v3/allOrders", params=params, auth=True)
        return [self._parse_order_from_exchange(order_data) for order_data in orders_data]
    
    async def create_order(self, order: Order) -> Order:
        """
        Create a new order.
        
        Args:
            order: Order to create
            
        Returns:
            Created order with exchange information
        """
        params = {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "type": order.order_type.value.upper(),
            "quantity": order.quantity,
            "newClientOrderId": order.client_order_id,
            "timeInForce": order.time_in_force.value.upper()
        }
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            params["price"] = order.price
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            params["stopPrice"] = order.stop_price
        
        order_data = await self._request("POST", "/api/v3/order", params=params, auth=True)
        
        # Update order with exchange data
        order.exchange_id = self.config.exchange_id
        order.exchange_order_id = str(order_data["orderId"])
        order.status = OrderStatus.OPEN
        
        return order
    
    async def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID
            symbol: Trading symbol (required for Binance)
            
        Returns:
            True if order was canceled, False otherwise
        """
        if not symbol:
            raise ValueError("Symbol is required for Binance cancel_order")
        
        params = {"symbol": symbol}
        
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id is required")
        
        try:
            await self._request("DELETE", "/api/v3/order", params=params, auth=True)
            return True
        except ExchangeAPIError as e:
            logger.error(f"Error canceling order: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol: Trading symbol (required for Binance)
            
        Returns:
            Number of orders canceled
        """
        if not symbol:
            raise ValueError("Symbol is required for Binance cancel_all_orders")
        
        params = {"symbol": symbol}
        
        try:
            result = await self._request("DELETE", "/api/v3/openOrders", params=params, auth=True)
            return len(result)
        except ExchangeAPIError as e:
            logger.error(f"Error canceling all orders: {e}")
            return 0
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker information
        """
        params = {"symbol": symbol}
        return await self._request("GET", "/api/v3/ticker/24hr", params=params)
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get orderbook for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of bids/asks to return
            
        Returns:
            Orderbook information
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        data = await self._request("GET", "/api/v3/depth", params=params)
        
        # Format response
        return {
            "symbol": symbol,
            "bids": [[float(price), float(qty)] for price, qty in data["bids"]],
            "asks": [[float(price), float(qty)] for price, qty in data["asks"]],
            "timestamp": int(time.time() * 1000)
        }
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades
        """
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        trades = await self._request("GET", "/api/v3/trades", params=params)
        
        # Format response
        return [{
            "id": trade["id"],
            "price": float(trade["price"]),
            "quantity": float(trade["qty"]),
            "time": trade["time"],
            "is_buyer_maker": trade["isBuyerMaker"],
            "is_best_match": trade["isBestMatch"]
        } for trade in trades]
    
    async def subscribe_to_market_data(
        self,
        symbol: str,
        channels: List[str],
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to market data.
        
        Args:
            symbol: Trading symbol
            channels: List of channels to subscribe to (e.g., trades, depth, ticker)
            callback: Callback function for market data
        """
        await super().subscribe_to_market_data(symbol, channels, callback)
        
        # Format symbol for Binance (lowercase)
        formatted_symbol = symbol.lower()
        
        # Map channels to Binance stream names
        stream_names = []
        for channel in channels:
            if channel == "trades":
                stream_names.append(f"{formatted_symbol}@trade")
            elif channel == "orderbook":
                stream_names.append(f"{formatted_symbol}@depth")
            elif channel == "ticker":
                stream_names.append(f"{formatted_symbol}@ticker")
            elif channel == "kline":
                stream_names.append(f"{formatted_symbol}@kline_1m")
        
        if not stream_names:
            return
        
        # Create WebSocket connection
        streams_param = "/".join(stream_names)
        ws_url = f"{self.config.get_api_url('websocket')}/stream?streams={streams_param}"
        
        try:
            ws_key = f"market_data_{symbol}_{','.join(channels)}"
            self.ws_connections[ws_key] = await websockets.connect(ws_url)
            logger.info(f"Connected to Binance market data stream for {symbol}")
            
            # Start listening for messages
            asyncio.create_task(self._listen_market_data_stream(ws_key, symbol, channels))
        except Exception as e:
            logger.error(f"Error connecting to market data stream: {e}")
    
    async def _listen_market_data_stream(self, ws_key: str, symbol: str, channels: List[str]):
        """
        Listen for messages from market data stream.
        
        Args:
            ws_key: WebSocket connection key
            symbol: Trading symbol
            channels: Subscribed channels
        """
        ws = self.ws_connections.get(ws_key)
        if not ws:
            return
        
        try:
            while not ws.closed:
                message = await ws.recv()
                data = json.loads(message)
                
                # Extract stream name and data
                stream = data.get("stream", "")
                stream_data = data.get("data", {})
                
                # Determine channel from stream name
                channel = None
                if "trade" in stream:
                    channel = "trades"
                    self._handle_market_data(symbol, channel, stream_data)
                elif "depth" in stream:
                    channel = "orderbook"
                    self._handle_market_data(symbol, channel, stream_data)
                elif "ticker" in stream:
                    channel = "ticker"
                    self._handle_market_data(symbol, channel, stream_data)
                elif "kline" in stream:
                    channel = "kline"
                    self._handle_market_data(symbol, channel, stream_data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Binance market data stream connection closed for {symbol}")
        except Exception as e:
            logger.error(f"Error in market data stream for {symbol}: {e}")
        
        # Try to reconnect
        await asyncio.sleep(5)
        await self.subscribe_to_market_data(symbol, channels, None)  # Reuse existing callbacks
    
    def _parse_order_from_exchange(self, order_data: Dict[str, Any]) -> Order:
        """
        Parse order data from Binance.
        
        Args:
            order_data: Order data from Binance
            
        Returns:
            Order object
        """
        # Map Binance order type to our OrderType
        order_type_map = {
            "LIMIT": OrderType.LIMIT,
            "MARKET": OrderType.MARKET,
            "STOP_LOSS": OrderType.STOP,
            "STOP_LOSS_LIMIT": OrderType.STOP_LIMIT,
            "TAKE_PROFIT": OrderType.STOP,
            "TAKE_PROFIT_LIMIT": OrderType.STOP_LIMIT
        }
        
        # Map Binance order side to our OrderSide
        order_side_map = {
            "BUY": OrderSide.BUY,
            "SELL": OrderSide.SELL
        }
        
        # Map Binance order status to our OrderStatus
        order_status_map = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        
        # Map Binance time in force to our TimeInForce
        time_in_force_map = {
            "GTC": TimeInForce.GTC,
            "IOC": TimeInForce.IOC,
            "FOK": TimeInForce.FOK
        }
        
        # Create Order object
        order = Order(
            symbol=order_data["symbol"],
            side=order_side_map.get(order_data["side"], OrderSide.BUY),
            order_type=order_type_map.get(order_data["type"], OrderType.MARKET),
            quantity=float(order_data["origQty"]),
            price=float(order_data["price"]) if order_data.get("price") else None,
            stop_price=float(order_data["stopPrice"]) if order_data.get("stopPrice") else None,
            client_order_id=order_data.get("clientOrderId"),
            time_in_force=time_in_force_map.get(order_data.get("timeInForce"), TimeInForce.GTC),
            exchange_id=self.config.exchange_id,
            exchange_order_id=str(order_data["orderId"])
        )
        
        # Update order status
        order.status = order_status_map.get(order_data["status"], OrderStatus.PENDING)
        order.filled_quantity = float(order_data["executedQty"])
        order.remaining_quantity = order.quantity - order.filled_quantity
        
        # Calculate average price
        if float(order_data["executedQty"]) > 0 and float(order_data.get("cummulativeQuoteQty", 0)) > 0:
            order.average_price = float(order_data["cummulativeQuoteQty"]) / float(order_data["executedQty"])
        
        # Set timestamps
        if "time" in order_data:
            order.created_at = datetime.fromtimestamp(order_data["time"] / 1000)
        
        if "updateTime" in order_data:
            order.updated_at = datetime.fromtimestamp(order_data["updateTime"] / 1000)
        
        return order


class ExchangeManager:
    """Manager for multiple exchange API connections."""
    
    def __init__(self):
        """Initialize exchange manager."""
        self.exchanges = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize all exchange connections."""
        for exchange in self.exchanges.values():
            await exchange.initialize()
        
        self.initialized = True
        logger.info("Exchange manager initialized")
    
    async def close(self):
        """Close all exchange connections."""
        for exchange in self.exchanges.values():
            await exchange.close()
        
        self.initialized = False
        logger.info("Exchange manager closed")
    
    def add_exchange(self, config: ExchangeConfig) -> ExchangeAPI:
        """
        Add an exchange to the manager.
        
        Args:
            config: Exchange configuration
            
        Returns:
            Exchange API instance
        """
        exchange_id = config.exchange_id.lower()
        
        # Create appropriate exchange API instance
        if exchange_id == "binance":
            exchange = BinanceAPI(config)
        # Add more exchange implementations here
        else:
            raise ValueError(f"Unsupported exchange: {exchange_id}")
        
        self.exchanges[exchange_id] = exchange
        logger.info(f"Added {exchange_id} exchange to manager")
        
        return exchange
    
    def get_exchange(self, exchange_id: str) -> ExchangeAPI:
        """
        Get exchange API instance.
        
        Args:
            exchange_id: Exchange identifier
            
        Returns:
            Exchange API instance
        """
        exchange_id = exchange_id.lower()
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange not found: {exchange_id}")
        
        return self.exchanges[exchange_id]
    
    def remove_exchange(self, exchange_id: str):
        """
        Remove an exchange from the manager.
        
        Args:
            exchange_id: Exchange identifier
        """
        exchange_id = exchange_id.lower()
        if exchange_id in self.exchanges:
            self.exchanges.pop(exchange_id)
            logger.info(f"Removed {exchange_id} exchange from manager")


# Example usage
async def example_usage():
    """Example of how to use the exchange API module."""
    # Create exchange manager
    manager = ExchangeManager()
    
    # Add Binance exchange
    config = ExchangeConfig("binance")
    config.use_testnet(True)  # Use testnet for testing
    
    # Add credentials if available
    if os.environ.get("BINANCE_API_KEY") and os.environ.get("BINANCE_API_SECRET"):
        credentials = ExchangeCredentials(
            api_key=os.environ["BINANCE_API_KEY"],
            api_secret=os.environ["BINANCE_API_SECRET"]
        )
        config.credentials = credentials
    
    # Add exchange to manager
    exchange = manager.add_exchange(config)
    
    # Initialize manager
    await manager.initialize()
    
    try:
        # Get exchange info
        exchange_info = await exchange.get_exchange_info()
        print(f"Exchange info: {exchange_info['timezone']}")
        
        # Get ticker
        ticker = await exchange.get_ticker("BTCUSDT")
        print(f"BTC/USDT ticker: {ticker['lastPrice']}")
        
        # Get orderbook
        orderbook = await exchange.get_orderbook("BTCUSDT", limit=5)
        print(f"BTC/USDT orderbook: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks")
        
        # If credentials are available, test authenticated endpoints
        if config.credentials:
            # Get account info
            account_info = await exchange.get_account_info()
            print(f"Account status: {account_info['accountType']}")
            
            # Get balances
            balances = await exchange.get_balances()
            print(f"Balances: {len(balances)} assets")
            
            # Create a test order
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=20000.0,
                time_in_force=TimeInForce.GTC
            )
            
            try:
                created_order = await exchange.create_order(order)
                print(f"Created order: {created_order.exchange_order_id}")
                
                # Cancel the order
                canceled = await exchange.cancel_order(
                    order_id=created_order.exchange_order_id,
                    symbol="BTCUSDT"
                )
                print(f"Order canceled: {canceled}")
            except ExchangeAPIError as e:
                print(f"Error creating/canceling order: {e}")
    
    finally:
        # Close manager
        await manager.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
