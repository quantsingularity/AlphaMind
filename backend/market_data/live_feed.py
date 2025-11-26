"""
## Live Market Data Feed Module for AlphaMind

## This module provides real-time market data integration with various exchanges
## and data providers. It handles connection management, data normalization,
## and streaming capabilities with robust retry logic, validation, and logging.
"""

import asyncio
from datetime import datetime
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from confluent_kafka import Producer
import websockets

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

# Add file handler for persistent logging
try:
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/market_data_feed.log")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
    )
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not set up file logging: {e}")


class DataValidationError(Exception):
    """Exception raised for data validation errors."""


class ConnectionError(Exception):
    """Exception raised for connection errors."""


class MarketDataConfig:
    """Configuration for market data connections and processing."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize market data configuration.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.api_keys: Dict[str, str] = {}
        self.endpoints: Dict[str, str] = {}
        self.streaming_endpoints: Dict[str, str] = {}
        self.retry_settings = {
            "max_retries": 5,
            "backoff_factor": 1.5,
            "initial_wait": 1.0,
            "max_wait": 60.0,  # Maximum wait time between retries
        }
        self.validation_settings = {
            "validate_responses": True,
            "required_fields": {
                "trades": ["symbol", "price", "quantity", "timestamp"],
                "quotes": ["symbol", "bid_price", "ask_price", "timestamp"],
                "bars": [
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "timestamp",
                ],
            },
            "type_validation": {
                "price": float,
                "quantity": float,
                "timestamp": (int, float),
                "volume": float,
            },
        }

        if config_path:
            self.load_from_file(config_path)
        else:
            self._set_defaults()

        logger.info("Market data configuration initialized")

    def _set_defaults(self):
        """Set default configuration values."""
        # Default REST API endpoints
        self.endpoints = {
            "binance": "https://api.binance.com/api/v3",
            "coinbase": "https://api.exchange.coinbase.com",
            "alpaca": "https://paper-api.alpaca.markets/v2",
            "iex": "https://cloud.iexapis.com/stable",
            "polygon": "https://api.polygon.io/v2",
        }

        # Default WebSocket endpoints
        self.streaming_endpoints = {
            "binance": "wss://stream.binance.com:9443/ws",
            "coinbase": "wss://ws-feed.exchange.coinbase.com",
            "alpaca": "wss://paper-api.alpaca.markets/stream",
            "polygon": "wss://socket.polygon.io/stocks",
        }

        logger.debug("Default configuration values set")

    def load_from_file(self, config_path: str):
        """
        Load configuration from a JSON file with validation.

        Args:
            config_path: Path to configuration file
        """
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            self._set_defaults()
            return

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Validate configuration structure
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a JSON object")

            self.api_keys = config.get("api_keys", {})
            self.endpoints = config.get("endpoints", self.endpoints)
            self.streaming_endpoints = config.get(
                "streaming_endpoints", self.streaming_endpoints
            )

            # Validate and merge retry settings
            if "retry_settings" in config:
                retry_config = config["retry_settings"]
                if not isinstance(retry_config, dict):
                    logger.warning("Invalid retry_settings format, using defaults")
                else:
                    for key, value in retry_config.items():
                        if key in self.retry_settings:
                            self.retry_settings[key] = value

            # Validate and merge validation settings
            if "validation_settings" in config:
                validation_config = config["validation_settings"]
                if not isinstance(validation_config, dict):
                    logger.warning("Invalid validation_settings format, using defaults")
                else:
                    for key, value in validation_config.items():
                        if key in self.validation_settings:
                            self.validation_settings[key] = value

            logger.info(f"Loaded market data configuration from {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
            self._set_defaults()
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            self._set_defaults()

    def save_to_file(self, config_path: str):
        """
        Save configuration to a JSON file.

        Args:
            config_path: Path to save configuration file
        """
        config = {
            "api_keys": self.api_keys,
            "endpoints": self.endpoints,
            "streaming_endpoints": self.streaming_endpoints,
            "retry_settings": self.retry_settings,
            "validation_settings": self.validation_settings,
        }

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved market data configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            return False


class MarketDataFeed:
    """Base class for market data feeds."""

    def __init__(self, config: MarketDataConfig):
        """
        Initialize market data feed.

        Args:
            config: Market data configuration
        """
        self.config = config
        self.session = None
        self.ws_connections = {}
        self.callbacks = {}
        self.running = False
        self.kafka_producer = None
        self.health_status = {
            "last_successful_fetch": None,
            "connection_errors": 0,
            "data_errors": 0,
            "reconnect_attempts": 0,
        }
        self.data_stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "validation_errors": 0,
        }

        logger.info("Market data feed instance created")

    async def initialize(self):
        """Initialize HTTP session and connections with error handling."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),  # 30 second timeout
                headers={"User-Agent": "AlphaMind/1.0"},
            )
            logger.info("Market data feed HTTP session initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize market data feed: {e}")
            return False

    async def close(self):
        """Close all connections and resources with error handling."""
        close_errors = []

        # Close HTTP session
        if self.session:
            try:
                await self.session.close()
                logger.info("HTTP session closed")
            except Exception as e:
                error_msg = f"Error closing HTTP session: {e}"
                logger.error(error_msg)
                close_errors.append(error_msg)

        # Close WebSocket connections
        for provider, ws in self.ws_connections.items():
            if ws and not ws.closed:
                try:
                    await ws.close()
                    logger.info(f"WebSocket connection to {provider} closed")
                except Exception as e:
                    error_msg = f"Error closing WebSocket connection to {provider}: {e}"
                    logger.error(error_msg)
                    close_errors.append(error_msg)

        self.running = False

        if close_errors:
            logger.warning(f"Market data feed closed with {len(close_errors)} errors")
            return False
        else:
            logger.info("Market data feed closed successfully")
            return True

    async def fetch_data(
        self, provider: str, endpoint: str, params: Optional[Dict] = None
    ) -> Dict:
        """
        Fetch data from REST API with enhanced retry logic and validation.

        Args:
            provider: Data provider name
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            API response as dictionary

        Raises:
            ConnectionError: If connection to provider fails after retries
            DataValidationError: If response validation fails
            ValueError: If provider is unknown
        """
        if not self.session:
            initialized = await self.initialize()
            if not initialized:
                raise ConnectionError(f"Failed to initialize session for {provider}")

        base_url = self.config.endpoints.get(provider)
        if not base_url:
            raise ValueError(f"Unknown provider: {provider}")

        url = f"{base_url}/{endpoint.lstrip('/')}"
        headers = {}

        # Add API key if available
        if provider in self.config.api_keys:
            if provider == "alpaca":
                headers["APCA-API-KEY-ID"] = self.config.api_keys[provider]["key"]
                headers["APCA-API-SECRET-KEY"] = self.config.api_keys[provider][
                    "secret"
                ]
            elif provider == "iex" or provider == "polygon":
                if params is None:
                    params = {}
                params["apiKey"] = self.config.api_keys[provider]
            else:
                headers["X-API-Key"] = self.config.api_keys[provider]

        # Implement enhanced retry logic
        retries = 0
        max_retries = self.config.retry_settings["max_retries"]
        backoff = self.config.retry_settings["initial_wait"]
        max_wait = self.config.retry_settings["max_wait"]

        while retries <= max_retries:
            try:
                start_time = time.time()
                logger.debug(
                    f"Fetching data from {provider}/{endpoint} (attempt {retries+1}/{max_retries+1})"
                )

                async with self.session.get(
                    url, params=params, headers=headers
                ) as response:
                    elapsed = time.time() - start_time

                    if response.status == 200:
                        data = await response.json()

                        # Validate response data
                        if self.config.validation_settings["validate_responses"]:
                            try:
                                self._validate_response_data(data, provider, endpoint)
                            except DataValidationError as e:
                                logger.warning(
                                    f"Data validation error for {provider}/{endpoint}: {e}"
                                )
                                self.data_stats["validation_errors"] += 1
                                # Continue despite validation error, but log it

                        logger.info(
                            f"Successfully fetched data from {provider}/{endpoint} in {elapsed:.2f}s"
                        )
                        self.health_status["last_successful_fetch"] = datetime.now()
                        return data

                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get("Retry-After", backoff))
                        logger.warning(
                            f"Rate limited by {provider}, retrying after {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                        retries += 1

                    elif response.status >= 500:  # Server error
                        logger.warning(
                            f"Server error from {provider}: {response.status}, retrying..."
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(
                            backoff * self.config.retry_settings["backoff_factor"],
                            max_wait,
                        )
                        retries += 1

                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Error response from {provider}: {response.status}, {error_text}"
                        )
                        response.raise_for_status()

            except aiohttp.ClientError as e:
                logger.error(f"Connection error fetching data from {provider}: {e}")
                self.health_status["connection_errors"] += 1

                if retries >= max_retries:
                    raise ConnectionError(
                        f"Failed to connect to {provider} after {max_retries} retries: {e}"
                    )

                await asyncio.sleep(backoff)
                backoff = min(
                    backoff * self.config.retry_settings["backoff_factor"], max_wait
                )
                retries += 1

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from {provider}: {e}")
                self.health_status["data_errors"] += 1

                if retries >= max_retries:
                    raise DataValidationError(
                        f"Invalid JSON from {provider} after {max_retries} retries: {e}"
                    )

                await asyncio.sleep(backoff)
                backoff = min(
                    backoff * self.config.retry_settings["backoff_factor"], max_wait
                )
                retries += 1

        raise ConnectionError(
            f"Failed to fetch data from {provider}/{endpoint} after {max_retries} retries"
        )

    def _validate_response_data(self, data: Any, provider: str, endpoint: str) -> bool:
        """
        Validate response data.

        Args:
            data: Response data
            provider: Data provider name
            endpoint: API endpoint path

        Returns:
            True if validation passes

        Raises:
            DataValidationError: If validation fails
        """
        # Basic validation - check if data is not None
        if data is None:
            raise DataValidationError("Response data is None")

        # Provider-specific validation
        if provider == "binance":
            if endpoint.endswith("/ticker/24hr"):
                if isinstance(data, list):
                    for item in data:
                        if not all(
                            k in item for k in ["symbol", "lastPrice", "volume"]
                        ):
                            raise DataValidationError(
                                "Missing required fields in Binance ticker data"
                            )
                else:
                    if not all(k in data for k in ["symbol", "lastPrice", "volume"]):
                        raise DataValidationError(
                            "Missing required fields in Binance ticker data"
                        )

        elif provider == "alpaca":
            if endpoint.endswith("/bars"):
                if not isinstance(data, dict) or "bars" not in data:
                    raise DataValidationError("Invalid Alpaca bars response format")

        # Type validation for common fields
        if isinstance(data, dict):
            for field, expected_type in self.config.validation_settings[
                "type_validation"
            ].items():
                if field in data:
                    if not isinstance(data[field], expected_type):
                        try:
                            # Attempt conversion for numeric types
                            if expected_type in (int, float) and isinstance(
                                data[field], (int, float, str)
                            ):
                                data[field] = expected_type(data[field])
                            else:
                                raise DataValidationError(
                                    f"Field {field} has invalid type: {type(data[field])}, expected {expected_type}"
                                )
                        except (ValueError, TypeError):
                            raise DataValidationError(
                                f"Field {field} has invalid type and cannot be converted"
                            )

        return True

    async def subscribe_to_stream(
        self,
        provider: str,
        symbols: List[str],
        callback: Callable[[Dict], None],
        channel: str = "trades",
    ):
        """
        Subscribe to real-time data stream with enhanced error handling.

        Args:
            provider: Data provider name
            symbols: List of symbols to subscribe to
            callback: Callback function for data processing
            channel: Data channel (trades, quotes, etc.)

        Returns:
            True if subscription successful, False otherwise
        """
        if provider not in self.config.streaming_endpoints:
            logger.error(f"Streaming not supported for provider: {provider}")
            return False

        # Validate symbols
        if not symbols or not all(isinstance(s, str) for s in symbols):
            logger.error(f"Invalid symbols list for {provider}: {symbols}")
            return False

        # Store callback
        callback_key = f"{provider}_{channel}_{','.join(symbols)}"
        self.callbacks[callback_key] = callback

        # Connect to WebSocket if not already connected
        if provider not in self.ws_connections or self.ws_connections[provider].closed:
            try:
                await self._connect_websocket(provider, symbols, channel)
                logger.info(
                    f"Successfully subscribed to {provider} {channel} for {len(symbols)} symbols"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to subscribe to {provider} stream: {e}")
                return False
        else:
            logger.info(
                f"Using existing connection to subscribe to {provider} {channel}"
            )
            return True

    async def _connect_websocket(self, provider: str, symbols: List[str], channel: str):
        """
        Establish WebSocket connection to data provider with retry logic.

        Args:
            provider: Data provider name
            symbols: List of symbols to subscribe to
            channel: Data channel (trades, quotes, etc.)

        Raises:
            ConnectionError: If connection fails after retries
        """
        ws_url = self.config.streaming_endpoints[provider]

        # Format subscription message based on provider
        if provider == "binance":
            # For Binance, we need to connect to individual streams
            streams = [f"{symbol.lower()}@{channel}" for symbol in symbols]
            ws_url = f"{ws_url}/stream?streams={'/'.join(streams)}"
            subscription_msg = None
        elif provider == "coinbase":
            subscription_msg = {
                "type": "subscribe",
                "product_ids": symbols,
                "channels": [channel],
            }
        elif provider == "alpaca":
            subscription_msg = {
                "action": "auth",
                "key": self.config.api_keys[provider]["key"],
                "secret": self.config.api_keys[provider]["secret"],
            }
            # After authentication, we'll send the subscription message
        elif provider == "polygon":
            subscription_msg = {
                "action": "auth",
                "params": self.config.api_keys[provider],
            }
            # After authentication, we'll send the subscription message
        else:
            raise ValueError(f"Unsupported streaming provider: {provider}")

        # Implement retry logic for WebSocket connection
        retries = 0
        max_retries = self.config.retry_settings["max_retries"]
        backoff = self.config.retry_settings["initial_wait"]
        max_wait = self.config.retry_settings["max_wait"]

        while retries <= max_retries:
            try:
                # Connect to WebSocket
                logger.info(
                    f"Connecting to {provider} WebSocket (attempt {retries+1}/{max_retries+1})"
                )
                self.ws_connections[provider] = await websockets.connect(
                    ws_url,
                    ping_interval=30,  # Send ping every 30 seconds
                    ping_timeout=10,  # Wait 10 seconds for pong response
                    close_timeout=5,  # Wait 5 seconds for close handshake
                )
                logger.info(f"Connected to {provider} WebSocket")

                # Send subscription message if needed
                if subscription_msg:
                    await self.ws_connections[provider].send(
                        json.dumps(subscription_msg)
                    )
                    logger.info(f"Sent subscription to {provider}")

                # For providers that require a second message after auth
                if provider == "alpaca":
                    sub_msg = {
                        "action": "subscribe",
                        "trades": symbols if channel == "trades" else [],
                        "quotes": symbols if channel == "quotes" else [],
                        "bars": symbols if channel == "bars" else [],
                    }
                    await self.ws_connections[provider].send(json.dumps(sub_msg))
                    logger.info(
                        f"Sent {channel} subscription to Alpaca for {len(symbols)} symbols"
                    )
                elif provider == "polygon":
                    sub_msg = {
                        "action": "subscribe",
                        "params": (
                            f"T.{','.join(symbols)}"
                            if channel == "trades"
                            else f"Q.{','.join(symbols)}"
                        ),
                    }
                    await self.ws_connections[provider].send(json.dumps(sub_msg))
                    logger.info(
                        f"Sent {channel} subscription to Polygon for {len(symbols)} symbols"
                    )

                # Start listening for messages
                self.running = True
                asyncio.create_task(self._listen_for_messages(provider))

                # Reset connection errors counter on successful connection
                self.health_status["reconnect_attempts"] = 0
                return

            except Exception as e:
                retries += 1
                self.health_status["reconnect_attempts"] += 1
                logger.error(
                    f"Error connecting to {provider} WebSocket (attempt {retries}/{max_retries+1}): {e}"
                )

                if retries > max_retries:
                    raise ConnectionError(
                        f"Failed to connect to {provider} WebSocket after {max_retries} attempts"
                    )

                wait_time = min(backoff, max_wait)
                logger.info(
                    f"Retrying connection to {provider} in {wait_time:.1f} seconds..."
                )
                await asyncio.sleep(wait_time)
                backoff = min(
                    backoff * self.config.retry_settings["backoff_factor"], max_wait
                )

    async def _listen_for_messages(self, provider: str):
        """
        Listen for WebSocket messages and process them with error handling.

        Args:
            provider: Data provider name
        """
        ws = self.ws_connections[provider]
        reconnect_delay = self.config.retry_settings["initial_wait"]
        max_reconnect_delay = self.config.retry_settings["max_wait"]

        while self.running:
            try:
                message = await ws.recv()
                self.data_stats["messages_received"] += 1

                try:
                    data = json.loads(message)

                    # Process message based on provider
                    try:
                        normalized_data = self._normalize_data(provider, data)

                        # Find matching callbacks and invoke them
                        for key, callback in self.callbacks.items():
                            if key.startswith(f"{provider}_"):
                                try:
                                    callback(normalized_data)
                                except Exception as e:
                                    logger.error(f"Error in callback for {key}: {e}")

                        # Publish to Kafka if configured
                        if self.kafka_producer:
                            self._publish_to_kafka(provider, normalized_data)

                        self.data_stats["messages_processed"] += 1

                    except Exception as e:
                        logger.error(f"Error normalizing {provider} data: {e}")
                        self.data_stats["validation_errors"] += 1

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {provider} WebSocket: {e}")
                    self.data_stats["validation_errors"] += 1

                # Reset reconnect delay on successful message
                reconnect_delay = self.config.retry_settings["initial_wait"]

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"{provider} WebSocket connection closed: {e}")
                break

            except Exception as e:
                logger.error(f"Error processing {provider} message: {e}")

                # Brief pause to prevent tight loop on persistent errors
                await asyncio.sleep(0.1)

        # Connection closed or error occurred, attempt reconnect if still running
        if self.running:
            logger.info(
                f"Attempting to reconnect to {provider} WebSocket in {reconnect_delay:.1f} seconds..."
            )
            await asyncio.sleep(reconnect_delay)

            # Find symbols and channels for this provider
            symbols = []
            channel = "trades"  # Default

            for key in self.callbacks.keys():
                if key.startswith(f"{provider}_"):
                    parts = key.split("_")
                    if len(parts) >= 2:
                        channel = parts[1]
                        if len(parts) >= 3:
                            symbols.extend(parts[2].split(","))

            if symbols:
                try:
                    await self._connect_websocket(provider, list(set(symbols)), channel)
                except Exception as e:
                    logger.error(f"Failed to reconnect to {provider} WebSocket: {e}")
                    # Increase reconnect delay for next attempt
                    reconnect_delay = min(
                        reconnect_delay * self.config.retry_settings["backoff_factor"],
                        max_reconnect_delay,
                    )
            else:
                logger.warning(f"No symbols found for {provider}, not reconnecting")

    def _normalize_data(self, provider: str, data: Dict) -> Dict:
        """
        Normalize data from different providers to a common format with validation.

        Args:
            provider: Data provider name
            data: Raw data from provider

        Returns:
            Normalized data dictionary

        Raises:
            DataValidationError: If data cannot be normalized
        """
        normalized = {
            "provider": provider,
            "timestamp": int(time.time() * 1000),
            "received_at": datetime.now().isoformat(),
        }

        try:
            if provider == "binance":
                if "data" in data:  # Combined stream format
                    stream_data = data["data"]
                    if "e" in stream_data and stream_data["e"] == "trade":
                        normalized.update(
                            {
                                "type": "trade",
                                "symbol": stream_data["s"],
                                "price": float(stream_data["p"]),
                                "quantity": float(stream_data["q"]),
                                "timestamp": stream_data["T"],
                                "trade_id": stream_data["t"],
                            }
                        )
                    elif "e" in stream_data and stream_data["e"] == "kline":
                        kline = stream_data["k"]
                        normalized.update(
                            {
                                "type": "bar",
                                "symbol": kline["s"],
                                "open": float(kline["o"]),
                                "high": float(kline["h"]),
                                "low": float(kline["l"]),
                                "close": float(kline["c"]),
                                "volume": float(kline["v"]),
                                "timestamp": kline["t"],
                                "interval": kline["i"],
                            }
                        )
                    else:
                        raise DataValidationError(
                            f"Unknown Binance stream type: {data}"
                        )
                elif "e" in data and data["e"] == "trade":  # Individual stream format
                    normalized.update(
                        {
                            "type": "trade",
                            "symbol": data["s"],
                            "price": float(data["p"]),
                            "quantity": float(data["q"]),
                            "timestamp": data["T"],
                            "trade_id": data["t"],
                        }
                    )
                else:
                    raise DataValidationError(f"Unknown Binance data format: {data}")

            elif provider == "coinbase":
                if data.get("type") == "match":
                    normalized.update(
                        {
                            "type": "trade",
                            "symbol": data["product_id"],
                            "price": float(data["price"]),
                            "quantity": float(data["size"]),
                            "timestamp": int(
                                datetime.fromisoformat(
                                    data["time"].replace("Z", "+00:00")
                                ).timestamp()
                                * 1000
                            ),
                            "trade_id": data["trade_id"],
                            "side": data["side"],
                        }
                    )
                elif data.get("type") == "ticker":
                    normalized.update(
                        {
                            "type": "quote",
                            "symbol": data["product_id"],
                            "bid_price": float(data["best_bid"]),
                            "ask_price": float(data["best_ask"]),
                            "price": float(data["price"]),
                            "timestamp": int(
                                datetime.fromisoformat(
                                    data["time"].replace("Z", "+00:00")
                                ).timestamp()
                                * 1000
                            ),
                        }
                    )
                elif data.get("type") == "heartbeat":
                    return {"type": "heartbeat", "provider": provider}
                else:
                    raise DataValidationError(
                        f"Unknown Coinbase data format or type: {data.get('type')}"
                    )

            elif provider == "alpaca":
                if isinstance(data, list) and data and data[0].get("T") == "t":
                    # Alpaca sends a list of data points
                    item = data[0]
                    normalized.update(
                        {
                            "type": "trade",
                            "symbol": item["S"],
                            "price": float(item["p"]),
                            "quantity": float(item["s"]),
                            "timestamp": item["t"]
                            // 1000000,  # Convert nanoseconds to milliseconds
                            "trade_id": item.get("i"),
                            "exchange": item.get("x"),
                        }
                    )
                elif isinstance(data, list) and data and data[0].get("T") == "q":
                    item = data[0]
                    normalized.update(
                        {
                            "type": "quote",
                            "symbol": item["S"],
                            "bid_price": float(item["bp"]),
                            "ask_price": float(item["ap"]),
                            "bid_size": float(item["bs"]),
                            "ask_size": float(item["as"]),
                            "timestamp": item["t"] // 1000000,
                            "exchange": item.get("x"),
                        }
                    )
                elif isinstance(data, list) and data and data[0].get("T") == "b":
                    item = data[0]
                    normalized.update(
                        {
                            "type": "bar",
                            "symbol": item["S"],
                            "open": float(item["o"]),
                            "high": float(item["h"]),
                            "low": float(item["l"]),
                            "close": float(item["c"]),
                            "volume": float(item["v"]),
                            "timestamp": item["t"] // 1000000,
                            "vwap": float(item["vw"]),
                        }
                    )
                elif data.get("T") == "subscription":
                    # Subscription response, not actual data
                    return {"type": "subscription_status", "provider": provider}
                elif data.get("T") == "success" or data.get("T") == "error":
                    # Auth success/error message
                    return {"type": "auth_status", "provider": provider}
                else:
                    raise DataValidationError(f"Unknown Alpaca data format: {data}")

            elif provider == "polygon":
                if isinstance(data, list) and data and data[0].get("ev") == "T":
                    item = data[0]
                    normalized.update(
                        {
                            "type": "trade",
                            "symbol": item["sym"],
                            "price": float(item["p"]),
                            "quantity": float(item["s"]),
                            "timestamp": item["t"],
                            "trade_id": item.get("i"),
                            "exchange": item.get("x"),
                        }
                    )
                elif isinstance(data, list) and data and data[0].get("ev") == "Q":
                    item = data[0]
                    normalized.update(
                        {
                            "type": "quote",
                            "symbol": item["sym"],
                            "bid_price": float(item["bp"]),
                            "ask_price": float(item["ap"]),
                            "bid_size": float(item["bs"]),
                            "ask_size": float(item["as"]),
                            "timestamp": item["t"],
                            "exchange": item.get("x"),
                        }
                    )
                elif data.get("ev") == "status":
                    return {"type": "status", "provider": provider}
                else:
                    raise DataValidationError(f"Unknown Polygon data format: {data}")

            # Apply general validation after normalization
            self._validate_data_format(normalized)
            return normalized

        except KeyError as e:
            raise DataValidationError(
                f"Missing key in {provider} data during normalization: {e}"
            )
        except ValueError as e:
            raise DataValidationError(
                f"Value conversion error in {provider} data during normalization: {e}"
            )

    def _validate_data_format(self, normalized_data: Dict):
        """Perform validation on normalized data format."""
        data_type = normalized_data.get("type")

        if data_type not in self.config.validation_settings["required_fields"]:
            # Allow non-standard types like 'heartbeat', 'auth_status', etc. to pass if they don't need validation
            if data_type not in [
                "heartbeat",
                "auth_status",
                "subscription_status",
                "status",
            ]:
                logger.warning(
                    f"Unknown normalized data type: {data_type}. Skipping detailed validation."
                )
            return

        required_fields = self.config.validation_settings["required_fields"][data_type]
        missing_fields = [f for f in required_fields if f not in normalized_data]
        if missing_fields:
            raise DataValidationError(
                f"Normalized data type '{data_type}' missing required fields: {', '.join(missing_fields)}"
            )

        # Re-apply type checking just in case
        for field, expected_type in self.config.validation_settings[
            "type_validation"
        ].items():
            if field in normalized_data:
                if not isinstance(normalized_data[field], expected_type):
                    # Attempt safe conversion for numbers that might be strings/ints
                    if expected_type in (int, float) and isinstance(
                        normalized_data[field], (int, float, str)
                    ):
                        try:
                            normalized_data[field] = expected_type(
                                normalized_data[field]
                            )
                        except (ValueError, TypeError):
                            raise DataValidationError(
                                f"Field {field} in normalized data has unexpected type {type(normalized_data[field])} and cannot be converted to {expected_type}"
                            )
                    elif isinstance(expected_type, tuple) and any(
                        isinstance(normalized_data[field], t) for t in expected_type
                    ):
                        pass  # Already handled by any()
                    else:
                        raise DataValidationError(
                            f"Field {field} in normalized data has unexpected type {type(normalized_data[field])}, expected {expected_type}"
                        )

    def set_kafka_producer(self, producer: Producer):
        """Set the confluent_kafka Producer instance."""
        self.kafka_producer = producer
        logger.info("Kafka producer set for market data feed")

    def _publish_to_kafka(self, provider: str, data: Dict):
        """Publish normalized data to Kafka."""
        if not self.kafka_producer:
            logger.warning("Kafka producer not configured.")
            return

        topic = (
            f"alphamind-market-data-{provider.lower()}-{data.get('type', 'general')}"
        )
        try:
            self.kafka_producer.produce(
                topic,
                value=json.dumps(data).encode("utf-8"),
                key=data.get("symbol", "N/A").encode("utf-8"),
                callback=self._delivery_report,
            )
            self.kafka_producer.poll(0)  # Trigger any callbacks
        except Exception as e:
            logger.error(f"Failed to publish message to Kafka topic {topic}: {e}")

    def _delivery_report(self, err, msg):
        """Kafka delivery report callback."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )

    def get_health_status(self) -> Dict:
        """Get the current health and performance status."""
        return {
            "is_running": self.running,
            "ws_connections": list(self.ws_connections.keys()),
            "config": {
                "max_retries": self.config.retry_settings["max_retries"],
                "validate_responses": self.config.validation_settings[
                    "validate_responses"
                ],
            },
            "metrics": {
                "last_successful_fetch": (
                    self.health_status["last_successful_fetch"].isoformat()
                    if self.health_status["last_successful_fetch"]
                    else "N/A"
                ),
                "connection_errors": self.health_status["connection_errors"],
                "data_errors": self.health_status["data_errors"],
                "reconnect_attempts": self.health_status["reconnect_attempts"],
                "messages_received": self.data_stats["messages_received"],
                "messages_processed": self.data_stats["messages_processed"],
                "validation_errors": self.data_stats["validation_errors"],
            },
        }


# Example usage (uncomment and run in an async environment)
# async def trade_callback(data: Dict):
#     if data.get("type") == "trade":
#         print(
#             f"[{data['received_at']}][{data['provider']}] TRADE: {data['symbol']} @ {data['price']} x {data['quantity']}"
#         )
#     elif data.get("type") == "heartbeat":
#         logger.debug(f"Received heartbeat from {data['provider']}")
#     elif data.get("type") == "auth_status":
#         logger.info(f"Auth status from {data['provider']}: {data}")

# async def main():
#     # Note: This requires a config.json with appropriate API keys for non-public data
#     config = MarketDataConfig(config_path="config.json")
#     feed = MarketDataFeed(config)

#     if not await feed.initialize():
#         return

#     # Optional: Set up Kafka Producer
#     # try:
#     #     producer = Producer({'bootstrap.servers': 'localhost:9092'})
#     #     feed.set_kafka_producer(producer)
#     # except Exception as e:
#     #     logger.warning(f"Could not initialize Kafka producer: {e}")

#     # Example REST fetch
#     try:
#         binance_ticker = await feed.fetch_data(
#             "binance", "/ticker/24hr", params={"symbol": "BTCUSDT"}
#         )
#         print(f"Binance BTCUSDT 24hr Ticker: {binance_ticker}")
#     except Exception as e:
#         print(f"Error fetching Binance data: {e}")

#     # Example streaming subscription (Binance)
#     try:
#         await feed.subscribe_to_stream(
#             "binance", ["BTCUSDT", "ETHUSDT"], trade_callback, channel="trade"
#         )
#     except Exception as e:
#         print(f"Error subscribing to Binance stream: {e}")

#     # Example streaming subscription (Coinbase)
#     try:
#         await feed.subscribe_to_stream(
#             "coinbase", ["BTC-USD", "ETH-USD"], trade_callback, channel="matches"
#         )
#     except Exception as e:
#         print(f"Error subscribing to Coinbase stream: {e}")


#     # Keep the main task alive to receive stream messages
#     print("Listening for streaming data... Press Ctrl+C to stop.")
#     try:
#         while feed.running:
#             await asyncio.sleep(1)
#     except asyncio.CancelledError:
#         pass
#     finally:
#         await feed.close()

# if __name__ == "__main__":
#     try:
#         # Set up asyncio loop
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(main())
#     except KeyboardInterrupt:
#         print("Program interrupted by user.")
#     except Exception as e:
#         logger.critical(f"Unhandled error in main execution: {e}")
