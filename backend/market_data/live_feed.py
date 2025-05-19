"""
Live Market Data Feed Module for AlphaMind

This module provides real-time market data integration with various exchanges
and data providers. It handles connection management, data normalization,
and streaming capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Callable

import aiohttp
import pandas as pd
import websockets
from confluent_kafka import Producer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            "initial_wait": 1.0
        }
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Default REST API endpoints
        self.endpoints = {
            "binance": "https://api.binance.com/api/v3",
            "coinbase": "https://api.exchange.coinbase.com",
            "alpaca": "https://paper-api.alpaca.markets/v2",
            "iex": "https://cloud.iexapis.com/stable",
            "polygon": "https://api.polygon.io/v2"
        }
        
        # Default WebSocket endpoints
        self.streaming_endpoints = {
            "binance": "wss://stream.binance.com:9443/ws",
            "coinbase": "wss://ws-feed.exchange.coinbase.com",
            "alpaca": "wss://paper-api.alpaca.markets/stream",
            "polygon": "wss://socket.polygon.io/stocks"
        }
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.api_keys = config.get("api_keys", {})
            self.endpoints = config.get("endpoints", self.endpoints)
            self.streaming_endpoints = config.get("streaming_endpoints", self.streaming_endpoints)
            self.retry_settings = config.get("retry_settings", self.retry_settings)
            
            logger.info(f"Loaded market data configuration from {config_path}")
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
            "retry_settings": self.retry_settings
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved market data configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")


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
    
    async def initialize(self):
        """Initialize HTTP session and connections."""
        self.session = aiohttp.ClientSession()
        logger.info("Market data feed initialized")
    
    async def close(self):
        """Close all connections and resources."""
        if self.session:
            await self.session.close()
        
        for ws in self.ws_connections.values():
            if ws and not ws.closed:
                await ws.close()
        
        self.running = False
        logger.info("Market data feed closed")
    
    async def fetch_data(self, provider: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Fetch data from REST API.
        
        Args:
            provider: Data provider name
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        if not self.session:
            await self.initialize()
        
        base_url = self.config.endpoints.get(provider)
        if not base_url:
            raise ValueError(f"Unknown provider: {provider}")
        
        url = f"{base_url}/{endpoint.lstrip('/')}"
        headers = {}
        
        # Add API key if available
        if provider in self.config.api_keys:
            if provider == "alpaca":
                headers["APCA-API-KEY-ID"] = self.config.api_keys[provider]["key"]
                headers["APCA-API-SECRET-KEY"] = self.config.api_keys[provider]["secret"]
            elif provider == "iex" or provider == "polygon":
                if params is None:
                    params = {}
                params["apiKey"] = self.config.api_keys[provider]
            else:
                headers["X-API-Key"] = self.config.api_keys[provider]
        
        # Implement retry logic
        retries = 0
        max_retries = self.config.retry_settings["max_retries"]
        backoff = self.config.retry_settings["initial_wait"]
        
        while retries <= max_retries:
            try:
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get("Retry-After", backoff))
                        logger.warning(f"Rate limited by {provider}, retrying after {retry_after}s")
                        await asyncio.sleep(retry_after)
                        retries += 1
                    else:
                        response.raise_for_status()
            except aiohttp.ClientError as e:
                logger.error(f"Error fetching data from {provider}: {e}")
                if retries >= max_retries:
                    raise
                
                await asyncio.sleep(backoff)
                backoff *= self.config.retry_settings["backoff_factor"]
                retries += 1
        
        raise RuntimeError(f"Failed to fetch data from {provider} after {max_retries} retries")
    
    async def subscribe_to_stream(self, provider: str, symbols: List[str], 
                                 callback: Callable[[Dict], None], channel: str = "trades"):
        """
        Subscribe to real-time data stream.
        
        Args:
            provider: Data provider name
            symbols: List of symbols to subscribe to
            callback: Callback function for data processing
            channel: Data channel (trades, quotes, etc.)
        """
        if provider not in self.config.streaming_endpoints:
            raise ValueError(f"Streaming not supported for provider: {provider}")
        
        # Store callback
        self.callbacks[f"{provider}_{channel}_{','.join(symbols)}"] = callback
        
        # Connect to WebSocket if not already connected
        if provider not in self.ws_connections or self.ws_connections[provider].closed:
            await self._connect_websocket(provider, symbols, channel)
    
    async def _connect_websocket(self, provider: str, symbols: List[str], channel: str):
        """
        Establish WebSocket connection to data provider.
        
        Args:
            provider: Data provider name
            symbols: List of symbols to subscribe to
            channel: Data channel (trades, quotes, etc.)
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
                "channels": [channel]
            }
        elif provider == "alpaca":
            subscription_msg = {
                "action": "auth",
                "key": self.config.api_keys[provider]["key"],
                "secret": self.config.api_keys[provider]["secret"]
            }
            # After authentication, we'll send the subscription message
        elif provider == "polygon":
            subscription_msg = {
                "action": "auth",
                "params": self.config.api_keys[provider]
            }
            # After authentication, we'll send the subscription message
        else:
            raise ValueError(f"Unsupported streaming provider: {provider}")
        
        try:
            # Connect to WebSocket
            self.ws_connections[provider] = await websockets.connect(ws_url)
            logger.info(f"Connected to {provider} WebSocket")
            
            # Send subscription message if needed
            if subscription_msg:
                await self.ws_connections[provider].send(json.dumps(subscription_msg))
                logger.info(f"Sent subscription to {provider}")
            
            # For providers that require a second message after auth
            if provider == "alpaca":
                sub_msg = {
                    "action": "subscribe",
                    "trades": symbols,
                    "quotes": symbols if channel == "quotes" else [],
                    "bars": symbols if channel == "bars" else []
                }
                await self.ws_connections[provider].send(json.dumps(sub_msg))
            elif provider == "polygon":
                sub_msg = {
                    "action": "subscribe",
                    "params": f"T.{','.join(symbols)}" if channel == "trades" else f"Q.{','.join(symbols)}"
                }
                await self.ws_connections[provider].send(json.dumps(sub_msg))
            
            # Start listening for messages
            self.running = True
            asyncio.create_task(self._listen_for_messages(provider))
            
        except Exception as e:
            logger.error(f"Error connecting to {provider} WebSocket: {e}")
            raise
    
    async def _listen_for_messages(self, provider: str):
        """
        Listen for WebSocket messages and process them.
        
        Args:
            provider: Data provider name
        """
        ws = self.ws_connections[provider]
        
        while self.running and not ws.closed:
            try:
                message = await ws.recv()
                data = json.loads(message)
                
                # Process message based on provider
                normalized_data = self._normalize_data(provider, data)
                
                # Find matching callbacks and invoke them
                for key, callback in self.callbacks.items():
                    if key.startswith(f"{provider}_"):
                        callback(normalized_data)
                
                # Publish to Kafka if configured
                if self.kafka_producer:
                    self._publish_to_kafka(provider, normalized_data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"{provider} WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error processing {provider} message: {e}")
        
        logger.info(f"Stopped listening to {provider} WebSocket")
    
    def _normalize_data(self, provider: str, data: Dict) -> Dict:
        """
        Normalize data from different providers to a common format.
        
        Args:
            provider: Data provider name
            data: Raw data from provider
            
        Returns:
            Normalized data dictionary
        """
        normalized = {
            "provider": provider,
            "timestamp": int(time.time() * 1000),
            "received_at": datetime.now().isoformat()
        }
        
        try:
            if provider == "binance":
                if "data" in data:  # Combined stream format
                    stream_data = data["data"]
                    if "e" in stream_data and stream_data["e"] == "trade":
                        normalized.update({
                            "type": "trade",
                            "symbol": stream_data["s"],
                            "price": float(stream_data["p"]),
                            "quantity": float(stream_data["q"]),
                            "timestamp": stream_data["T"],
                            "trade_id": stream_data["t"]
                        })
                    elif "e" in stream_data and stream_data["e"] == "kline":
                        k = stream_data["k"]
                        normalized.update({
                            "type": "kline",
                            "symbol": stream_data["s"],
                            "interval": k["i"],
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                            "timestamp": k["t"]
                        })
            
            elif provider == "coinbase":
                if data.get("type") == "match" or data.get("type") == "last_match":
                    normalized.update({
                        "type": "trade",
                        "symbol": data["product_id"],
                        "price": float(data["price"]),
                        "quantity": float(data["size"]),
                        "timestamp": pd.to_datetime(data["time"]).timestamp() * 1000,
                        "trade_id": data["trade_id"]
                    })
            
            elif provider == "alpaca":
                if data.get("T") == "t":  # Trade
                    normalized.update({
                        "type": "trade",
                        "symbol": data["S"],
                        "price": float(data["p"]),
                        "quantity": float(data["s"]),
                        "timestamp": pd.to_datetime(data["t"]).timestamp() * 1000,
                        "trade_id": data.get("i", "")
                    })
                elif data.get("T") == "q":  # Quote
                    normalized.update({
                        "type": "quote",
                        "symbol": data["S"],
                        "bid_price": float(data["bp"]),
                        "bid_size": float(data["bs"]),
                        "ask_price": float(data["ap"]),
                        "ask_size": float(data["as"]),
                        "timestamp": pd.to_datetime(data["t"]).timestamp() * 1000
                    })
            
            elif provider == "polygon":
                if data.get("ev") == "T":  # Trade
                    normalized.update({
                        "type": "trade",
                        "symbol": data["sym"],
                        "price": float(data["p"]),
                        "quantity": float(data["s"]),
                        "timestamp": data["t"],
                        "trade_id": data.get("i", "")
                    })
                elif data.get("ev") == "Q":  # Quote
                    normalized.update({
                        "type": "quote",
                        "symbol": data["sym"],
                        "bid_price": float(data["bp"]),
                        "bid_size": float(data["bs"]),
                        "ask_price": float(data["ap"]),
                        "ask_size": float(data["as"]),
                        "timestamp": data["t"]
                    })
        
        except Exception as e:
            logger.error(f"Error normalizing {provider} data: {e}")
            normalized["error"] = str(e)
            normalized["raw_data"] = data
        
        return normalized
    
    def configure_kafka(self, bootstrap_servers: str, topic_prefix: str = "market_data"):
        """
        Configure Kafka producer for data publishing.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic_prefix: Prefix for Kafka topics
        """
        self.kafka_config = {
            "bootstrap.servers": bootstrap_servers,
            "client.id": "alphamind-market-data"
        }
        self.kafka_topic_prefix = topic_prefix
        self.kafka_producer = Producer(self.kafka_config)
        logger.info(f"Configured Kafka producer with bootstrap servers: {bootstrap_servers}")
    
    def _publish_to_kafka(self, provider: str, data: Dict):
        """
        Publish data to Kafka.
        
        Args:
            provider: Data provider name
            data: Normalized data
        """
        if not self.kafka_producer:
            return
        
        try:
            # Determine topic based on data type and provider
            data_type = data.get("type", "unknown")
            topic = f"{self.kafka_topic_prefix}.{provider}.{data_type}"
            
            # Use symbol as key for partitioning
            key = data.get("symbol", "").encode("utf-8")
            
            # Serialize data to JSON
            value = json.dumps(data).encode("utf-8")
            
            # Produce message
            self.kafka_producer.produce(topic, key=key, value=value)
            self.kafka_producer.poll(0)  # Non-blocking poll
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {e}")


class HistoricalDataClient:
    """Client for fetching historical market data."""
    
    def __init__(self, config: MarketDataConfig):
        """
        Initialize historical data client.
        
        Args:
            config: Market data configuration
        """
        self.config = config
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession()
        logger.info("Historical data client initialized")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
        logger.info("Historical data client closed")
    
    async def fetch_historical_data(self, provider: str, symbol: str, 
                                   interval: str = "1d", 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 1h, 1d, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of records
            
        Returns:
            DataFrame with historical data
        """
        if not self.session:
            await self.initialize()
        
        if provider == "binance":
            return await self._fetch_binance_historical(symbol, interval, start_date, end_date, limit)
        elif provider == "alpaca":
            return await self._fetch_alpaca_historical(symbol, interval, start_date, end_date, limit)
        elif provider == "polygon":
            return await self._fetch_polygon_historical(symbol, interval, start_date, end_date, limit)
        else:
            raise ValueError(f"Historical data not supported for provider: {provider}")
    
    async def _fetch_binance_historical(self, symbol: str, interval: str, 
                                      start_date: Optional[str], 
                                      end_date: Optional[str],
                                      limit: int) -> pd.DataFrame:
        """Fetch historical data from Binance."""
        endpoint = "/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        # Convert dates to timestamps if provided
        if start_date:
            start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
            params["startTime"] = start_ts
        
        if end_date:
            end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
            params["endTime"] = end_ts
        
        base_url = self.config.endpoints.get("binance")
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                    ])
                    
                    # Convert types
                    numeric_columns = ["open", "high", "low", "close", "volume", 
                                      "quote_asset_volume", "taker_buy_base_asset_volume", 
                                      "taker_buy_quote_asset_volume"]
                    
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    
                    return df
                else:
                    response.raise_for_status()
        except Exception as e:
            logger.error(f"Error fetching Binance historical data: {e}")
            raise
    
    async def _fetch_alpaca_historical(self, symbol: str, interval: str, 
                                     start_date: Optional[str], 
                                     end_date: Optional[str],
                                     limit: int) -> pd.DataFrame:
        """Fetch historical data from Alpaca."""
        # Map interval to Alpaca timeframe
        interval_map = {
            "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
            "1h": "1Hour", "1d": "1Day"
        }
        
        alpaca_interval = interval_map.get(interval, "1Day")
        endpoint = f"/stocks/{symbol}/bars"
        
        params = {
            "timeframe": alpaca_interval,
            "limit": limit
        }
        
        if start_date:
            params["start"] = start_date
        
        if end_date:
            params["end"] = end_date
        
        headers = {
            "APCA-API-KEY-ID": self.config.api_keys["alpaca"]["key"],
            "APCA-API-SECRET-KEY": self.config.api_keys["alpaca"]["secret"]
        }
        
        base_url = self.config.endpoints.get("alpaca")
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data["bars"])
                    
                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["t"])
                    df.set_index("timestamp", inplace=True)
                    
                    # Rename columns
                    df.rename(columns={
                        "o": "open", "h": "high", "l": "low", 
                        "c": "close", "v": "volume"
                    }, inplace=True)
                    
                    return df
                else:
                    response.raise_for_status()
        except Exception as e:
            logger.error(f"Error fetching Alpaca historical data: {e}")
            raise
    
    async def _fetch_polygon_historical(self, symbol: str, interval: str, 
                                      start_date: Optional[str], 
                                      end_date: Optional[str],
                                      limit: int) -> pd.DataFrame:
        """Fetch historical data from Polygon."""
        # Map interval to Polygon timespan
        interval_map = {
            "1m": "minute", "5m": "minute", "15m": "minute", "30m": "minute",
            "1h": "hour", "1d": "day"
        }
        
        # Extract multiplier from interval
        if interval in ["5m", "15m", "30m"]:
            multiplier = interval[:-1]
        elif interval == "1m":
            multiplier = 1
        elif interval == "1h":
            multiplier = 1
        elif interval == "1d":
            multiplier = 1
        else:
            multiplier = 1
        
        timespan = interval_map.get(interval, "day")
        endpoint = f"/aggs/ticker/{symbol}/range/{multiplier}/{timespan}"
        
        params = {
            "apiKey": self.config.api_keys["polygon"],
            "limit": limit
        }
        
        if start_date:
            params["from"] = start_date.replace("-", "")
        
        if end_date:
            params["to"] = end_date.replace("-", "")
        
        base_url = self.config.endpoints.get("polygon")
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to DataFrame
                    if "results" in data and data["results"]:
                        df = pd.DataFrame(data["results"])
                        
                        # Convert timestamp to datetime
                        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                        df.set_index("timestamp", inplace=True)
                        
                        # Rename columns
                        df.rename(columns={
                            "o": "open", "h": "high", "l": "low", 
                            "c": "close", "v": "volume"
                        }, inplace=True)
                        
                        return df
                    else:
                        return pd.DataFrame()
                else:
                    response.raise_for_status()
        except Exception as e:
            logger.error(f"Error fetching Polygon historical data: {e}")
            raise


class MarketDataManager:
    """Manager for all market data operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize market data manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = MarketDataConfig(config_path)
        self.live_feed = MarketDataFeed(self.config)
        self.historical_client = HistoricalDataClient(self.config)
        self.data_cache = {}
        self.running = False
    
    async def initialize(self):
        """Initialize all components."""
        await self.live_feed.initialize()
        await self.historical_client.initialize()
        self.running = True
        logger.info("Market data manager initialized")
    
    async def close(self):
        """Close all connections and resources."""
        self.running = False
        await self.live_feed.close()
        await self.historical_client.close()
        logger.info("Market data manager closed")
    
    async def get_live_data(self, provider: str, symbols: List[str], 
                          callback: Callable[[Dict], None], channel: str = "trades"):
        """
        Subscribe to live market data.
        
        Args:
            provider: Data provider name
            symbols: List of symbols to subscribe to
            callback: Callback function for data processing
            channel: Data channel (trades, quotes, etc.)
        """
        await self.live_feed.subscribe_to_stream(provider, symbols, callback, channel)
    
    async def get_historical_data(self, provider: str, symbol: str, 
                                interval: str = "1d", 
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                limit: int = 1000) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            provider: Data provider name
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 1h, 1d, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of records
            
        Returns:
            DataFrame with historical data
        """
        return await self.historical_client.fetch_historical_data(
            provider, symbol, interval, start_date, end_date, limit
        )
    
    def configure_kafka(self, bootstrap_servers: str, topic_prefix: str = "market_data"):
        """
        Configure Kafka integration for data streaming.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic_prefix: Prefix for Kafka topics
        """
        self.live_feed.configure_kafka(bootstrap_servers, topic_prefix)
    
    def save_config(self, config_path: str):
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        self.config.save_to_file(config_path)


# Example usage
async def example_usage():
    """Example of how to use the market data module."""
    # Initialize manager
    manager = MarketDataManager()
    await manager.initialize()
    
    try:
        # Define callback for live data
        def process_trade(data):
            print(f"Received trade: {data}")
        
        # Subscribe to live data
        await manager.get_live_data("binance", ["BTCUSDT", "ETHUSDT"], process_trade)
        
        # Fetch historical data
        df = await manager.get_historical_data(
            "binance", "BTCUSDT", interval="1d", 
            start_date="2023-01-01", end_date="2023-01-31"
        )
        print(f"Historical data:\n{df.head()}")
        
        # Keep running for a while to receive live data
        await asyncio.sleep(60)
        
    finally:
        # Clean up
        await manager.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
