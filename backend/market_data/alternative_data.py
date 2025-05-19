"""
Alternative Data Plugin Architecture for AlphaMind

This module provides a flexible plugin system for integrating alternative data sources
such as news, social media, satellite imagery, and other non-traditional data
into trading strategies and analysis.
"""

import asyncio
import importlib
import inspect
import json
import logging
import os
import pkgutil
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Callable, Any, Tuple, Set, Type

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of alternative data sources."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    SATELLITE = "satellite"
    WEATHER = "weather"
    ECONOMIC = "economic"
    SENTIMENT = "sentiment"
    TRANSACTION = "transaction"
    WEB_TRAFFIC = "web_traffic"
    SUPPLY_CHAIN = "supply_chain"
    CUSTOM = "custom"


class DataFrequency(Enum):
    """Data update frequency."""
    REAL_TIME = "real_time"  # Continuous updates
    INTRADAY = "intraday"    # Multiple times per day
    DAILY = "daily"          # Once per day
    WEEKLY = "weekly"        # Once per week
    MONTHLY = "monthly"      # Once per month
    QUARTERLY = "quarterly"  # Once per quarter
    ANNUAL = "annual"        # Once per year
    CUSTOM = "custom"        # Custom frequency


class DataFormat(Enum):
    """Data output format."""
    JSON = "json"
    CSV = "csv"
    DATAFRAME = "dataframe"
    TEXT = "text"
    IMAGE = "image"
    BINARY = "binary"
    CUSTOM = "custom"


class PluginStatus(Enum):
    """Plugin status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"
    UNLOADED = "unloaded"


class PluginMetadata:
    """Metadata for data source plugins."""
    
    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        author: str,
        source_type: DataSourceType,
        frequency: DataFrequency,
        output_format: DataFormat,
        requires_api_key: bool = False,
        requires_authentication: bool = False,
        free_tier_available: bool = True,
        rate_limit: Optional[int] = None,
        documentation_url: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize plugin metadata.
        
        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
            author: Plugin author
            source_type: Type of data source
            frequency: Data update frequency
            output_format: Data output format
            requires_api_key: Whether plugin requires API key
            requires_authentication: Whether plugin requires authentication
            free_tier_available: Whether free tier is available
            rate_limit: Rate limit (requests per minute)
            documentation_url: URL to plugin documentation
            tags: List of tags for categorization
        """
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.source_type = source_type
        self.frequency = frequency
        self.output_format = output_format
        self.requires_api_key = requires_api_key
        self.requires_authentication = requires_authentication
        self.free_tier_available = free_tier_available
        self.rate_limit = rate_limit
        self.documentation_url = documentation_url
        self.tags = tags or []
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.
        
        Returns:
            Dictionary representation of metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "source_type": self.source_type.value,
            "frequency": self.frequency.value,
            "output_format": self.output_format.value,
            "requires_api_key": self.requires_api_key,
            "requires_authentication": self.requires_authentication,
            "free_tier_available": self.free_tier_available,
            "rate_limit": self.rate_limit,
            "documentation_url": self.documentation_url,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """
        Create metadata from dictionary.
        
        Args:
            data: Dictionary representation of metadata
            
        Returns:
            PluginMetadata instance
        """
        metadata = cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            source_type=DataSourceType(data["source_type"]),
            frequency=DataFrequency(data["frequency"]),
            output_format=DataFormat(data["output_format"]),
            requires_api_key=data["requires_api_key"],
            requires_authentication=data["requires_authentication"],
            free_tier_available=data["free_tier_available"],
            rate_limit=data["rate_limit"],
            documentation_url=data["documentation_url"],
            tags=data["tags"]
        )
        
        if "created_at" in data:
            metadata.created_at = datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            metadata.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return metadata


class DataSourcePlugin(ABC):
    """Base class for data source plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data source plugin.
        
        Args:
            config: Plugin configuration
        """
        self.config = config or {}
        self.metadata = self._get_metadata()
        self.status = PluginStatus.INACTIVE
        self.last_error = None
        self.last_fetch_time = None
        self.cache = {}
        self.cache_expiry = {}
    
    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        pass
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> Any:
        """
        Fetch data from source.
        
        Args:
            **kwargs: Additional parameters for data fetching
            
        Returns:
            Fetched data
        """
        pass
    
    async def initialize(self) -> bool:
        """
        Initialize plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.status = PluginStatus.LOADING
            
            # Validate configuration
            if not self._validate_config():
                self.status = PluginStatus.ERROR
                self.last_error = "Invalid configuration"
                return False
            
            # Perform any necessary setup
            await self._setup()
            
            self.status = PluginStatus.ACTIVE
            return True
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Error initializing plugin {self.metadata.name}: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """
        Shutdown plugin.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        try:
            # Perform any necessary cleanup
            await self._cleanup()
            
            self.status = PluginStatus.UNLOADED
            return True
        except Exception as e:
            self.status = PluginStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Error shutting down plugin {self.metadata.name}: {e}")
            return False
    
    async def get_data(self, use_cache: bool = True, cache_ttl: Optional[int] = None, **kwargs) -> Any:
        """
        Get data from source, with optional caching.
        
        Args:
            use_cache: Whether to use cached data if available
            cache_ttl: Cache time-to-live in seconds (if None, use default)
            **kwargs: Additional parameters for data fetching
            
        Returns:
            Fetched data
        """
        if self.status != PluginStatus.ACTIVE:
            raise ValueError(f"Plugin {self.metadata.name} is not active")
        
        # Generate cache key from kwargs
        cache_key = self._generate_cache_key(**kwargs)
        
        # Check cache if enabled
        if use_cache and cache_key in self.cache:
            if cache_key in self.cache_expiry and datetime.now() < self.cache_expiry[cache_key]:
                logger.debug(f"Using cached data for {self.metadata.name}")
                return self.cache[cache_key]
        
        try:
            # Fetch data
            data = await self.fetch_data(**kwargs)
            self.last_fetch_time = datetime.now()
            
            # Update cache if enabled
            if use_cache:
                self.cache[cache_key] = data
                
                # Set cache expiry
                if cache_ttl is not None:
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=cache_ttl)
                else:
                    # Default cache TTL based on data frequency
                    default_ttl = self._get_default_cache_ttl()
                    self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=default_ttl)
            
            return data
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error fetching data from {self.metadata.name}: {e}")
            raise
    
    def _validate_config(self) -> bool:
        """
        Validate plugin configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check for required API key if needed
        if self.metadata.requires_api_key and "api_key" not in self.config:
            logger.error(f"API key required for {self.metadata.name}")
            return False
        
        # Check for required authentication if needed
        if self.metadata.requires_authentication and (
            "username" not in self.config or "password" not in self.config
        ):
            logger.error(f"Authentication required for {self.metadata.name}")
            return False
        
        return True
    
    async def _setup(self):
        """Perform plugin setup."""
        pass
    
    async def _cleanup(self):
        """Perform plugin cleanup."""
        pass
    
    def _generate_cache_key(self, **kwargs) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            **kwargs: Parameters for data fetching
            
        Returns:
            Cache key
        """
        # Sort kwargs by key to ensure consistent cache keys
        sorted_items = sorted(kwargs.items())
        
        # Convert to string
        key_parts = [f"{k}={v}" for k, v in sorted_items]
        key_str = ",".join(key_parts)
        
        return key_str
    
    def _get_default_cache_ttl(self) -> int:
        """
        Get default cache TTL based on data frequency.
        
        Returns:
            Cache TTL in seconds
        """
        if self.metadata.frequency == DataFrequency.REAL_TIME:
            return 60  # 1 minute
        elif self.metadata.frequency == DataFrequency.INTRADAY:
            return 300  # 5 minutes
        elif self.metadata.frequency == DataFrequency.DAILY:
            return 86400  # 1 day
        elif self.metadata.frequency == DataFrequency.WEEKLY:
            return 604800  # 1 week
        elif self.metadata.frequency == DataFrequency.MONTHLY:
            return 2592000  # 30 days
        elif self.metadata.frequency == DataFrequency.QUARTERLY:
            return 7776000  # 90 days
        elif self.metadata.frequency == DataFrequency.ANNUAL:
            return 31536000  # 365 days
        else:
            return 3600  # 1 hour default


class PluginManager:
    """Manager for data source plugins."""
    
    def __init__(self, plugin_dir: Optional[str] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dir: Directory containing plugins
        """
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, DataSourcePlugin] = {}
        self.plugin_classes: Dict[str, Type[DataSourcePlugin]] = {}
    
    async def discover_plugins(self) -> List[str]:
        """
        Discover available plugins.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        # Discover built-in plugins
        discovered.extend(await self._discover_builtin_plugins())
        
        # Discover plugins from directory if specified
        if self.plugin_dir:
            discovered.extend(await self._discover_directory_plugins())
        
        return discovered
    
    async def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load and initialize plugin.
        
        Args:
            plugin_name: Plugin name
            config: Plugin configuration
            
        Returns:
            True if plugin loaded successfully, False otherwise
        """
        if plugin_name in self.plugins:
            logger.warning(f"Plugin {plugin_name} already loaded")
            return True
        
        try:
            # Create plugin instance
            if plugin_name in self.plugin_classes:
                plugin_class = self.plugin_classes[plugin_name]
                plugin = plugin_class(config)
                
                # Initialize plugin
                if await plugin.initialize():
                    self.plugins[plugin_name] = plugin
                    logger.info(f"Loaded plugin: {plugin_name}")
                    return True
                else:
                    logger.error(f"Failed to initialize plugin: {plugin_name}")
                    return False
            else:
                logger.error(f"Plugin not found: {plugin_name}")
                return False
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload plugin.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            True if plugin unloaded successfully, False otherwise
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return True
        
        try:
            plugin = self.plugins[plugin_name]
            
            # Shutdown plugin
            if await plugin.shutdown():
                del self.plugins[plugin_name]
                logger.info(f"Unloaded plugin: {plugin_name}")
                return True
            else:
                logger.error(f"Failed to shutdown plugin: {plugin_name}")
                return False
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    async def get_plugin_data(
        self,
        plugin_name: str,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Get data from plugin.
        
        Args:
            plugin_name: Plugin name
            use_cache: Whether to use cached data if available
            cache_ttl: Cache time-to-live in seconds
            **kwargs: Additional parameters for data fetching
            
        Returns:
            Fetched data
        """
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not loaded")
        
        plugin = self.plugins[plugin_name]
        return await plugin.get_data(use_cache=use_cache, cache_ttl=cache_ttl, **kwargs)
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin metadata or None if plugin not found
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].metadata
        elif plugin_name in self.plugin_classes:
            # Create temporary instance to get metadata
            plugin = self.plugin_classes[plugin_name]({})
            return plugin._get_metadata()
        else:
            return None
    
    def get_all_plugins_metadata(self) -> Dict[str, PluginMetadata]:
        """
        Get metadata for all available plugins.
        
        Returns:
            Dictionary of plugin name to metadata
        """
        metadata = {}
        
        # Get metadata for loaded plugins
        for name, plugin in self.plugins.items():
            metadata[name] = plugin.metadata
        
        # Get metadata for discovered but not loaded plugins
        for name, plugin_class in self.plugin_classes.items():
            if name not in metadata:
                plugin = plugin_class({})
                metadata[name] = plugin._get_metadata()
        
        return metadata
    
    def get_plugins_by_type(self, source_type: DataSourceType) -> List[str]:
        """
        Get plugins by source type.
        
        Args:
            source_type: Data source type
            
        Returns:
            List of plugin names
        """
        plugins = []
        
        for name, metadata in self.get_all_plugins_metadata().items():
            if metadata.source_type == source_type:
                plugins.append(name)
        
        return plugins
    
    def get_plugins_by_tag(self, tag: str) -> List[str]:
        """
        Get plugins by tag.
        
        Args:
            tag: Plugin tag
            
        Returns:
            List of plugin names
        """
        plugins = []
        
        for name, metadata in self.get_all_plugins_metadata().items():
            if tag in metadata.tags:
                plugins.append(name)
        
        return plugins
    
    async def _discover_builtin_plugins(self) -> List[str]:
        """
        Discover built-in plugins.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        # Register built-in plugins
        self._register_plugin_class("NewsAPI", NewsAPIPlugin)
        discovered.append("NewsAPI")
        
        self._register_plugin_class("TwitterSentiment", TwitterSentimentPlugin)
        discovered.append("TwitterSentiment")
        
        self._register_plugin_class("RedditSentiment", RedditSentimentPlugin)
        discovered.append("RedditSentiment")
        
        self._register_plugin_class("SatelliteImagery", SatelliteImageryPlugin)
        discovered.append("SatelliteImagery")
        
        self._register_plugin_class("WeatherData", WeatherDataPlugin)
        discovered.append("WeatherData")
        
        self._register_plugin_class("EconomicIndicators", EconomicIndicatorsPlugin)
        discovered.append("EconomicIndicators")
        
        self._register_plugin_class("WebTrafficAnalytics", WebTrafficAnalyticsPlugin)
        discovered.append("WebTrafficAnalytics")
        
        return discovered
    
    async def _discover_directory_plugins(self) -> List[str]:
        """
        Discover plugins from directory.
        
        Returns:
            List of discovered plugin names
        """
        if not self.plugin_dir or not os.path.isdir(self.plugin_dir):
            return []
        
        discovered = []
        
        # Add plugin directory to path
        sys.path.insert(0, self.plugin_dir)
        
        try:
            # Iterate through Python files in directory
            for filename in os.listdir(self.plugin_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = filename[:-3]  # Remove .py extension
                    
                    try:
                        # Import module
                        module = importlib.import_module(module_name)
                        
                        # Find plugin classes in module
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (
                                issubclass(obj, DataSourcePlugin) and 
                                obj != DataSourcePlugin and
                                obj.__module__ == module.__name__
                            ):
                                self._register_plugin_class(name, obj)
                                discovered.append(name)
                    except Exception as e:
                        logger.error(f"Error loading plugin module {module_name}: {e}")
        finally:
            # Remove plugin directory from path
            if self.plugin_dir in sys.path:
                sys.path.remove(self.plugin_dir)
        
        return discovered
    
    def _register_plugin_class(self, name: str, plugin_class: Type[DataSourcePlugin]):
        """
        Register plugin class.
        
        Args:
            name: Plugin name
            plugin_class: Plugin class
        """
        self.plugin_classes[name] = plugin_class


class NewsAPIPlugin(DataSourcePlugin):
    """Plugin for fetching news data from NewsAPI."""
    
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return PluginMetadata(
            name="NewsAPI",
            version="1.0.0",
            description="Fetches news articles from various sources using NewsAPI",
            author="AlphaMind Team",
            source_type=DataSourceType.NEWS,
            frequency=DataFrequency.INTRADAY,
            output_format=DataFormat.DATAFRAME,
            requires_api_key=True,
            requires_authentication=False,
            free_tier_available=True,
            rate_limit=100,
            documentation_url="https://newsapi.org/docs",
            tags=["news", "articles", "headlines"]
        )
    
    async def fetch_data(
        self,
        query: Optional[str] = None,
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
        page: int = 1
    ) -> pd.DataFrame:
        """
        Fetch news data.
        
        Args:
            query: Search query
            sources: List of news sources
            categories: List of news categories
            countries: List of country codes
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code
            sort_by: Sort order (publishedAt, relevancy, popularity)
            page_size: Number of results per page
            page: Page number
            
        Returns:
            DataFrame of news articles
        """
        if "api_key" not in self.config:
            raise ValueError("API key required for NewsAPI")
        
        api_key = self.config["api_key"]
        base_url = "https://newsapi.org/v2"
        
        # Determine endpoint
        if query:
            endpoint = "/everything"
        else:
            endpoint = "/top-headlines"
        
        # Build parameters
        params = {
            "apiKey": api_key,
            "language": language,
            "pageSize": page_size,
            "page": page
        }
        
        if query:
            params["q"] = query
            params["sortBy"] = sort_by
            
            if from_date:
                params["from"] = from_date
            
            if to_date:
                params["to"] = to_date
            
            if sources:
                params["sources"] = ",".join(sources)
        else:
            if categories:
                params["category"] = categories[0]  # API only supports one category
            
            if countries:
                params["country"] = countries[0]  # API only supports one country
            
            if sources:
                params["sources"] = ",".join(sources)
        
        # Make request
        url = f"{base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "ok":
                raise ValueError(f"API error: {data.get('message', 'Unknown error')}")
            
            # Convert to DataFrame
            articles = data["articles"]
            df = pd.DataFrame(articles)
            
            # Extract source name
            if "source" in df.columns:
                df["source_name"] = df["source"].apply(lambda x: x.get("name", "") if isinstance(x, dict) else "")
                df.drop("source", axis=1, inplace=True)
            
            # Convert publishedAt to datetime
            if "publishedAt" in df.columns:
                df["publishedAt"] = pd.to_datetime(df["publishedAt"])
            
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {e}")
            raise


class TwitterSentimentPlugin(DataSourcePlugin):
    """Plugin for fetching and analyzing Twitter sentiment."""
    
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return PluginMetadata(
            name="TwitterSentiment",
            version="1.0.0",
            description="Fetches tweets and analyzes sentiment for specified keywords or accounts",
            author="AlphaMind Team",
            source_type=DataSourceType.SOCIAL_MEDIA,
            frequency=DataFrequency.REAL_TIME,
            output_format=DataFormat.DATAFRAME,
            requires_api_key=True,
            requires_authentication=True,
            free_tier_available=False,
            rate_limit=450,
            documentation_url="https://developer.twitter.com/en/docs",
            tags=["twitter", "social media", "sentiment"]
        )
    
    async def fetch_data(
        self,
        query: Optional[str] = None,
        accounts: Optional[List[str]] = None,
        hashtags: Optional[List[str]] = None,
        count: int = 100,
        include_sentiment: bool = True,
        include_entities: bool = False,
        language: str = "en",
        result_type: str = "mixed",
        since_id: Optional[str] = None,
        max_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch Twitter data and analyze sentiment.
        
        Args:
            query: Search query
            accounts: List of Twitter accounts
            hashtags: List of hashtags
            count: Number of tweets to fetch
            include_sentiment: Whether to include sentiment analysis
            include_entities: Whether to include entities (hashtags, mentions, etc.)
            language: Language code
            result_type: Result type (mixed, recent, popular)
            since_id: Only return results with ID greater than this
            max_id: Only return results with ID less than this
            
        Returns:
            DataFrame of tweets with sentiment
        """
        if "api_key" not in self.config or "api_secret" not in self.config:
            raise ValueError("API key and secret required for Twitter API")
        
        if "access_token" not in self.config or "access_token_secret" not in self.config:
            raise ValueError("Access token and secret required for Twitter API")
        
        # In a real implementation, this would use the Twitter API
        # For this example, we'll simulate the response
        
        # Simulate tweets
        tweets = []
        current_time = datetime.now()
        
        for i in range(count):
            tweet_time = current_time - timedelta(minutes=i)
            
            tweet = {
                "id": f"1{i:09d}",
                "created_at": tweet_time.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": f"Sample tweet about {query or hashtags or accounts} #{i}",
                "user": {
                    "id": f"user{i % 10}",
                    "screen_name": f"user{i % 10}",
                    "name": f"User {i % 10}",
                    "followers_count": 1000 + (i % 1000),
                    "friends_count": 500 + (i % 500)
                },
                "retweet_count": i % 100,
                "favorite_count": i % 200,
                "lang": language
            }
            
            tweets.append(tweet)
        
        # Convert to DataFrame
        df = pd.DataFrame(tweets)
        
        # Extract user information
        if "user" in df.columns:
            user_df = pd.json_normalize(df["user"])
            user_cols = ["id", "screen_name", "name", "followers_count", "friends_count"]
            for col in user_cols:
                if col in user_df.columns:
                    df[f"user_{col}"] = user_df[col]
            df.drop("user", axis=1, inplace=True)
        
        # Convert created_at to datetime
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S +0000 %Y")
        
        # Add sentiment analysis if requested
        if include_sentiment:
            # Simulate sentiment analysis
            np.random.seed(42)  # For reproducibility
            df["sentiment_score"] = np.random.normal(0, 0.5, size=len(df))
            df["sentiment_magnitude"] = np.abs(df["sentiment_score"]) + np.random.uniform(0, 0.5, size=len(df))
            df["sentiment"] = pd.cut(
                df["sentiment_score"],
                bins=[-1, -0.3, 0.3, 1],
                labels=["negative", "neutral", "positive"]
            )
        
        return df


class RedditSentimentPlugin(DataSourcePlugin):
    """Plugin for fetching and analyzing Reddit sentiment."""
    
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return PluginMetadata(
            name="RedditSentiment",
            version="1.0.0",
            description="Fetches posts and comments from Reddit and analyzes sentiment",
            author="AlphaMind Team",
            source_type=DataSourceType.SOCIAL_MEDIA,
            frequency=DataFrequency.INTRADAY,
            output_format=DataFormat.DATAFRAME,
            requires_api_key=True,
            requires_authentication=True,
            free_tier_available=True,
            rate_limit=60,
            documentation_url="https://www.reddit.com/dev/api/",
            tags=["reddit", "social media", "sentiment"]
        )
    
    async def fetch_data(
        self,
        subreddits: List[str],
        query: Optional[str] = None,
        time_period: str = "day",
        sort: str = "hot",
        limit: int = 100,
        include_comments: bool = True,
        include_sentiment: bool = True
    ) -> pd.DataFrame:
        """
        Fetch Reddit data and analyze sentiment.
        
        Args:
            subreddits: List of subreddits
            query: Search query
            time_period: Time period (hour, day, week, month, year, all)
            sort: Sort order (hot, new, top, rising, controversial)
            limit: Number of posts to fetch
            include_comments: Whether to include comments
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            DataFrame of Reddit posts and comments with sentiment
        """
        if "client_id" not in self.config or "client_secret" not in self.config:
            raise ValueError("Client ID and secret required for Reddit API")
        
        if "username" not in self.config or "password" not in self.config:
            raise ValueError("Username and password required for Reddit API")
        
        # In a real implementation, this would use the Reddit API
        # For this example, we'll simulate the response
        
        # Simulate posts
        posts = []
        current_time = datetime.now()
        
        for i in range(limit):
            post_time = current_time - timedelta(hours=i)
            subreddit = subreddits[i % len(subreddits)]
            
            post = {
                "id": f"t3_{i:06d}",
                "subreddit": subreddit,
                "title": f"Sample post about {query or subreddit} #{i}",
                "selftext": f"This is the content of post #{i} in r/{subreddit}",
                "author": f"user{i % 20}",
                "created_utc": post_time.timestamp(),
                "score": 100 + (i % 1000),
                "upvote_ratio": 0.5 + (i % 100) / 200,
                "num_comments": i % 50
            }
            
            posts.append(post)
        
        # Convert to DataFrame
        df = pd.DataFrame(posts)
        
        # Convert created_utc to datetime
        if "created_utc" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_utc"], unit="s")
            df.drop("created_utc", axis=1, inplace=True)
        
        # Add comments if requested
        if include_comments:
            # Simulate comments
            comments = []
            
            for i, post in enumerate(posts):
                num_comments = post["num_comments"]
                
                for j in range(num_comments):
                    comment_time = pd.to_datetime(post["created_utc"], unit="s") + timedelta(minutes=j)
                    
                    comment = {
                        "id": f"t1_{i:06d}_{j:03d}",
                        "post_id": post["id"],
                        "subreddit": post["subreddit"],
                        "body": f"This is comment #{j} on post #{i}",
                        "author": f"commenter{j % 50}",
                        "created_utc": comment_time.timestamp(),
                        "score": j % 100,
                        "is_submitter": j == 0
                    }
                    
                    comments.append(comment)
            
            # Convert to DataFrame
            comments_df = pd.DataFrame(comments)
            
            # Convert created_utc to datetime
            if "created_utc" in comments_df.columns:
                comments_df["created_at"] = pd.to_datetime(comments_df["created_utc"], unit="s")
                comments_df.drop("created_utc", axis=1, inplace=True)
            
            # Add sentiment analysis if requested
            if include_sentiment:
                # Simulate sentiment analysis for posts
                np.random.seed(42)  # For reproducibility
                df["sentiment_score"] = np.random.normal(0, 0.5, size=len(df))
                df["sentiment_magnitude"] = np.abs(df["sentiment_score"]) + np.random.uniform(0, 0.5, size=len(df))
                df["sentiment"] = pd.cut(
                    df["sentiment_score"],
                    bins=[-1, -0.3, 0.3, 1],
                    labels=["negative", "neutral", "positive"]
                )
                
                # Simulate sentiment analysis for comments
                comments_df["sentiment_score"] = np.random.normal(0, 0.5, size=len(comments_df))
                comments_df["sentiment_magnitude"] = np.abs(comments_df["sentiment_score"]) + np.random.uniform(0, 0.5, size=len(comments_df))
                comments_df["sentiment"] = pd.cut(
                    comments_df["sentiment_score"],
                    bins=[-1, -0.3, 0.3, 1],
                    labels=["negative", "neutral", "positive"]
                )
            
            # Return both posts and comments
            return {
                "posts": df,
                "comments": comments_df
            }
        
        # Add sentiment analysis if requested
        if include_sentiment:
            # Simulate sentiment analysis
            np.random.seed(42)  # For reproducibility
            df["sentiment_score"] = np.random.normal(0, 0.5, size=len(df))
            df["sentiment_magnitude"] = np.abs(df["sentiment_score"]) + np.random.uniform(0, 0.5, size=len(df))
            df["sentiment"] = pd.cut(
                df["sentiment_score"],
                bins=[-1, -0.3, 0.3, 1],
                labels=["negative", "neutral", "positive"]
            )
        
        return df


class SatelliteImageryPlugin(DataSourcePlugin):
    """Plugin for fetching and analyzing satellite imagery."""
    
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return PluginMetadata(
            name="SatelliteImagery",
            version="1.0.0",
            description="Fetches and analyzes satellite imagery for specified locations",
            author="AlphaMind Team",
            source_type=DataSourceType.SATELLITE,
            frequency=DataFrequency.DAILY,
            output_format=DataFormat.IMAGE,
            requires_api_key=True,
            requires_authentication=False,
            free_tier_available=False,
            rate_limit=10,
            documentation_url="https://www.planet.com/docs/",
            tags=["satellite", "imagery", "geospatial"]
        )
    
    async def fetch_data(
        self,
        location: Union[str, Tuple[float, float]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resolution: str = "medium",
        image_type: str = "visual",
        cloud_cover_max: float = 0.1,
        analyze: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch satellite imagery.
        
        Args:
            location: Location name or coordinates (latitude, longitude)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resolution: Image resolution (low, medium, high)
            image_type: Image type (visual, ndvi, infrared)
            cloud_cover_max: Maximum cloud cover (0-1)
            analyze: Whether to analyze the imagery
            
        Returns:
            Dictionary with imagery data and analysis
        """
        if "api_key" not in self.config:
            raise ValueError("API key required for satellite imagery API")
        
        # In a real implementation, this would use a satellite imagery API
        # For this example, we'll simulate the response
        
        # Parse location
        if isinstance(location, str):
            # Simulate geocoding
            latitude = 37.7749
            longitude = -122.4194
        else:
            latitude, longitude = location
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Simulate imagery data
        imagery_data = {
            "location": {
                "name": location if isinstance(location, str) else f"{latitude}, {longitude}",
                "latitude": latitude,
                "longitude": longitude
            },
            "time_range": {
                "start_date": start_date,
                "end_date": end_date
            },
            "parameters": {
                "resolution": resolution,
                "image_type": image_type,
                "cloud_cover_max": cloud_cover_max
            },
            "images": []
        }
        
        # Generate simulated images
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1
        
        # Limit to 10 images for simulation
        num_images = min(days, 10)
        
        for i in range(num_images):
            image_date = start_dt + timedelta(days=i * days // num_images)
            cloud_cover = np.random.uniform(0, cloud_cover_max)
            
            image = {
                "id": f"img_{i:04d}",
                "date": image_date.strftime("%Y-%m-%d"),
                "cloud_cover": cloud_cover,
                "resolution": resolution,
                "type": image_type,
                "url": f"https://example.com/satellite/{latitude}_{longitude}_{image_date.strftime('%Y%m%d')}_{image_type}.jpg"
            }
            
            imagery_data["images"].append(image)
        
        # Add analysis if requested
        if analyze:
            # Simulate analysis
            analysis = {
                "vegetation_index": np.random.uniform(0.3, 0.8),
                "urban_area_percentage": np.random.uniform(0.1, 0.9),
                "water_area_percentage": np.random.uniform(0, 0.3),
                "change_detection": {
                    "vegetation_change": np.random.uniform(-0.1, 0.1),
                    "urban_expansion": np.random.uniform(0, 0.05),
                    "water_level_change": np.random.uniform(-0.05, 0.05)
                }
            }
            
            imagery_data["analysis"] = analysis
        
        return imagery_data


class WeatherDataPlugin(DataSourcePlugin):
    """Plugin for fetching weather data."""
    
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return PluginMetadata(
            name="WeatherData",
            version="1.0.0",
            description="Fetches historical and forecast weather data for specified locations",
            author="AlphaMind Team",
            source_type=DataSourceType.WEATHER,
            frequency=DataFrequency.INTRADAY,
            output_format=DataFormat.DATAFRAME,
            requires_api_key=True,
            requires_authentication=False,
            free_tier_available=True,
            rate_limit=1000,
            documentation_url="https://openweathermap.org/api",
            tags=["weather", "climate", "forecast"]
        )
    
    async def fetch_data(
        self,
        location: Union[str, Tuple[float, float]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        forecast: bool = False,
        days: int = 7,
        hourly: bool = False
    ) -> pd.DataFrame:
        """
        Fetch weather data.
        
        Args:
            location: Location name or coordinates (latitude, longitude)
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            forecast: Whether to fetch forecast data
            days: Number of days for forecast
            hourly: Whether to include hourly data
            
        Returns:
            DataFrame of weather data
        """
        if "api_key" not in self.config:
            raise ValueError("API key required for weather API")
        
        # In a real implementation, this would use a weather API
        # For this example, we'll simulate the response
        
        # Parse location
        if isinstance(location, str):
            # Simulate geocoding
            latitude = 37.7749
            longitude = -122.4194
            location_name = location
        else:
            latitude, longitude = location
            location_name = f"{latitude}, {longitude}"
        
        # Set default dates if not provided
        if forecast:
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        else:
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Generate simulated weather data
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Determine time step
        if hourly:
            time_step = timedelta(hours=1)
            periods = int((end_dt - start_dt).total_seconds() / 3600) + 1
        else:
            time_step = timedelta(days=1)
            periods = (end_dt - start_dt).days + 1
        
        # Generate timestamps
        timestamps = [start_dt + time_step * i for i in range(periods)]
        
        # Generate weather data
        data = []
        
        for ts in timestamps:
            # Base temperature with seasonal variation
            day_of_year = ts.timetuple().tm_yday
            seasonal_factor = np.sin((day_of_year - 15) / 365 * 2 * np.pi)
            
            # Daily variation
            if hourly:
                hour_factor = np.sin((ts.hour - 3) / 24 * 2 * np.pi)
                temp_variation = 5 * hour_factor
            else:
                temp_variation = 0
            
            # Base temperature around 15°C with seasonal variation of ±10°C
            base_temp = 15 + 10 * seasonal_factor
            
            # Add random variation
            temp = base_temp + temp_variation + np.random.normal(0, 2)
            
            # Generate other weather parameters
            humidity = 50 + 20 * seasonal_factor + np.random.normal(0, 5)
            pressure = 1013 + np.random.normal(0, 3)
            wind_speed = 5 + np.random.exponential(3)
            wind_direction = np.random.uniform(0, 360)
            precipitation = max(0, np.random.exponential(1) - 0.5) if np.random.random() < 0.3 else 0
            
            # Weather condition
            if precipitation > 1:
                condition = "Rain"
            elif precipitation > 0:
                condition = "Light Rain"
            elif humidity > 80:
                condition = "Cloudy"
            elif humidity > 60:
                condition = "Partly Cloudy"
            else:
                condition = "Clear"
            
            data.append({
                "timestamp": ts,
                "location": location_name,
                "latitude": latitude,
                "longitude": longitude,
                "temperature": temp,
                "humidity": humidity,
                "pressure": pressure,
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "precipitation": precipitation,
                "condition": condition,
                "is_forecast": forecast
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df


class EconomicIndicatorsPlugin(DataSourcePlugin):
    """Plugin for fetching economic indicators."""
    
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return PluginMetadata(
            name="EconomicIndicators",
            version="1.0.0",
            description="Fetches economic indicators and financial data",
            author="AlphaMind Team",
            source_type=DataSourceType.ECONOMIC,
            frequency=DataFrequency.DAILY,
            output_format=DataFormat.DATAFRAME,
            requires_api_key=True,
            requires_authentication=False,
            free_tier_available=True,
            rate_limit=500,
            documentation_url="https://fred.stlouisfed.org/docs/api/fred/",
            tags=["economic", "indicators", "financial"]
        )
    
    async def fetch_data(
        self,
        indicators: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "monthly",
        countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch economic indicators.
        
        Args:
            indicators: List of indicator codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (daily, weekly, monthly, quarterly, annual)
            countries: List of country codes
            
        Returns:
            DataFrame of economic indicators
        """
        if "api_key" not in self.config:
            raise ValueError("API key required for economic data API")
        
        # In a real implementation, this would use an economic data API
        # For this example, we'll simulate the response
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            # Default to 5 years of data
            start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        
        # Set default countries if not provided
        if not countries:
            countries = ["US"]
        
        # Generate simulated economic data
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Determine time step based on frequency
        if frequency == "daily":
            time_step = timedelta(days=1)
            periods = (end_dt - start_dt).days + 1
        elif frequency == "weekly":
            time_step = timedelta(weeks=1)
            periods = (end_dt - start_dt).days // 7 + 1
        elif frequency == "monthly":
            # Approximate months as 30 days
            time_step = timedelta(days=30)
            periods = (end_dt - start_dt).days // 30 + 1
        elif frequency == "quarterly":
            # Approximate quarters as 90 days
            time_step = timedelta(days=90)
            periods = (end_dt - start_dt).days // 90 + 1
        elif frequency == "annual":
            # Approximate years as 365 days
            time_step = timedelta(days=365)
            periods = (end_dt - start_dt).days // 365 + 1
        else:
            raise ValueError(f"Invalid frequency: {frequency}")
        
        # Generate timestamps
        timestamps = [start_dt + time_step * i for i in range(periods)]
        
        # Define indicator properties
        indicator_props = {
            "GDP": {"mean": 3.0, "std": 0.5, "trend": 0.1},
            "CPI": {"mean": 2.0, "std": 0.3, "trend": 0.05},
            "UNEMPLOYMENT": {"mean": 5.0, "std": 0.5, "trend": -0.02},
            "INTEREST_RATE": {"mean": 2.0, "std": 0.2, "trend": 0.01},
            "RETAIL_SALES": {"mean": 4.0, "std": 1.0, "trend": 0.2},
            "INDUSTRIAL_PRODUCTION": {"mean": 2.5, "std": 0.8, "trend": 0.1},
            "HOUSING_STARTS": {"mean": 1500, "std": 100, "trend": 5},
            "CONSUMER_SENTIMENT": {"mean": 95, "std": 5, "trend": 0.1}
        }
        
        # Generate data for each country and indicator
        data = []
        
        for country in countries:
            for indicator in indicators:
                if indicator in indicator_props:
                    props = indicator_props[indicator]
                else:
                    # Default properties
                    props = {"mean": 0.0, "std": 1.0, "trend": 0.0}
                
                # Generate time series with trend and noise
                values = []
                for i, ts in enumerate(timestamps):
                    trend_component = props["trend"] * i
                    noise_component = np.random.normal(0, props["std"])
                    value = props["mean"] + trend_component + noise_component
                    values.append(value)
                
                # Add to data
                for ts, value in zip(timestamps, values):
                    data.append({
                        "timestamp": ts,
                        "country": country,
                        "indicator": indicator,
                        "value": value,
                        "frequency": frequency
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df


class WebTrafficAnalyticsPlugin(DataSourcePlugin):
    """Plugin for fetching web traffic analytics."""
    
    def _get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return PluginMetadata(
            name="WebTrafficAnalytics",
            version="1.0.0",
            description="Fetches web traffic analytics for specified domains",
            author="AlphaMind Team",
            source_type=DataSourceType.WEB_TRAFFIC,
            frequency=DataFrequency.DAILY,
            output_format=DataFormat.DATAFRAME,
            requires_api_key=True,
            requires_authentication=False,
            free_tier_available=True,
            rate_limit=100,
            documentation_url="https://www.similarweb.com/corp/developer/apis/",
            tags=["web traffic", "analytics", "digital"]
        )
    
    async def fetch_data(
        self,
        domains: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        granularity: str = "daily",
        country: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch web traffic analytics.
        
        Args:
            domains: List of domains
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            metrics: List of metrics to fetch
            granularity: Data granularity (daily, weekly, monthly)
            country: Country code
            
        Returns:
            DataFrame of web traffic analytics
        """
        if "api_key" not in self.config:
            raise ValueError("API key required for web traffic analytics API")
        
        # In a real implementation, this would use a web traffic analytics API
        # For this example, we'll simulate the response
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not start_date:
            # Default to 30 days of data
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Set default metrics if not provided
        if not metrics:
            metrics = ["visits", "unique_visitors", "page_views", "bounce_rate", "avg_visit_duration"]
        
        # Generate simulated web traffic data
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Determine time step based on granularity
        if granularity == "daily":
            time_step = timedelta(days=1)
            periods = (end_dt - start_dt).days + 1
        elif granularity == "weekly":
            time_step = timedelta(weeks=1)
            periods = (end_dt - start_dt).days // 7 + 1
        elif granularity == "monthly":
            # Approximate months as 30 days
            time_step = timedelta(days=30)
            periods = (end_dt - start_dt).days // 30 + 1
        else:
            raise ValueError(f"Invalid granularity: {granularity}")
        
        # Generate timestamps
        timestamps = [start_dt + time_step * i for i in range(periods)]
        
        # Generate data for each domain
        data = []
        
        for domain in domains:
            # Base metrics for domain (larger domains have more traffic)
            domain_size_factor = len(domain) / 10  # Simple heuristic
            base_visits = 10000 * domain_size_factor
            base_unique_visitors = 8000 * domain_size_factor
            base_page_views = 30000 * domain_size_factor
            base_bounce_rate = 0.3 + np.random.uniform(-0.1, 0.1)
            base_avg_duration = 180 + np.random.uniform(-30, 30)
            
            for ts in timestamps:
                # Day of week effect (weekends have less traffic)
                day_of_week = ts.weekday()
                weekend_factor = 0.7 if day_of_week >= 5 else 1.0
                
                # Time trend (slight growth over time)
                days_from_start = (ts - start_dt).days
                trend_factor = 1.0 + 0.001 * days_from_start
                
                # Random variation
                random_factor = np.random.normal(1.0, 0.05)
                
                # Calculate metrics
                visits = base_visits * weekend_factor * trend_factor * random_factor
                unique_visitors = base_unique_visitors * weekend_factor * trend_factor * random_factor
                page_views = base_page_views * weekend_factor * trend_factor * random_factor
                bounce_rate = base_bounce_rate + np.random.normal(0, 0.02)
                avg_visit_duration = base_avg_duration + np.random.normal(0, 10)
                
                # Ensure valid values
                bounce_rate = max(0.0, min(1.0, bounce_rate))
                avg_visit_duration = max(0, avg_visit_duration)
                
                # Create record
                record = {
                    "timestamp": ts,
                    "domain": domain,
                    "granularity": granularity
                }
                
                if "visits" in metrics:
                    record["visits"] = int(visits)
                
                if "unique_visitors" in metrics:
                    record["unique_visitors"] = int(unique_visitors)
                
                if "page_views" in metrics:
                    record["page_views"] = int(page_views)
                
                if "bounce_rate" in metrics:
                    record["bounce_rate"] = bounce_rate
                
                if "avg_visit_duration" in metrics:
                    record["avg_visit_duration"] = avg_visit_duration
                
                if country:
                    record["country"] = country
                
                data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df


# Example usage
async def example_usage():
    """Example of how to use the alternative data plugin architecture."""
    # Create plugin manager
    manager = PluginManager()
    
    # Discover available plugins
    plugins = await manager.discover_plugins()
    print(f"Discovered plugins: {plugins}")
    
    # Load news plugin
    news_config = {"api_key": "your_api_key_here"}
    await manager.load_plugin("NewsAPI", news_config)
    
    # Get news data
    news_data = await manager.get_plugin_data(
        "NewsAPI",
        query="artificial intelligence",
        from_date="2023-01-01",
        to_date="2023-01-31"
    )
    
    print(f"News data shape: {news_data.shape}")
    print(f"News data columns: {news_data.columns.tolist()}")
    
    # Load Twitter sentiment plugin
    twitter_config = {
        "api_key": "your_api_key_here",
        "api_secret": "your_api_secret_here",
        "access_token": "your_access_token_here",
        "access_token_secret": "your_access_token_secret_here"
    }
    await manager.load_plugin("TwitterSentiment", twitter_config)
    
    # Get Twitter sentiment data
    twitter_data = await manager.get_plugin_data(
        "TwitterSentiment",
        query="$AAPL",
        count=50
    )
    
    print(f"Twitter data shape: {twitter_data.shape}")
    print(f"Twitter data columns: {twitter_data.columns.tolist()}")
    
    # Unload plugins
    await manager.unload_plugin("NewsAPI")
    await manager.unload_plugin("TwitterSentiment")


if __name__ == "__main__":
    asyncio.run(example_usage())
