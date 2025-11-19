#""""""
## Alternative Data Plugin Architecture for AlphaMind
#
## This module provides a flexible plugin system for integrating alternative data sources
## such as news, social media, satellite imagery, and other non-traditional data
## into trading strategies and analysis with robust retry logic, validation, and logging.
#""""""

# from abc import ABC, abstractmethod
# import asyncio
# from datetime import datetime, timedelta
# from enum import Enum
# import importlib
# import inspect
# import json
# import logging
# import os
# import pkgutil
# import re
# import sys
# import time
# import traceback
# from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# from bs4 import BeautifulSoup
# import numpy as np
# import pandas as pd
# import requests

# Configure enhanced logging
# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
# logger = logging.getLogger(__name__)

# Add file handler for persistent logging
# try:
#     file_handler = logging.FileHandler("logs/alternative_data.log")
#     file_handler.setFormatter(
#         logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
    )
#     logger.addHandler(file_handler)
# except Exception as e:
#     logger.warning(f"Could not set up file logging: {e}")


# class DataValidationError(Exception):
#    """Exception raised for data validation errors."""
#
##     pass
#
#
## class PluginError(Exception):
#    """Exception raised for plugin-related errors."""

#     pass


# class DataSourceType(Enum):
#    """Types of alternative data sources."""
#
##     NEWS = "news"
##     SOCIAL_MEDIA = "social_media"
##     SATELLITE = "satellite"
##     WEATHER = "weather"
##     ECONOMIC = "economic"
##     SENTIMENT = "sentiment"
##     TRANSACTION = "transaction"
##     WEB_TRAFFIC = "web_traffic"
##     SUPPLY_CHAIN = "supply_chain"
##     SEC_FILINGS = "sec_filings"  # Added SEC filings as a specific type
##     CUSTOM = "custom"
#
#
## class DataFrequency(Enum):
#    """Data update frequency."""

#     REAL_TIME = "real_time"  # Continuous updates
#     INTRADAY = "intraday"  # Multiple times per day
#     DAILY = "daily"  # Once per day
#     WEEKLY = "weekly"  # Once per week
#     MONTHLY = "monthly"  # Once per month
#     QUARTERLY = "quarterly"  # Once per quarter
#     ANNUAL = "annual"  # Once per year
#     CUSTOM = "custom"  # Custom frequency


# class DataFormat(Enum):
#    """Data output format."""
#
##     JSON = "json"
##     CSV = "csv"
##     DATAFRAME = "dataframe"
##     TEXT = "text"
##     IMAGE = "image"
##     BINARY = "binary"
##     CUSTOM = "custom"
#
#
## class PluginStatus(Enum):
#    """Plugin status."""

#     ACTIVE = "active"
#     INACTIVE = "inactive"
#     ERROR = "error"
#     LOADING = "loading"
#     UNLOADED = "unloaded"
#     RETRYING = "retrying"  # Added status for retry operations


# class PluginMetadata:
#    """Metadata for data source plugins."""
#
##     def __init__(
##         self,
##         name: str,
##         version: str,
##         description: str,
##         author: str,
##         source_type: DataSourceType,
##         frequency: DataFrequency,
##         output_format: DataFormat,
##         requires_api_key: bool = False,
##         requires_authentication: bool = False,
##         free_tier_available: bool = True,
##         rate_limit: Optional[int] = None,
##         documentation_url: Optional[str] = None,
##         tags: Optional[List[str]] = None,
#    ):
#        """"""
#         Initialize plugin metadata.

#         Args:
#             name: Plugin name
#             version: Plugin version
#             description: Plugin description
#             author: Plugin author
#             source_type: Type of data source
#             frequency: Data update frequency
#             output_format: Data output format
#             requires_api_key: Whether plugin requires API key
#             requires_authentication: Whether plugin requires authentication
#             free_tier_available: Whether free tier is available
#             rate_limit: Rate limit (requests per minute)
#             documentation_url: URL to plugin documentation
#             tags: List of tags for categorization
#        """"""
#        # Validate inputs
##         if not name or not isinstance(name, str):
##             raise ValueError("Plugin name must be a non-empty string")
##         if not version or not isinstance(version, str):
##             raise ValueError("Plugin version must be a non-empty string")
##         if not description or not isinstance(description, str):
##             raise ValueError("Plugin description must be a non-empty string")
##         if not author or not isinstance(author, str):
##             raise ValueError("Plugin author must be a non-empty string")
##         if not isinstance(source_type, DataSourceType):
##             raise ValueError("Invalid source_type, must be a DataSourceType enum")
##         if not isinstance(frequency, DataFrequency):
##             raise ValueError("Invalid frequency, must be a DataFrequency enum")
##         if not isinstance(output_format, DataFormat):
##             raise ValueError("Invalid output_format, must be a DataFormat enum")
#
##         self.name = name
##         self.version = version
##         self.description = description
##         self.author = author
##         self.source_type = source_type
##         self.frequency = frequency
##         self.output_format = output_format
##         self.requires_api_key = requires_api_key
##         self.requires_authentication = requires_authentication
##         self.free_tier_available = free_tier_available
##         self.rate_limit = rate_limit
##         self.documentation_url = documentation_url
##         self.tags = tags or []
##         self.created_at = datetime.now()
##         self.updated_at = self.created_at
#
##         logger.debug(f"Created metadata for plugin: {name} v{version}")
#
##     def to_dict(self) -> Dict[str, Any]:
#        """"""
#         Convert metadata to dictionary.

#         Returns:
#             Dictionary representation of metadata
#        """"""
##         return {
#            "name": self.name,
#            "version": self.version,
#            "description": self.description,
#            "author": self.author,
#            "source_type": self.source_type.value,
#            "frequency": self.frequency.value,
#            "output_format": self.output_format.value,
#            "requires_api_key": self.requires_api_key,
#            "requires_authentication": self.requires_authentication,
#            "free_tier_available": self.free_tier_available,
#            "rate_limit": self.rate_limit,
#            "documentation_url": self.documentation_url,
#            "tags": self.tags,
#            "created_at": self.created_at.isoformat(),
#            "updated_at": self.updated_at.isoformat(),
#        }
#
##     @classmethod
##     def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
#        """"""
#         Create metadata from dictionary with validation.

#         Args:
#             data: Dictionary representation of metadata

#         Returns:
#             PluginMetadata instance

#         Raises:
#             ValueError: If required fields are missing or invalid
#        """"""
#        # Validate required fields
##         required_fields = [
#            "name",
#            "version",
#            "description",
#            "author",
#            "source_type",
#            "frequency",
#            "output_format",
#        ]
##         missing_fields = [field for field in required_fields if field not in data]
##         if missing_fields:
##             raise ValueError(
##                 f"Missing required fields in metadata: {', '.join(missing_fields)}"
#            )
#
##         try:
##             metadata = cls(
##                 name=data["name"],
##                 version=data["version"],
##                 description=data["description"],
##                 author=data["author"],
##                 source_type=DataSourceType(data["source_type"]),
##                 frequency=DataFrequency(data["frequency"]),
##                 output_format=DataFormat(data["output_format"]),
##                 requires_api_key=data.get("requires_api_key", False),
##                 requires_authentication=data.get("requires_authentication", False),
##                 free_tier_available=data.get("free_tier_available", True),
##                 rate_limit=data.get("rate_limit"),
##                 documentation_url=data.get("documentation_url"),
##                 tags=data.get("tags", []),
#            )
#
##             if "created_at" in data:
##                 metadata.created_at = datetime.fromisoformat(data["created_at"])
#
##             if "updated_at" in data:
##                 metadata.updated_at = datetime.fromisoformat(data["updated_at"])
#
##             return metadata
##         except (ValueError, KeyError) as e:
##             logger.error(f"Error creating metadata from dictionary: {e}")
##             raise ValueError(f"Invalid metadata format: {e}")
#
#
## class DataSourcePlugin(ABC):
#    """Base class for data source plugins."""

#     def __init__(self, config: Optional[Dict[str, Any]] = None):
#        """"""
##         Initialize data source plugin.
#
##         Args:
##             config: Plugin configuration
#        """"""
#         self.config = config or {}
#         self.metadata = self._get_metadata()
#         self.status = PluginStatus.INACTIVE
#         self.last_error = None
#         self.last_fetch_time = None
#         self.cache = {}
#         self.cache_expiry = {}
#         self.retry_settings = {
            "max_retries": 5,
            "backoff_factor": 1.5,
            "initial_wait": 1.0,
            "max_wait": 60.0,
        }
#         self.validation_settings = {
            "validate_data": True,
            "schema": {},  # Plugin-specific schema
            "required_fields": [],  # Plugin-specific required fields
        }
#         self.health_metrics = {
            "fetch_count": 0,
            "error_count": 0,
            "retry_count": 0,
            "validation_errors": 0,
            "last_success": None,
        }

#         logger.info(
#             f"Initialized plugin: {self.metadata.name} v{self.metadata.version}"
        )

#     @abstractmethod
#     def _get_metadata(self) -> PluginMetadata:
#        """"""
##         Get plugin metadata.
#
##         Returns:
##             Plugin metadata
#        """"""
#         pass

#     @abstractmethod
#     async def fetch_data(self, **kwargs) -> Any:
#        """"""
##         Fetch data from source.
#
##         Args:
##             **kwargs: Additional parameters for data fetching
#
##         Returns:
##             Fetched data
#        """"""
#         pass

#     async def initialize(self) -> bool:
#        """"""
##         Initialize plugin with enhanced error handling.
#
##         Returns:
##             True if initialization successful, False otherwise
#        """"""
#         try:
#             self.status = PluginStatus.LOADING
#             logger.info(f"Initializing plugin: {self.metadata.name}")

            # Update retry and validation settings from config
#             if "retry_settings" in self.config:
#                 for key, value in self.config["retry_settings"].items():
#                     if key in self.retry_settings:
#                         self.retry_settings[key] = value

#             if "validation_settings" in self.config:
#                 for key, value in self.config["validation_settings"].items():
#                     if key in self.validation_settings:
#                         self.validation_settings[key] = value

            # Validate configuration
#             if not self._validate_config():
#                 self.status = PluginStatus.ERROR
#                 self.last_error = "Invalid configuration"
#                 logger.error(
#                     f"Plugin {self.metadata.name} initialization failed: Invalid configuration"
                )
#                 return False

            # Perform any necessary setup
#             setup_success = await self._setup()
#             if not setup_success:
#                 self.status = PluginStatus.ERROR
#                 if not self.last_error:
#                     self.last_error = "Setup failed"
#                 logger.error(
#                     f"Plugin {self.metadata.name} setup failed: {self.last_error}"
                )
#                 return False

#             self.status = PluginStatus.ACTIVE
#             logger.info(f"Plugin {self.metadata.name} initialized successfully")
#             return True
#         except Exception as e:
#             self.status = PluginStatus.ERROR
#             self.last_error = str(e)
#             logger.error(f"Error initializing plugin {self.metadata.name}: {e}")
#             logger.debug(traceback.format_exc())
#             return False

#     async def shutdown(self) -> bool:
#        """"""
##         Shutdown plugin with enhanced error handling.
#
##         Returns:
##             True if shutdown successful, False otherwise
#        """"""
#         try:
#             logger.info(f"Shutting down plugin: {self.metadata.name}")

            # Perform any necessary cleanup
#             cleanup_success = await self._cleanup()
#             if not cleanup_success:
#                 logger.warning(
#                     f"Plugin {self.metadata.name} cleanup had issues: {self.last_error}"
                )
                # Continue with shutdown despite cleanup issues

#             self.status = PluginStatus.UNLOADED
#             logger.info(f"Plugin {self.metadata.name} shutdown successfully")
#             return True
#         except Exception as e:
#             self.status = PluginStatus.ERROR
#             self.last_error = str(e)
#             logger.error(f"Error shutting down plugin {self.metadata.name}: {e}")
#             logger.debug(traceback.format_exc())
#             return False

#     async def get_data(
#         self, use_cache: bool = True, cache_ttl: Optional[int] = None, **kwargs
#     ) -> Any:
#        """"""
##         Get data from source with enhanced caching, retry logic, and validation.
#
##         Args:
##             use_cache: Whether to use cached data if available
##             cache_ttl: Cache time-to-live in seconds (if None, use default)
##             **kwargs: Additional parameters for data fetching
#
##         Returns:
##             Fetched data
#
##         Raises:
##             PluginError: If plugin is not active or data fetching fails
##             DataValidationError: If data validation fails
#        """"""
#         if self.status != PluginStatus.ACTIVE:
#             raise PluginError(
#                 f"Plugin {self.metadata.name} is not active (status: {self.status.value})"
            )

        # Generate cache key from kwargs
#         cache_key = self._generate_cache_key(**kwargs)

        # Check cache if enabled
#         if use_cache and cache_key in self.cache:
#             if (
#                 cache_key in self.cache_expiry
#                 and datetime.now() < self.cache_expiry[cache_key]
            ):
#                 logger.debug(
#                     f"Using cached data for {self.metadata.name} (key: {cache_key[:30]}...)"
                )
#                 return self.cache[cache_key]
#             else:
#                 logger.debug(
#                     f"Cache expired for {self.metadata.name} (key: {cache_key[:30]}...)"
                )

        # Implement retry logic
#         retries = 0
#         max_retries = self.retry_settings["max_retries"]
#         backoff = self.retry_settings["initial_wait"]
#         max_wait = self.retry_settings["max_wait"]

#         while retries <= max_retries:
#             try:
                # Update status during retries
#                 if retries > 0:
#                     self.status = PluginStatus.RETRYING
#                     logger.info(
#                         f"Retry attempt {retries}/{max_retries} for {self.metadata.name}"
                    )
#                     self.health_metrics["retry_count"] += 1

                # Fetch data
#                 start_time = time.time()
#                 data = await self.fetch_data(**kwargs)
#                 elapsed = time.time() - start_time

                # Update metrics
#                 self.last_fetch_time = datetime.now()
#                 self.health_metrics["fetch_count"] += 1
#                 self.health_metrics["last_success"] = self.last_fetch_time

                # Reset status if we were retrying
#                 if self.status == PluginStatus.RETRYING:
#                     self.status = PluginStatus.ACTIVE

                # Validate data if enabled
#                 if self.validation_settings["validate_data"]:
#                     try:
#                         self._validate_data(data)
#                     except DataValidationError as e:
#                         logger.warning(
#                             f"Data validation error for {self.metadata.name}: {e}"
                        )
#                         self.health_metrics["validation_errors"] += 1
                        # Continue despite validation error, but log it

                # Update cache if enabled
#                 if use_cache:
#                     self.cache[cache_key] = data

                    # Set cache expiry
#                     if cache_ttl is not None:
#                         self.cache_expiry[cache_key] = datetime.now() + timedelta(
#                             seconds=cache_ttl
                        )
#                     else:
                        # Default cache TTL based on data frequency
#                         default_ttl = self._get_default_cache_ttl()
#                         self.cache_expiry[cache_key] = datetime.now() + timedelta(
#                             seconds=default_ttl
                        )

#                 logger.info(
#                     f"Successfully fetched data from {self.metadata.name} in {elapsed:.2f}s"
                )
#                 return data

#             except Exception as e:
#                 retries += 1
#                 self.health_metrics["error_count"] += 1
#                 self.last_error = str(e)

#                 if retries <= max_retries:
#                     wait_time = min(backoff, max_wait)
#                     logger.warning(
#                         f"Error fetching data from {self.metadata.name} (attempt {retries}/{max_retries+1}): {e}"
                    )
#                     logger.info(f"Retrying in {wait_time:.1f} seconds...")
#                     await asyncio.sleep(wait_time)
#                     backoff = min(
#                         backoff * self.retry_settings["backoff_factor"], max_wait
                    )
#                 else:
#                     logger.error(
#                         f"Failed to fetch data from {self.metadata.name} after {max_retries} retries: {e}"
                    )
#                     logger.debug(traceback.format_exc())
#                     raise PluginError(
#                         f"Failed to fetch data after {max_retries} retries: {e}"
                    )

#     def _validate_config(self) -> bool:
#        """"""
##         Validate plugin configuration with enhanced checks.
#
##         Returns:
##             True if configuration is valid, False otherwise
#        """"""
        # Check for required API key if needed
#         if self.metadata.requires_api_key:
#             if "api_key" not in self.config:
#                 logger.error(f"API key required for {self.metadata.name}")
#                 return False

            # Check if API key is valid (not empty)
#             if not self.config["api_key"]:
#                 logger.error(f"Empty API key provided for {self.metadata.name}")
#                 return False

        # Check for required authentication if needed
#         if self.metadata.requires_authentication:
#             if "username" not in self.config or "password" not in self.config:
#                 logger.error(f"Authentication required for {self.metadata.name}")
#                 return False

            # Check if credentials are valid (not empty)
#             if not self.config.get("username") or not self.config.get("password"):
#                 logger.error(f"Empty credentials provided for {self.metadata.name}")
#                 return False

        # Plugin-specific configuration validation
#         try:
#             return self._validate_plugin_config()
#         except Exception as e:
#             logger.error(f"Error in plugin-specific configuration validation: {e}")
#             return False

#     def _validate_plugin_config(self) -> bool:
#        """"""
##         Plugin-specific configuration validation.
##         Override in subclasses for custom validation.
#
##         Returns:
##             True if configuration is valid, False otherwise
#        """"""
#         return True

#     def _validate_data(self, data: Any) -> bool:
#        """"""
##         Validate data against schema.
#
##         Args:
##             data: Data to validate
#
##         Returns:
##             True if data is valid
#
##         Raises:
##             DataValidationError: If validation fails
#        """"""
        # Check if data is None
#         if data is None:
#             raise DataValidationError("Data is None")

        # Check required fields if specified
#         if self.validation_settings["required_fields"]:
#             if isinstance(data, dict):
#                 missing_fields = [
#                     field
#                     for field in self.validation_settings["required_fields"]
#                     if field not in data
                ]
#                 if missing_fields:
#                     raise DataValidationError(
#                         f"Missing required fields: {', '.join(missing_fields)}"
                    )
#             elif isinstance(data, pd.DataFrame):
#                 missing_columns = [
#                     field
#                     for field in self.validation_settings["required_fields"]
#                     if field not in data.columns
                ]
#                 if missing_columns:
#                     raise DataValidationError(
#                         f"Missing required columns: {', '.join(missing_columns)}"
                    )

        # Check schema if specified
#         if self.validation_settings["schema"]:
#             if isinstance(data, dict):
#                 for field, field_type in self.validation_settings["schema"].items():
#                     if field in data and not isinstance(data[field], field_type):
#                         raise DataValidationError(
#                             f"Field {field} has invalid type: {type(data[field])}, expected {field_type}"
                        )

        # Plugin-specific data validation
#         try:
#             return self._validate_plugin_data(data)
#         except Exception as e:
#             raise DataValidationError(f"Plugin-specific validation error: {e}")

#     def _validate_plugin_data(self, data: Any) -> bool:
#        """"""
##         Plugin-specific data validation.
##         Override in subclasses for custom validation.
#
##         Args:
##             data: Data to validate
#
##         Returns:
##             True if data is valid
#
##         Raises:
##             DataValidationError: If validation fails
#        """"""
#         return True

#     async def _setup(self) -> bool:
#        """"""
##         Perform plugin setup with error handling.
##         Override in subclasses for custom setup.
#
##         Returns:
##             True if setup successful, False otherwise
#        """"""
#         return True

#     async def _cleanup(self) -> bool:
#        """"""
##         Perform plugin cleanup with error handling.
##         Override in subclasses for custom cleanup.
#
##         Returns:
##             True if cleanup successful, False otherwise
#        """"""
#         return True

#     def _generate_cache_key(self, **kwargs) -> str:
#        """"""
##         Generate cache key from parameters.
#
##         Args:
##             **kwargs: Parameters for data fetching
#
##         Returns:
##             Cache key
#        """"""
        # Sort kwargs by key to ensure consistent cache keys
#         sorted_items = sorted(kwargs.items())

        # Convert to string
#         key_parts = [f"{k}={v}" for k, v in sorted_items]
#         key_str = ",".join(key_parts)

        # Add plugin name and version to make keys unique across plugins
#         return f"{self.metadata.name}_v{self.metadata.version}:{key_str}"

#     def _get_default_cache_ttl(self) -> int:
#        """"""
##         Get default cache TTL based on data frequency.
#
##         Returns:
##             Cache TTL in seconds
#        """"""
#         if self.metadata.frequency == DataFrequency.REAL_TIME:
#             return 60  # 1 minute
#         elif self.metadata.frequency == DataFrequency.INTRADAY:
#             return 300  # 5 minutes
#         elif self.metadata.frequency == DataFrequency.DAILY:
#             return 86400  # 1 day
#         elif self.metadata.frequency == DataFrequency.WEEKLY:
#             return 604800  # 1 week
#         elif self.metadata.frequency == DataFrequency.MONTHLY:
#             return 2592000  # 30 days
#         elif self.metadata.frequency == DataFrequency.QUARTERLY:
#             return 7776000  # 90 days
#         elif self.metadata.frequency == DataFrequency.ANNUAL:
#             return 31536000  # 365 days
#         else:
#             return 3600  # 1 hour default

#     def get_health_status(self) -> Dict[str, Any]:
#        """"""
##         Get health status of plugin.
#
##         Returns:
##             Health status dictionary
#        """"""
#         status = {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "status": self.status.value,
            "last_error": self.last_error,
            "last_fetch_time": (
#                 self.last_fetch_time.isoformat() if self.last_fetch_time else None
            ),
            "cache_size": len(self.cache),
            "health_metrics": self.health_metrics,
            "timestamp": datetime.now().isoformat(),
        }

#         return status


# class PluginManager:
#    """Manager for data source plugins with enhanced error handling and monitoring."""
#
##     def __init__(self, plugin_dir: Optional[str] = None):
#        """"""
#         Initialize plugin manager.

#         Args:
#             plugin_dir: Directory containing plugins
#        """"""
##         self.plugin_dir = plugin_dir
##         self.plugins: Dict[str, DataSourcePlugin] = {}
##         self.plugin_classes: Dict[str, Type[DataSourcePlugin]] = {}
##         self.health_status = {
#            "active_plugins": 0,
#            "error_plugins": 0,
#            "total_plugins": 0,
#            "last_discovery": None,
#        }
#
##         logger.info(
##             f"Initialized plugin manager"
##             + (f" with plugin directory: {plugin_dir}" if plugin_dir else "")
#        )
#
##     async def discover_plugins(self) -> List[str]:
#        """"""
#         Discover available plugins with enhanced error handling.

#         Returns:
#             List of discovered plugin names
#        """"""
##         discovered = []
##         errors = []
#
##         try:
#            # Discover built-in plugins
##             builtin_plugins = await self._discover_builtin_plugins()
##             discovered.extend(builtin_plugins)
##             logger.info(f"Discovered {len(builtin_plugins)} built-in plugins")
#
#            # Discover plugins from directory if specified
##             if self.plugin_dir:
##                 if not os.path.exists(self.plugin_dir):
##                     logger.warning(
##                         f"Plugin directory does not exist: {self.plugin_dir}"
#                    )
##                 else:
##                     directory_plugins = await self._discover_directory_plugins()
##                     discovered.extend(directory_plugins)
##                     logger.info(
##                         f"Discovered {len(directory_plugins)} plugins from directory: {self.plugin_dir}"
#                    )
#
#            # Update health status
##             self.health_status["total_plugins"] = len(self.plugin_classes)
##             self.health_status["last_discovery"] = datetime.now()
#
##             return discovered
#
##         except Exception as e:
##             logger.error(f"Error discovering plugins: {e}")
##             logger.debug(traceback.format_exc())
##             return discovered
#
##     async def _discover_builtin_plugins(self) -> List[str]:
#        """"""
#         Discover built-in plugins with error handling.

#         Returns:
#             List of discovered plugin names
#        """"""
##         discovered = []
#
#        # This would be implemented to discover built-in plugins
#        # For now, just return an empty list
#
##         return discovered
#
##     async def _discover_directory_plugins(self) -> List[str]:
#        """"""
#         Discover plugins from directory with error handling.

#         Returns:
#             List of discovered plugin names
#        """"""
##         discovered = []
#
##         if not self.plugin_dir or not os.path.exists(self.plugin_dir):
##             return discovered
#
##         try:
#            # Add plugin directory to path
##             if self.plugin_dir not in sys.path:
##                 sys.path.insert(0, self.plugin_dir)
#
#            # Scan for Python files
##             for file in os.listdir(self.plugin_dir):
##                 if file.endswith(".py") and not file.startswith("__"):
##                     module_name = file[:-3]  # Remove .py extension
#
##                     try:
#                        # Import module
##                         module = importlib.import_module(module_name)
#
#                        # Find plugin classes
##                         for name, obj in inspect.getmembers(module):
##                             if (
##                                 inspect.isclass(obj)
##                                 and issubclass(obj, DataSourcePlugin)
##                                 and obj != DataSourcePlugin
#                            ):
#
#                                # Create temporary instance to get metadata
##                                 try:
##                                     temp_instance = obj({})
##                                     metadata = temp_instance._get_metadata()
#
#                                    # Register plugin class
##                                     self.plugin_classes[metadata.name] = obj
##                                     discovered.append(metadata.name)
#
##                                     logger.info(
##                                         f"Discovered plugin: {metadata.name} v{metadata.version} ({module_name}.{name})"
#                                    )
##                                 except Exception as e:
##                                     logger.error(
##                                         f"Error initializing plugin class {module_name}.{name}: {e}"
#                                    )
#
##                     except Exception as e:
##                         logger.error(f"Error importing module {module_name}: {e}")
#
##             return discovered
#
##         except Exception as e:
##             logger.error(f"Error discovering plugins from directory: {e}")
##             return discovered
#
##     async def load_plugin(
##         self, plugin_name: str, config: Optional[Dict[str, Any]] = None
##     ) -> bool:
#        """"""
#         Load and initialize plugin with enhanced error handling.

#         Args:
#             plugin_name: Plugin name
#             config: Plugin configuration

#         Returns:
#             True if plugin loaded successfully, False otherwise
#        """"""
##         if plugin_name in self.plugins:
##             logger.warning(f"Plugin {plugin_name} already loaded")
##             return True
#
##         try:
#            # Create plugin instance
##             if plugin_name in self.plugin_classes:
##                 logger.info(f"Loading plugin: {plugin_name}")
##                 plugin_class = self.plugin_classes[plugin_name]
##                 plugin = plugin_class(config or {})
#
#                # Initialize plugin
##                 if await plugin.initialize():
##                     self.plugins[plugin_name] = plugin
#
#                    # Update health status
##                     self.health_status["active_plugins"] = sum(
#                        1
##                         for p in self.plugins.values()
##                         if p.status == PluginStatus.ACTIVE
#                    )
##                     self.health_status["error_plugins"] = sum(
#                        1
##                         for p in self.plugins.values()
##                         if p.status == PluginStatus.ERROR
#                    )
#
##                     logger.info(f"Successfully loaded plugin: {plugin_name}")
##                     return True
##                 else:
##                     logger.error(f"Failed to initialize plugin: {plugin_name}")
#
#                    # Update health status
##                     self.health_status["error_plugins"] += 1
#
##                     return False
##             else:
##                 logger.error(f"Plugin not found: {plugin_name}")
##                 return False
##         except Exception as e:
##             logger.error(f"Error loading plugin {plugin_name}: {e}")
##             logger.debug(traceback.format_exc())
##             return False
#
##     async def unload_plugin(self, plugin_name: str) -> bool:
#        """"""
#         Unload plugin with enhanced error handling.

#         Args:
#             plugin_name: Plugin name

#         Returns:
#             True if plugin unloaded successfully, False otherwise
#        """"""
##         if plugin_name not in self.plugins:
##             logger.warning(f"Plugin {plugin_name} not loaded")
##             return True
#
##         try:
##             plugin = self.plugins[plugin_name]
##             logger.info(f"Unloading plugin: {plugin_name}")
#
#            # Shutdown plugin
##             if await plugin.shutdown():
##                 del self.plugins[plugin_name]
#
#                # Update health status
##                 self.health_status["active_plugins"] = sum(
##                     1 for p in self.plugins.values() if p.status == PluginStatus.ACTIVE
#                )
##                 self.health_status["error_plugins"] = sum(
##                     1 for p in self.plugins.values() if p.status == PluginStatus.ERROR
#                )
#
##                 logger.info(f"Successfully unloaded plugin: {plugin_name}")
##                 return True
##             else:
##                 logger.error(f"Failed to shutdown plugin: {plugin_name}")
##                 return False
##         except Exception as e:
##             logger.error(f"Error unloading plugin {plugin_name}: {e}")
##             logger.debug(traceback.format_exc())
##             return False
#
##     async def get_data(self, plugin_name: str, **kwargs) -> Any:
#        """"""
#         Get data from plugin with enhanced error handling.

#         Args:
#             plugin_name: Plugin name
#             **kwargs: Additional parameters for data fetching

#         Returns:
#             Fetched data

#         Raises:
#             PluginError: If plugin is not loaded or data fetching fails
#        """"""
##         if plugin_name not in self.plugins:
##             raise PluginError(f"Plugin {plugin_name} not loaded")
#
##         plugin = self.plugins[plugin_name]
#
##         try:
##             return await plugin.get_data(**kwargs)
##         except Exception as e:
##             logger.error(f"Error getting data from plugin {plugin_name}: {e}")
##             raise
#
##     def get_plugin_metadata(self, plugin_name: str) -> Optional[Dict[str, Any]]:
#        """"""
#         Get plugin metadata.

#         Args:
#             plugin_name: Plugin name

#         Returns:
#             Plugin metadata as dictionary, or None if plugin not found
#        """"""
##         if plugin_name in self.plugins:
##             return self.plugins[plugin_name].metadata.to_dict()
##         elif plugin_name in self.plugin_classes:
##             try:
##                 temp_instance = self.plugin_classes[plugin_name]({})
##                 return temp_instance._get_metadata().to_dict()
##             except Exception as e:
##                 logger.error(f"Error getting metadata for plugin {plugin_name}: {e}")
##                 return None
##         else:
##             return None
#
##     def get_all_plugin_metadata(self) -> Dict[str, Dict[str, Any]]:
#        """"""
#         Get metadata for all discovered plugins.

#         Returns:
#             Dictionary of plugin name to metadata
#        """"""
##         result = {}
#
#        # Get metadata for loaded plugins
##         for name, plugin in self.plugins.items():
##             result[name] = plugin.metadata.to_dict()
#
#        # Get metadata for discovered but not loaded plugins
##         for name, plugin_class in self.plugin_classes.items():
##             if name not in result:
##                 try:
##                     temp_instance = plugin_class({})
##                     result[name] = temp_instance._get_metadata().to_dict()
##                 except Exception as e:
##                     logger.error(f"Error getting metadata for plugin {name}: {e}")
#
##         return result
#
##     def get_health_status(self) -> Dict[str, Any]:
#        """"""
#         Get health status of plugin manager and all plugins.

#         Returns:
#             Health status dictionary
#        """"""
##         status = self.health_status.copy()
##         status["timestamp"] = datetime.now().isoformat()
#
#        # Add plugin statuses
##         status["plugins"] = {}
##         for name, plugin in self.plugins.items():
##             status["plugins"][name] = {
#                "status": plugin.status.value,
#                "last_error": plugin.last_error,
#                "last_fetch_time": (
##                     plugin.last_fetch_time.isoformat()
##                     if plugin.last_fetch_time
##                     else None
#                ),
#                "health_metrics": plugin.health_metrics,
#            }
#
##         return status
#
##     async def reload_plugin(self, plugin_name: str) -> bool:
#        """"""
#         Reload plugin with enhanced error handling.

#         Args:
#             plugin_name: Plugin name

#         Returns:
#             True if plugin reloaded successfully, False otherwise
#        """"""
##         if plugin_name not in self.plugins:
##             logger.warning(f"Plugin {plugin_name} not loaded, cannot reload")
##             return False
#
##         try:
#            # Get current config
##             config = self.plugins[plugin_name].config
#
#            # Unload plugin
##             if not await self.unload_plugin(plugin_name):
##                 logger.error(f"Failed to unload plugin {plugin_name} for reload")
##                 return False
#
#            # Load plugin with same config
##             if await self.load_plugin(plugin_name, config):
##                 logger.info(f"Successfully reloaded plugin: {plugin_name}")
##                 return True
##             else:
##                 logger.error(f"Failed to reload plugin: {plugin_name}")
##                 return False
##         except Exception as e:
##             logger.error(f"Error reloading plugin {plugin_name}: {e}")
##             logger.debug(traceback.format_exc())
##             return False
#
#
## Example SEC filings plugin implementation
## class SECFilingsPlugin(DataSourcePlugin):
#    """Plugin for SEC filings data."""

#     def _get_metadata(self) -> PluginMetadata:
#        """Get plugin metadata."""
##         return PluginMetadata(
##             name="sec_filings",
##             version="1.0.0",
##             description="SEC filings data provider",
##             author="AlphaMind",
##             source_type=DataSourceType.SEC_FILINGS,
##             frequency=DataFrequency.DAILY,
##             output_format=DataFormat.JSON,
##             requires_api_key=False,
##             tags=["sec", "filings", "8-k", "10-k", "10-q"],
#        )
#
##     async def _setup(self) -> bool:
#        """Set up SEC filings plugin."""
#         try:
            # Set up validation settings
#             self.validation_settings["required_fields"] = [
                "filing_type",
                "filing_date",
                "company_name",
                "ticker",
            ]
#             self.validation_settings["schema"] = {
                "filing_type": str,
                "filing_date": str,
                "company_name": str,
                "ticker": str,
                "cik": str,
                "url": str,
            }

            # Import required libraries
#             try:
#                 from sec_edgar_downloader import Downloader

#                 self.downloader = Downloader("alphamind_sec_plugin")
#                 logger.info("SEC Edgar downloader initialized")
#             except ImportError:
#                 logger.error("sec_edgar_downloader library not installed")
#                 self.last_error = "Required library not installed: sec_edgar_downloader"
#                 return False

#             return True
#         except Exception as e:
#             self.last_error = str(e)
#             logger.error(f"Error setting up SEC filings plugin: {e}")
#             return False

#     async def fetch_data(
#         self, ticker: str = None, filing_type: str = "8-K", count: int = 10, **kwargs
#     ) -> List[Dict[str, Any]]:
#        """"""
##         Fetch SEC filings data.
#
##         Args:
##             ticker: Company ticker symbol
##             filing_type: SEC filing type (8-K, 10-K, 10-Q, etc.)
##             count: Number of filings to fetch
#
##         Returns:
##             List of filing data
#        """"""
#         if not ticker:
#             raise ValueError("Ticker is required")

        # Validate filing type
#         valid_filing_types = [
            "8-K",
            "10-K",
            "10-Q",
            "13F",
            "SC 13G",
            "SC 13D",
            "4",
            "S-1",
        ]
#         if filing_type not in valid_filing_types:
#             raise ValueError(
#                 f"Invalid filing type: {filing_type}. Must be one of {valid_filing_types}"
            )

        # Download filings
#         logger.info(f"Downloading {count} {filing_type} filings for {ticker}")

        # This would actually download and process filings
        # For demonstration, return mock data
#         mock_filings = []
#         for i in range(count):
#             filing_date = (datetime.now() - timedelta(days=i * 7)).strftime("%Y-%m-%d")
#             mock_filings.append(
                {
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                    "company_name": f"{ticker} Corporation",
                    "ticker": ticker,
                    "cik": f"000{i}12345",
                    "url": f"https://www.sec.gov/Archives/edgar/data/{i}12345/{filing_type.replace(' ', '')}-{filing_date}.html",
                    "content_summary": f"Sample {filing_type} filing content for {ticker}",
                }
            )

#         logger.info(f"Retrieved {len(mock_filings)} {filing_type} filings for {ticker}")
#         return mock_filings
