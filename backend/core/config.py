# """"""
## Configuration management module for AlphaMind.
#
## This module provides utilities for loading, validating, and accessing
## configuration settings throughout the AlphaMind system.
# """"""

# from dataclasses import dataclass, field
# import json
# import logging
# import os
# from typing import Any, Callable, Dict, List, Optional, Union

# import yaml

# from .exceptions import ConfigurationException

# Configure logging
# logger = logging.getLogger(__name__)


# @dataclass
# class ConfigItem:
#    """Configuration item with validation."""
#
##     key: str
##     default_value: Any
##     description: str = ""
##     required: bool = False
##     validator: Optional[Callable[[Any], bool]] = None
##     validator_message: str = "Invalid configuration value"
#
##     def validate(self, value: Any) -> bool:
#        """Validate the configuration value."""

#         Args:
#             value: Value to validate

#         Returns:
#             True if the value is valid, False otherwise
#        """"""
##         if value is None and self.required:
##             return False
#
##         if self.validator and value is not None:
##             return self.validator(value)
#
##         return True
#
#
## class ConfigManager:
#    """Manages configuration settings."""

#     def __init__(self):
#        """Initialize configuration manager."""
##         self.config: Dict[str, Any] = {}
##         self.schema: Dict[str, ConfigItem] = {}
##         self.config_sources: List[str] = []
#
##     def register_config_item(self, item: ConfigItem) -> None:
#        """Register a configuration item."""

#         Args:
#             item: Configuration item to register
#        """"""
##         self.schema[item.key] = item
#
#        # Set default value if not already set
##         if item.key not in self.config:
##             self.config[item.key] = item.default_value
#
##     def register_config_items(self, items: List[ConfigItem]) -> None:
#        """Register multiple configuration items."""

#         Args:
#             items: Configuration items to register
#        """"""
##         for item in items:
##             self.register_config_item(item)
#
##     def load_from_dict(self, config_dict: Dict[str, Any], source: str = "dict") -> None:
#        """Load configuration from a dictionary."""

#         Args:
#             config_dict: Dictionary containing configuration values
#             source: Source of the configuration
#        """"""
##         self.config.update(config_dict)
##         self.config_sources.append(source)
##         logger.info(f"Loaded configuration from {source}")
#
##     def load_from_json(self, file_path: str) -> None:
#        """Load configuration from a JSON file."""

#         Args:
#             file_path: Path to the JSON file

#         Raises:
#             ConfigurationException: If the file cannot be loaded
#        """"""
##         try:
##             with open(file_path, "r") as f:
##                 config_dict = json.load(f)
#
##             self.load_from_dict(config_dict, source=file_path)
##         except Exception as e:
##             raise ConfigurationException(
##                 message=f"Failed to load configuration from {file_path}: {str(e)}",
##                 error_code="CONFIG_LOAD_ERROR",
##                 details={"file_path": file_path},
#            )
#
##     def load_from_yaml(self, file_path: str) -> None:
#        """Load configuration from a YAML file."""

#         Args:
#             file_path: Path to the YAML file

#         Raises:
#             ConfigurationException: If the file cannot be loaded
#        """"""
##         try:
##             with open(file_path, "r") as f:
##                 config_dict = yaml.safe_load(f)
#
##             self.load_from_dict(config_dict, source=file_path)
##         except Exception as e:
##             raise ConfigurationException(
##                 message=f"Failed to load configuration from {file_path}: {str(e)}",
##                 error_code="CONFIG_LOAD_ERROR",
##                 details={"file_path": file_path},
#            )
#
##     def load_from_env(self, prefix: str = "ALPHAMIND_") -> None:
#        """Load configuration from environment variables."""

#         Args:
#             prefix: Prefix for environment variables
#        """"""
##         config_dict = {}
#
##         for key, value in os.environ.items():
##             if key.startswith(prefix):
##                 config_key = key[len(prefix) :].lower()
##                 config_dict[config_key] = value
#
##         self.load_from_dict(config_dict, source="environment")
#
##     def validate(self) -> List[str]:
#        """Validate the configuration."""

#         Returns:
#             List of validation error messages
#        """"""
##         errors = []
#
##         for key, item in self.schema.items():
##             value = self.config.get(key)
#
##             if not item.validate(value):
##                 if item.required and value is None:
##                     errors.append(f"Required configuration '{key}' is missing")
##                 else:
##                     errors.append(f"Configuration '{key}': {item.validator_message}")
#
##         return errors
#
##     def validate_or_raise(self) -> None:
#        """Validate the configuration and raise an exception if invalid."""

#         Raises:
#             ConfigurationException: If the configuration is invalid
#        """"""
##         errors = self.validate()
#
##         if errors:
##             raise ConfigurationException(
##                 message="Configuration validation failed",
##                 error_code="CONFIG_VALIDATION_ERROR",
##                 details={"errors": errors},
#            )
#
##     def get(self, key: str, default: Any = None) -> Any:
#        """Get a configuration value."""

#         Args:
#             key: Configuration key
#             default: Default value if key is not found

#         Returns:
#             Configuration value or default
#        """"""
##         return self.config.get(key, default)
#
##     def set(self, key: str, value: Any) -> None:
#        """Set a configuration value."""

#         Args:
#             key: Configuration key
#             value: Configuration value

#         Raises:
#             ConfigurationException: If the value is invalid
#        """"""
#        # Validate if schema exists for this key
##         if key in self.schema:
##             item = self.schema[key]
##             if not item.validate(value):
##                 raise ConfigurationException(
##                     message=f"Invalid configuration value for '{key}'",
##                     error_code="CONFIG_INVALID_VALUE",
##                     details={
#                        "key": key,
#                        "value": value,
#                        "message": item.validator_message,
#                    },
#                )
#
##         self.config[key] = value
#
##     def get_all(self) -> Dict[str, Any]:
#        """Get all configuration values."""

#         Returns:
#             Dictionary of all configuration values
#        """"""
##         return self.config.copy()
#
##     def get_sources(self) -> List[str]:
#        """Get the sources of the configuration."""

#         Returns:
#             List of configuration sources
#        """"""
##         return self.config_sources.copy()
#
#
## Create a global instance
## config_manager = ConfigManager()
#
#
## Common validators
## def is_positive(value: Union[int, float]) -> bool:
#    """Check if a value is positive."""

#     Args:
#         value: Value to check

#     Returns:
#         True if the value is positive, False otherwise
#    """"""
##     return value > 0
#
#
## def is_non_negative(value: Union[int, float]) -> bool:
#    """Check if a value is non-negative."""

#     Args:
#         value: Value to check

#     Returns:
#         True if the value is non-negative, False otherwise
#    """"""
##     return value >= 0
#
#
## def is_percentage(value: Union[int, float]) -> bool:
#    """Check if a value is a percentage (0-100)."""

#     Args:
#         value: Value to check

#     Returns:
#         True if the value is a percentage, False otherwise
#    """"""
##     return 0 <= value <= 100
#
#
## def is_probability(value: Union[int, float]) -> bool:
#    """Check if a value is a probability (0-1)."""

#     Args:
#         value: Value to check

#     Returns:
#         True if the value is a probability, False otherwise
#    """"""
##     return 0 <= value <= 1
#
#
## def is_in_list(valid_values: List[Any]) -> Callable[[Any], bool]:
#    """Create a validator that checks if a value is in a list."""

#     Args:
#         valid_values: List of valid values

#     Returns:
#         Validator function
#    """"""
#
##     def validator(value: Any) -> bool:
##         return value in valid_values
#
##     return validator
#
#
## def is_url(value: str) -> bool:
#    """Check if a value is a URL."""

#     Args:
#         value: Value to check

#     Returns:
#         True if the value is a URL, False otherwise
#    """"""
##     return value.startswith(("http://", "https://"))
#
#
## def is_email(value: str) -> bool:
#    """Check if a value is an email address."""

#     Args:
#         value: Value to check

#     Returns:
#         True if the value is an email address, False otherwise
#    """"""
##     return "@" in value and "." in value.split("@")[1]
