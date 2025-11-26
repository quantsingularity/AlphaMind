"""
Configuration management module for AlphaMind.

This module provides utilities for loading, validating, and accessing
configuration settings throughout the AlphaMind system.
"""

from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import yaml


# Assuming this custom exception exists in a .exceptions file (as per the import line)
# For this implementation, we define a simple placeholder.
class ConfigurationException(Exception):
    """Custom exception raised for configuration errors."""

    def __init__(self, message: str, error_code: str, details: Dict[str, Any]):
        super().__init__(message)
        self.error_code = error_code
        self.details = details


# Configure logging
logger = logging.getLogger("AlphaMind.Config")
logger.setLevel(logging.INFO)


@dataclass
class ConfigItem:
    """Configuration item definition with built-in validation metadata."""

    key: str
    default_value: Any
    description: str = ""
    required: bool = False
    # Callable is a function that takes the value and returns a boolean
    validator: Optional[Callable[[Any], bool]] = None
    validator_message: str = "Invalid configuration value"

    def validate(self, value: Any) -> bool:
        """
        Validate the configuration value against requirements and custom validator.

        Args:
            value: Value to validate

        Returns:
            True if the value is valid, False otherwise
        """
        # 1. Check for required missing value
        if value is None and self.required:
            return False

        # 2. Check against custom validator only if value is present
        if self.validator and value is not None:
            # First, check if the value is of the expected type before running the validator
            # We don't explicitly check type here, relying on the validator logic itself.
            return self.validator(value)

        return True


class ConfigManager:
    """
    Manages configuration settings, supporting schema registration, multiple load sources, and validation.
    """

    def __init__(self):
        """Initialize configuration manager."""
        self.config: Dict[str, Any] = {}
        self.schema: Dict[str, ConfigItem] = {}
        # Tracks where configuration values were loaded from (for debugging)
        self.config_sources: List[str] = []

    def register_config_item(self, item: ConfigItem) -> None:
        """
        Register a configuration item and apply its default value.

        Args:
            item: Configuration item to register
        """
        if item.key in self.schema:
            logger.warning(f"Configuration item '{item.key}' redefined.")

        self.schema[item.key] = item

        # Set default value only if the key has not been loaded from a source yet
        if item.key not in self.config:
            self.config[item.key] = item.default_value

    def register_config_items(self, items: List[ConfigItem]) -> None:
        """
        Register multiple configuration items.

        Args:
            items: Configuration items to register
        """
        for item in items:
            self.register_config_item(item)

    def load_from_dict(self, config_dict: Dict[str, Any], source: str = "dict") -> None:
        """
        Load configuration from a dictionary. New values overwrite existing ones.

        Args:
            config_dict: Dictionary containing configuration values
            source: Source of the configuration
        """
        self.config.update(config_dict)
        self.config_sources.append(source)
        logger.info(f"Loaded configuration from {source}")

    def load_from_json(self, file_path: str) -> None:
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to the JSON file

        Raises:
            ConfigurationException: If the file cannot be loaded or parsed
        """
        try:
            with open(file_path, "r") as f:
                config_dict = json.load(f)

            self.load_from_dict(config_dict, source=file_path)
        except Exception as e:
            raise ConfigurationException(
                message=f"Failed to load configuration from {file_path}: {str(e)}",
                error_code="CONFIG_LOAD_ERROR",
                details={"file_path": file_path},
            )

    def load_from_yaml(self, file_path: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            file_path: Path to the YAML file

        Raises:
            ConfigurationException: If the file cannot be loaded or parsed
        """
        try:
            with open(file_path, "r") as f:
                # Use safe_load for security when loading YAML
                config_dict = yaml.safe_load(f)

            if config_dict is None:
                config_dict = {}  # Handle empty YAML files gracefully

            self.load_from_dict(config_dict, source=file_path)
        except Exception as e:
            raise ConfigurationException(
                message=f"Failed to load configuration from {file_path}: {str(e)}",
                error_code="CONFIG_LOAD_ERROR",
                details={"file_path": file_path},
            )

    def load_from_env(self, prefix: str = "ALPHAMIND_") -> None:
        """
        Load configuration from environment variables.
        Keys are converted to lowercase after removing the prefix.
        E.g., ALPHAMIND_API_KEY -> api_key

        Args:
            prefix: Prefix for environment variables
        """
        config_dict = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                config_dict[config_key] = value

        self.load_from_dict(config_dict, source="environment")

    def validate(self) -> List[str]:
        """
        Validate the configuration against the registered schema.

        Returns:
            List of validation error messages. Returns empty list if validation passes.
        """
        errors = []

        for key, item in self.schema.items():
            value = self.config.get(key)

            if not item.validate(value):
                if item.required and value is None:
                    errors.append(f"Required configuration '{key}' is missing")
                else:
                    errors.append(
                        f"Configuration '{key}': {item.validator_message} (Value: {value})"
                    )

        if errors:
            logger.error(f"Configuration validation failed with {len(errors)} errors.")
        else:
            logger.info("Configuration validated successfully against schema.")

        return errors

    def validate_or_raise(self) -> None:
        """
        Validate the configuration and raise an exception if invalid.

        Raises:
            ConfigurationException: If the configuration is invalid
        """
        errors = self.validate()

        if errors:
            raise ConfigurationException(
                message="Configuration validation failed",
                error_code="CONFIG_VALIDATION_ERROR",
                details={"errors": errors},
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key is not found (overrides schema default)

        Returns:
            Configuration value or provided default
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value. If a schema exists, the value is validated first.

        Args:
            key: Configuration key
            value: Configuration value

        Raises:
            ConfigurationException: If the value is invalid according to the schema
        """
        # Validate if schema exists for this key
        if key in self.schema:
            item = self.schema[key]
            if not item.validate(value):
                raise ConfigurationException(
                    message=f"Invalid configuration value for '{key}'",
                    error_code="CONFIG_INVALID_VALUE",
                    details={
                        "key": key,
                        "value": value,
                        "message": item.validator_message,
                    },
                )

        self.config[key] = value

    def get_all(self) -> Dict[str, Any]:
        """
        Get all current configuration values.

        Returns:
            Dictionary of all configuration values
        """
        return self.config.copy()

    def get_sources(self) -> List[str]:
        """
        Get the sources from which configuration has been loaded.

        Returns:
            List of configuration sources
        """
        return self.config_sources.copy()


# Create a global instance for convenient access across the system
config_manager = ConfigManager()


# --- Common Validators ---


def is_positive(value: Union[int, float]) -> bool:
    """Check if a value is positive (> 0)."""
    try:
        return value > 0
    except (TypeError, ValueError):
        return False


def is_non_negative(value: Union[int, float]) -> bool:
    """Check if a value is non-negative (>= 0)."""
    try:
        return value >= 0
    except (TypeError, ValueError):
        return False


def is_percentage(value: Union[int, float]) -> bool:
    """Check if a value is a percentage (0 <= value <= 100)."""
    try:
        return 0 <= value <= 100
    except (TypeError, ValueError):
        return False


def is_probability(value: Union[int, float]) -> bool:
    """Check if a value is a probability (0 <= value <= 1)."""
    try:
        return 0 <= value <= 1
    except (TypeError, ValueError):
        return False


def is_in_list(valid_values: List[Any]) -> Callable[[Any], bool]:
    """Create a validator that checks if a value is in a list (enum-like)."""

    def validator(value: Any) -> bool:
        return value in valid_values

    return validator


def is_url(value: str) -> bool:
    """Check if a value is a URL (simple check)."""
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def is_email(value: str) -> bool:
    """Check if a value is an email address (simple check)."""
    return isinstance(value, str) and "@" in value and "." in value.split("@")[-1]
