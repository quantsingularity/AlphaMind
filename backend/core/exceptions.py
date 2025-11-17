"""
Exception handling module for AlphaMind.

This module provides a comprehensive exception hierarchy and utilities
for consistent error handling throughout the AlphaMind system.
"""

import datetime
from enum import Enum
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""

    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    EXECUTION = "execution"
    DATA = "data"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class AlphaMindException(Exception):
    """Base exception class for all AlphaMind exceptions."""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Dict[str, Any] = None,
    ):
        """Initialize AlphaMindException.

        Args:
            message: Error message
            error_code: Error code
            category: Error category
            severity: Error severity
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.datetime.now()

        # Log the error based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{error_code}: {message}")
        elif severity == ErrorSeverity.ERROR:
            logger.error(f"{error_code}: {message}")
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"{error_code}: {message}")
        else:
            logger.info(f"{error_code}: {message}")

        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class ValidationException(AlphaMindException):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        details: Dict[str, Any] = None,
        field: Optional[str] = None,
    ):
        """Initialize ValidationException.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
            field: Field that failed validation
        """
        details = details or {}
        if field:
            details["field"] = field

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.WARNING,
            details=details,
        )


class ConfigurationException(AlphaMindException):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "CONFIG_ERROR",
        details: Dict[str, Any] = None,
        config_key: Optional[str] = None,
    ):
        """Initialize ConfigurationException.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
            config_key: Configuration key that caused the error
        """
        details = details or {}
        if config_key:
            details["config_key"] = config_key

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            details=details,
        )


class ConnectionException(AlphaMindException):
    """Exception raised for connection errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "CONNECTION_ERROR",
        details: Dict[str, Any] = None,
        service: Optional[str] = None,
    ):
        """Initialize ConnectionException.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
            service: Service that failed to connect
        """
        details = details or {}
        if service:
            details["service"] = service

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.ERROR,
            details=details,
        )


class ExecutionException(AlphaMindException):
    """Exception raised for execution errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "EXECUTION_ERROR",
        details: Dict[str, Any] = None,
        operation: Optional[str] = None,
    ):
        """Initialize ExecutionException.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
            operation: Operation that failed
        """
        details = details or {}
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            details=details,
        )


class DataException(AlphaMindException):
    """Exception raised for data errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "DATA_ERROR",
        details: Dict[str, Any] = None,
        data_source: Optional[str] = None,
    ):
        """Initialize DataException.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
            data_source: Data source that caused the error
        """
        details = details or {}
        if data_source:
            details["data_source"] = data_source

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.ERROR,
            details=details,
        )


class SystemException(AlphaMindException):
    """Exception raised for system errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "SYSTEM_ERROR",
        details: Dict[str, Any] = None,
    ):
        """Initialize SystemException.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
        """
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details=details,
        )


def handle_exception(
    exc: Exception,
    default_message: str = "An unexpected error occurred",
    log_traceback: bool = True,
) -> Dict[str, Any]:
    """Handle an exception and return a standardized error response.

    Args:
        exc: Exception to handle
        default_message: Default message if exception is not an AlphaMindException
        log_traceback: Whether to log the traceback

    Returns:
        Standardized error response dictionary
    """
    if log_traceback:
        logger.error("Exception traceback:", exc_info=True)

    if isinstance(exc, AlphaMindException):
        return exc.to_dict()
    else:
        # Wrap unknown exceptions in SystemException
        system_exc = SystemException(
            message=str(exc) or default_message,
            error_code="UNHANDLED_EXCEPTION",
            details={"exception_type": exc.__class__.__name__},
        )
        return system_exc.to_dict()


class ErrorCollector:
    """Collects and aggregates errors."""

    def __init__(self):
        """Initialize error collector."""
        self.errors: List[Dict[str, Any]] = []

    def add_error(self, error: AlphaMindException) -> None:
        """Add an error to the collector.

        Args:
            error: Error to add
        """
        self.errors.append(error.to_dict())

    def add_validation_error(self, message: str, field: Optional[str] = None) -> None:
        """Add a validation error to the collector.

        Args:
            message: Error message
            field: Field that failed validation
        """
        error = ValidationException(message=message, field=field)
        self.add_error(error)

    def has_errors(self) -> bool:
        """Check if the collector has any errors.

        Returns:
            True if the collector has errors, False otherwise
        """
        return len(self.errors) > 0

    def has_critical_errors(self) -> bool:
        """Check if the collector has any critical errors.

        Returns:
            True if the collector has critical errors, False otherwise
        """
        return any(e["severity"] == ErrorSeverity.CRITICAL.value for e in self.errors)

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all collected errors.

        Returns:
            List of error dictionaries
        """
        return self.errors.copy()

    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert collector to dictionary representation.

        Returns:
            Dictionary representation of the collector
        """
        return {
            "has_errors": self.has_errors(),
            "has_critical_errors": self.has_critical_errors(),
            "error_count": len(self.errors),
            "errors": self.get_errors(),
        }
