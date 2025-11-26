"""
Logging module for AlphaMind.

This module provides a consistent logging framework for the entire AlphaMind system,
including log formatting, rotation, and integration with monitoring systems.
"""

import datetime
import json
import logging
import logging.handlers
import os
import sys
import traceback
from typing import Any, Dict, Optional

# Default log format for standard text logging
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Ideal for production environments integrating with log management systems.
    """

    def __init__(self, include_stack_info: bool = False):
        """
        Initialize JSON formatter.

        Args:
            include_stack_info: Whether to include stack info in logs
        """
        super().__init__()
        self.include_stack_info = include_stack_info

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }

        # Include exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "N/A",
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Include stack info if requested
        if self.include_stack_info and record.stack_info:
            log_data["stack_info"] = record.stack_info

        # Include custom fields (e.g., from `extra` dict or `ContextFilter`)
        for key, value in record.__dict__.items():
            # Skip standard attributes already handled or internal
            if key in (
                "message",
                "asctime",
                "levelno",
                "levelname",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "process",
                "name",
                "msg",
                "args",
                "extra_fields",
                "processName",
            ):
                continue

            # Check for fields added by ContextFilter or logger.log(..., extra={...})
            if key not in log_data:
                # Safely include objects, skipping non-serializable ones
                try:
                    # Convert to string if it's not a basic type
                    log_data[key] = (
                        value
                        if isinstance(value, (str, int, float, bool, dict, list))
                        else str(value)
                    )
                except Exception:
                    log_data[key] = (
                        f"Non-serializable object of type {type(value).__name__}"
                    )

        # If a ContextFilter was used, merge its fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class LoggerManager:
    """Manages logging configuration, handlers, and logger instances for the AlphaMind system."""

    def __init__(self):
        """Initialize logger manager."""
        # The root logger for the entire application, named 'alphamind'
        self.root_logger = logging.getLogger("alphamind")
        self.handlers: Dict[str, logging.Handler] = {}
        self.loggers: Dict[str, logging.Logger] = {}

    def configure(
        self,
        level: int = logging.INFO,
        log_format: str = DEFAULT_LOG_FORMAT,
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_file_path: Optional[str] = None,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
        json_format: bool = False,
        include_stack_info: bool = False,
    ) -> None:
        """
        Configure logging for the root 'alphamind' logger.

        Args:
            level: Log level (e.g., logging.INFO)
            log_format: Log format string (ignored if json_format is True)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to a rotating file
            log_file_path: Path to log file (defaults to ./logs/alphamind.log)
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup log files to keep
            json_format: Whether to use the structured JSON format
            include_stack_info: Whether to include stack info in JSON logs
        """
        # Configure root logger
        self.root_logger.setLevel(level)

        # Remove existing handlers to allow reconfiguration
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        self.handlers.clear()

        # Create formatter
        if json_format:
            formatter = JsonFormatter(include_stack_info=include_stack_info)
        else:
            formatter = logging.Formatter(log_format)

        # Add console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.root_logger.addHandler(console_handler)
            self.handlers["console"] = console_handler

        # Add file handler (RotatingFileHandler for log rotation)
        if log_to_file:
            if not log_file_path:
                log_dir = os.path.join(os.getcwd(), "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file_path = os.path.join(log_dir, "alphamind.log")

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)
            self.root_logger.addHandler(file_handler)
            self.handlers["file"] = file_handler

        self.root_logger.info(
            f"Logging configured at level {logging.getLevelName(level)}. JSON format: {json_format}"
        )

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance prefixed under the main 'alphamind' namespace.
        E.g., get_logger("core.datamanager") returns 'alphamind.core.datamanager'.

        Args:
            name: Sub-module name (e.g., 'backtest' or 'api.connector')

        Returns:
            Logger instance
        """
        logger_name = f"alphamind.{name}" if name else "alphamind"
        logger = logging.getLogger(logger_name)
        # Loggers inherit settings from the root, so no need to explicitly add handlers
        self.loggers[name] = logger
        return logger

    def set_level(self, level: int, logger_name: Optional[str] = None) -> None:
        """
        Set log level for a specific logger or the root logger.

        Args:
            level: Log level
            logger_name: Logger name (sub-module name used in get_logger), or None for the root logger.
        """
        if logger_name:
            if logger_name in self.loggers:
                self.loggers[logger_name].setLevel(level)
        else:
            self.root_logger.setLevel(level)
            # Optionally update all existing sub-loggers
            for logger in self.loggers.values():
                logger.setLevel(level)

    def add_context_to_logs(self, context: Dict[str, Any]) -> logging.Filter:
        """
        Adds context information (e.g., 'session_id', 'user_id', 'strategy_name')
        to all subsequent log records by applying a filter to the root logger.

        Args:
            context: Context information to add (key-value pairs)

        Returns:
            The created log filter instance, which can be removed later.
        """

        class ContextFilter(logging.Filter):
            """A filter that injects a fixed set of context into the log record."""

            def filter(self, record):
                # Use a dedicated field to prevent collisions with built-in attributes
                record.extra_fields = getattr(record, "extra_fields", {})
                record.extra_fields.update(context)
                return True

        log_filter = ContextFilter()
        self.root_logger.addFilter(log_filter)
        self.root_logger.info(f"Added global context fields: {list(context.keys())}")
        return log_filter


# Create a global instance for convenient, centralized access
logger_manager = LoggerManager()

# Configure default logging upon module load: INFO level, console only, standard format
logger_manager.configure(level=logging.INFO, log_to_console=True, log_to_file=False)


# Helper function to get a logger
def get_logger(name: str = "") -> logging.Logger:
    """
    Convenience function to get a logger instance.

    Args:
        name: Logger name (e.g., 'data_ingestion')

    Returns:
        Logger instance (e.g., 'alphamind.data_ingestion')
    """
    return logger_manager.get_logger(name)
