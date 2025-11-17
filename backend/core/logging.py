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
from typing import Any, Callable, Dict, List, Optional, Union

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_stack_info: bool = False):
        """Initialize JSON formatter.

        Args:
            include_stack_info: Whether to include stack info in logs
        """
        super().__init__()
        self.include_stack_info = include_stack_info

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

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
        }

        # Include exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Include stack info if requested
        if self.include_stack_info and record.stack_info:
            log_data["stack_info"] = record.stack_info

        # Include custom fields
        for key, value in getattr(record, "extra_fields", {}).items():
            if key not in log_data:
                log_data[key] = value

        return json.dumps(log_data)


class LoggerManager:
    """Manages logging configuration for the AlphaMind system."""

    def __init__(self):
        """Initialize logger manager."""
        self.root_logger = logging.getLogger("alphamind")
        self.handlers = {}
        self.loggers = {}

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
        """Configure logging.

        Args:
            level: Log level
            log_format: Log format string
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_file_path: Path to log file
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup log files to keep
            json_format: Whether to use JSON format
            include_stack_info: Whether to include stack info in JSON logs
        """
        # Configure root logger
        self.root_logger.setLevel(level)

        # Remove existing handlers
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)

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

        # Add file handler
        if log_to_file:
            if not log_file_path:
                log_dir = os.path.join(os.getcwd(), "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file_path = os.path.join(log_dir, "alphamind.log")

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)
            self.root_logger.addHandler(file_handler)
            self.handlers["file"] = file_handler

        self.root_logger.info(
            f"Logging configured at level {logging.getLevelName(level)}"
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        logger_name = f"alphamind.{name}" if name else "alphamind"
        logger = logging.getLogger(logger_name)
        self.loggers[name] = logger
        return logger

    def set_level(self, level: int, logger_name: Optional[str] = None) -> None:
        """Set log level for a specific logger or all loggers.

        Args:
            level: Log level
            logger_name: Logger name, or None for all loggers
        """
        if logger_name:
            if logger_name in self.loggers:
                self.loggers[logger_name].setLevel(level)
        else:
            self.root_logger.setLevel(level)
            for logger in self.loggers.values():
                logger.setLevel(level)

    def add_context_to_logs(self, context: Dict[str, Any]) -> logging.Filter:
        """Add context information to all logs.

        Args:
            context: Context information to add

        Returns:
            Log filter that can be removed later
        """

        class ContextFilter(logging.Filter):
            def filter(self, record):
                record.extra_fields = getattr(record, "extra_fields", {})
                record.extra_fields.update(context)
                return True

        log_filter = ContextFilter()
        self.root_logger.addFilter(log_filter)
        return log_filter


# Create a global instance
logger_manager = LoggerManager()

# Configure default logging
logger_manager.configure()


# Helper function to get a logger
def get_logger(name: str = "") -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logger_manager.get_logger(name)
