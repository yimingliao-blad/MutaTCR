"""
Logging utilities for TCRP Benchmark V2.

This module provides centralized logging configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "tcrp_benchmark",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up logging with console and optional file output.

    Configures both the named logger and the root logger to ensure
    all loggers in the application output correctly.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Log message format

    Returns:
        Configured logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger (so all loggers inherit this config)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers from root
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Return named logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    return logger


def get_logger(name: str = "tcrp_benchmark") -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class that provides logging functionality."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"tcrp_benchmark.{self.__class__.__name__}")


if __name__ == "__main__":
    # Test logging utilities
    print("Testing logging utilities...")

    logger = setup_logging("test_logger", level="DEBUG")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    print("\nLogging utilities test completed!")
