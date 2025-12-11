"""Logging configuration for the Scouter project."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logger
    logger = logging.getLogger("scouter")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Set up root logger to avoid duplicate logs
    root_logger = logging.getLogger()
    root_logger.setLevel(
        logging.WARNING
    )  # Only show warnings and above from other libraries

    # Ensure scouter logger propagates
    logger.propagate = False
