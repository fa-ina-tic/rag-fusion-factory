"""Logging configuration for RAG Fusion Factory."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from ..config.settings import get_logging_config


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    logger_name: Optional[str] = None,
    file_logging: Optional[bool] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom log format string
        logger_name: Name for the logger (defaults to 'rag_fusion_factory')
        file_logging: Enable file logging
        log_file: Path to log file
    
    Returns:
        Configured logger instance
    """
    log_config = get_logging_config()
    
    log_level = level or log_config.get("level", "INFO")
    log_format = format_string or log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    name = logger_name or "rag_fusion_factory"
    enable_file_logging = file_logging if file_logging is not None else log_config.get("file_logging", False)
    log_file_path = log_file or log_config.get("log_file", "logs/rag_fusion_factory.log")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if log_format.lower() == "json":
        # Use a standard format for JSON-style logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if enabled
    if enable_file_logging:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        max_bytes = _parse_size(log_config.get("max_file_size", "10MB"))
        backup_count = log_config.get("backup_count", 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes."""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"rag_fusion_factory.{name}")


# Create default application logger
app_logger = setup_logging()


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)