"""
Utility functions and configuration for the recommendation system.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "recommender",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Config:
    """
    Configuration class for managing project paths and settings.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            base_dir: Base directory for the project (defaults to project root)
        """
        if base_dir is None:
            # Assume utils.py is in src/, so parent.parent is project root
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)

        # Data directories
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"

        # Source directory
        self.src_dir = self.base_dir / "src"

        # Results directories
        self.results_dir = self.base_dir / "results"
        self.models_dir = self.results_dir / "models"
        self.metrics_dir = self.results_dir / "metrics"

        # Notebooks directory
        self.notebooks_dir = self.base_dir / "notebooks"

        # Tests directory
        self.tests_dir = self.base_dir / "tests"

    def ensure_dirs(self) -> None:
        """Create all necessary directories if they don't exist."""
        dirs = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.metrics_dir,
            self.notebooks_dir,
            self.tests_dir
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config(base_dir={self.base_dir})"
