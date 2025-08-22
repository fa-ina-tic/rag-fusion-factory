# RAG Fusion Factory - Main package

from .config.settings import config, config_manager
from .utils.logging import setup_logging, get_logger
from .models import (
    SearchResult,
    SearchResults,
    TrainingExample,
    ModelConfiguration,
    NormalizedResults,
    RankedResults,
    TrainingPair,
)

__version__ = "0.1.0"

__all__ = [
    "config",
    "config_manager",
    "setup_logging",
    "get_logger",
    "SearchResult",
    "SearchResults",
    "TrainingExample", 
    "ModelConfiguration",
    "NormalizedResults",
    "RankedResults",
    "TrainingPair",
]