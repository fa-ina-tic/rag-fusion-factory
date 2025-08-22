"""Core data models for RAG Fusion Factory."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SearchResult:
    """Individual search result from a search engine."""
    document_id: str
    relevance_score: float
    content: str
    metadata: Dict[str, Any]
    engine_source: str


@dataclass
class SearchResults:
    """Collection of search results from a single engine."""
    query: str
    results: List[SearchResult]
    engine_id: str
    timestamp: datetime
    total_results: int


@dataclass
class TrainingExample:
    """Training example for model optimization."""
    query: str
    engine_results: Dict[str, SearchResults]
    ground_truth_labels: Dict[str, float]  # document_id -> relevance_score
    optimal_weights: Optional[np.ndarray] = None


@dataclass
class ModelConfiguration:
    """Configuration parameters for the ML models."""
    xgboost_params: Dict[str, Any]
    normalization_method: str
    contrastive_learning_params: Dict[str, Any]
    hyperparameter_search_space: Dict[str, Any]


@dataclass
class NormalizedResults:
    """Search results with normalized scores."""
    query: str
    results: List[SearchResult]
    engine_id: str
    normalization_method: str
    timestamp: datetime


@dataclass
class RankedResults:
    """Final ranked results after fusion."""
    query: str
    results: List[SearchResult]
    fusion_weights: np.ndarray
    confidence_scores: List[float]
    timestamp: datetime
    total_results: int


@dataclass
class TrainingPair:
    """Contrastive learning training pair."""
    positive_result: SearchResult
    negative_result: SearchResult
    query: str
    label: float  # 1 for positive pair, 0 for negative pair