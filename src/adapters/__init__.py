# Search engine adapters package

from .base import (
    SearchEngineAdapter,
    SearchEngineError,
    SearchEngineTimeoutError,
    SearchEngineConnectionError,
)
from .elasticsearch_adapter import ElasticsearchAdapter
from .solr_adapter import SolrAdapter
from .registry import AdapterRegistry, adapter_registry, get_adapter_registry

__all__ = [
    "SearchEngineAdapter",
    "SearchEngineError", 
    "SearchEngineTimeoutError",
    "SearchEngineConnectionError",
    "ElasticsearchAdapter",
    "SolrAdapter",
    "AdapterRegistry",
    "adapter_registry",
    "get_adapter_registry",
]