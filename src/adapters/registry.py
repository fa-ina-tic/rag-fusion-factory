"""Search engine adapter registry for registration and discovery."""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .base import SearchEngineAdapter
from .elasticsearch_adapter import ElasticsearchAdapter
from .solr_adapter import SolrAdapter
from .opensearch_adapter import OpenSearchAdapter
from .mock_adapter import InMemoryMockAdapter, FileMockAdapter
from .custom_adapter import RestApiAdapter, DatabaseAdapter, WebScrapingAdapter


logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for search engine adapters.
    
    Manages registration, discovery, and instantiation of search engine adapters.
    """
    
    def __init__(self):
        """Initialize the adapter registry."""
        self._adapters: Dict[str, Type[SearchEngineAdapter]] = {}
        self._instances: Dict[str, SearchEngineAdapter] = {}
        
        # Register built-in adapters
        self._register_builtin_adapters()
    
    def _register_builtin_adapters(self) -> None:
        """Register built-in search engine adapters."""
        self.register_adapter('elasticsearch', ElasticsearchAdapter)
        self.register_adapter('solr', SolrAdapter)
        self.register_adapter('opensearch', OpenSearchAdapter)
        self.register_adapter('mock_inmemory', InMemoryMockAdapter)
        self.register_adapter('mock_file', FileMockAdapter)
        self.register_adapter('rest_api', RestApiAdapter)
        self.register_adapter('database', DatabaseAdapter)
        self.register_adapter('web_scraping', WebScrapingAdapter)
        logger.info("Registered built-in adapters: elasticsearch, solr, opensearch, mock_inmemory, mock_file, rest_api, database, web_scraping")
    
    def register_adapter(self, engine_type: str, adapter_class: Type[SearchEngineAdapter]) -> None:
        """Register a search engine adapter class.
        
        Args:
            engine_type: Unique identifier for the engine type
            adapter_class: Adapter class that extends SearchEngineAdapter
            
        Raises:
            ValueError: If the adapter class is invalid
        """
        if not issubclass(adapter_class, SearchEngineAdapter):
            raise ValueError(f"Adapter class must extend SearchEngineAdapter: {adapter_class}")
        
        if engine_type in self._adapters:
            logger.warning(f"Overriding existing adapter registration for '{engine_type}'")
        
        self._adapters[engine_type] = adapter_class
        logger.info(f"Registered adapter '{engine_type}': {adapter_class.__name__}")
    
    def unregister_adapter(self, engine_type: str) -> bool:
        """Unregister a search engine adapter.
        
        Args:
            engine_type: Engine type to unregister
            
        Returns:
            True if adapter was unregistered, False if not found
        """
        if engine_type not in self._adapters:
            return False
        
        # Remove any instances of this adapter type
        instances_to_remove = [
            engine_id for engine_id, instance in self._instances.items()
            if instance.__class__ == self._adapters[engine_type]
        ]
        
        for engine_id in instances_to_remove:
            del self._instances[engine_id]
        
        del self._adapters[engine_type]
        logger.info(f"Unregistered adapter '{engine_type}'")
        return True
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapter types.
        
        Returns:
            List of registered adapter type names
        """
        return list(self._adapters.keys())
    
    def create_adapter(self, engine_type: str, engine_id: str, 
                      config: Dict[str, Any], timeout: float = 30.0) -> SearchEngineAdapter:
        """Create a new adapter instance.
        
        Args:
            engine_type: Type of search engine adapter
            engine_id: Unique identifier for this adapter instance
            config: Configuration dictionary for the adapter
            timeout: Default timeout for operations
            
        Returns:
            Configured adapter instance
            
        Raises:
            ValueError: If engine_type is not registered or engine_id already exists
        """
        if engine_type not in self._adapters:
            available = ', '.join(self.get_available_adapters())
            raise ValueError(f"Unknown engine type '{engine_type}'. Available: {available}")
        
        if engine_id in self._instances:
            raise ValueError(f"Adapter instance '{engine_id}' already exists")
        
        try:
            adapter_class = self._adapters[engine_type]
            adapter_instance = adapter_class(engine_id, config, timeout)
            
            self._instances[engine_id] = adapter_instance
            logger.info(f"Created adapter instance '{engine_id}' of type '{engine_type}'")
            
            return adapter_instance
            
        except Exception as e:
            logger.error(f"Failed to create adapter '{engine_id}' of type '{engine_type}': {str(e)}")
            raise ValueError(f"Failed to create adapter: {str(e)}") from e
    
    def get_adapter(self, engine_id: str) -> Optional[SearchEngineAdapter]:
        """Get an existing adapter instance by ID.
        
        Args:
            engine_id: Unique identifier of the adapter instance
            
        Returns:
            Adapter instance if found, None otherwise
        """
        return self._instances.get(engine_id)
    
    def remove_adapter(self, engine_id: str) -> bool:
        """Remove an adapter instance.
        
        Args:
            engine_id: Unique identifier of the adapter instance
            
        Returns:
            True if adapter was removed, False if not found
        """
        if engine_id not in self._instances:
            return False
        
        del self._instances[engine_id]
        logger.info(f"Removed adapter instance '{engine_id}'")
        return True
    
    def get_all_adapters(self) -> Dict[str, SearchEngineAdapter]:
        """Get all registered adapter instances.
        
        Returns:
            Dictionary mapping engine_id to adapter instance
        """
        return self._instances.copy()
    
    def get_adapters_by_type(self, engine_type: str) -> List[SearchEngineAdapter]:
        """Get all adapter instances of a specific type.
        
        Args:
            engine_type: Type of search engine adapter
            
        Returns:
            List of adapter instances of the specified type
        """
        if engine_type not in self._adapters:
            return []
        
        adapter_class = self._adapters[engine_type]
        return [
            instance for instance in self._instances.values()
            if isinstance(instance, adapter_class)
        ]
    
    def create_adapters_from_config(self, adapters_config: List[Dict[str, Any]]) -> List[SearchEngineAdapter]:
        """Create multiple adapters from configuration list.
        
        Args:
            adapters_config: List of adapter configurations, each containing:
                - engine_type: Type of search engine
                - engine_id: Unique identifier
                - config: Engine-specific configuration
                - timeout: Optional timeout (default: 30.0)
                
        Returns:
            List of created adapter instances
            
        Raises:
            ValueError: If any adapter creation fails
        """
        created_adapters = []
        
        for adapter_config in adapters_config:
            try:
                engine_type = adapter_config['engine_type']
                engine_id = adapter_config['engine_id']
                config = adapter_config['config']
                timeout = adapter_config.get('timeout', 30.0)
                
                adapter = self.create_adapter(engine_type, engine_id, config, timeout)
                created_adapters.append(adapter)
                
            except KeyError as e:
                raise ValueError(f"Missing required configuration key: {str(e)}") from e
            except Exception as e:
                # Clean up any adapters created so far
                for adapter in created_adapters:
                    self.remove_adapter(adapter.engine_id)
                raise ValueError(f"Failed to create adapters: {str(e)}") from e
        
        logger.info(f"Created {len(created_adapters)} adapters from configuration")
        return created_adapters
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health checks on all registered adapter instances.
        
        Returns:
            Dictionary mapping engine_id to health status
        """
        health_results = {}
        
        for engine_id, adapter in self._instances.items():
            try:
                is_healthy = await adapter.health_check_with_timeout()
                health_results[engine_id] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for adapter '{engine_id}': {str(e)}")
                health_results[engine_id] = False
        
        return health_results
    
    def get_adapter_info(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an adapter instance.
        
        Args:
            engine_id: Unique identifier of the adapter instance
            
        Returns:
            Dictionary containing adapter information, None if not found
        """
        adapter = self.get_adapter(engine_id)
        if not adapter:
            return None
        
        return {
            'engine_id': adapter.engine_id,
            'engine_type': adapter.__class__.__name__.replace('Adapter', '').lower(),
            'configuration': adapter.get_configuration(),
            'is_healthy': adapter.is_healthy,
            'last_health_check': adapter.last_health_check.isoformat() if adapter.last_health_check else None,
            'timeout': adapter.timeout
        }
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status and statistics.
        
        Returns:
            Dictionary containing registry status information
        """
        adapter_types = {}
        for engine_id, adapter in self._instances.items():
            adapter_type = adapter.__class__.__name__.replace('Adapter', '').lower()
            if adapter_type not in adapter_types:
                adapter_types[adapter_type] = []
            adapter_types[adapter_type].append(engine_id)
        
        return {
            'registered_types': list(self._adapters.keys()),
            'total_instances': len(self._instances),
            'instances_by_type': adapter_types,
            'healthy_instances': sum(1 for adapter in self._instances.values() if adapter.is_healthy)
        }


# Global registry instance
adapter_registry = AdapterRegistry()


def get_adapter_registry() -> AdapterRegistry:
    """Get the global adapter registry instance.
    
    Returns:
        Global AdapterRegistry instance
    """
    return adapter_registry