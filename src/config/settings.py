"""Configuration management for RAG Fusion Factory using OmegaConf."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """Configuration manager using OmegaConf for YAML-based configuration."""
    
    def __init__(self, config_dir: str = "config", environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (development, production, etc.)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "default")
        self._config: Optional[DictConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML files."""
        # Load default configuration
        default_config_path = self.config_dir / "default.yaml"
        if not default_config_path.exists():
            raise FileNotFoundError(f"Default configuration file not found: {default_config_path}")
        
        config = OmegaConf.load(default_config_path)
        
        # Load environment-specific configuration if it exists
        env_config_path = self.config_dir / f"{self.environment}.yaml"
        if env_config_path.exists():
            env_config = OmegaConf.load(env_config_path)
            config = OmegaConf.merge(config, env_config)
        
        # Load user-specific configuration if it exists
        user_config_path = self.config_dir / "user.yaml"
        if user_config_path.exists():
            user_config = OmegaConf.load(user_config_path)
            config = OmegaConf.merge(config, user_config)
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        self._config = config
    
    def _apply_env_overrides(self, config: DictConfig) -> DictConfig:
        """Apply environment variable overrides to configuration."""
        # Define environment variable mappings
        env_mappings = {
            "RAG_API_HOST": "api.host",
            "RAG_API_PORT": "api.port", 
            "RAG_API_DEBUG": "api.debug",
            "RAG_SEARCH_TIMEOUT": "search.timeout",
            "RAG_MAX_CONCURRENT_ENGINES": "search.max_concurrent_engines",
            "RAG_MODEL_CACHE_DIR": "model.cache_dir",
            "RAG_LOG_LEVEL": "logging.level",
            "RAG_TRAINING_BATCH_SIZE": "training.batch_size",
            "RAG_NORMALIZATION_METHOD": "normalization.default_method",
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_value.lower() in ("true", "false"):
                    env_value = env_value.lower() == "true"
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif env_value.replace(".", "").isdigit():
                    env_value = float(env_value)
                
                OmegaConf.set(config, config_path, env_value)
        
        return config
    
    @property
    def config(self) -> DictConfig:
        """Get the current configuration."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'api.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def is_read_only(self) -> bool:
        """
        Check if configuration is read-only.
        Configuration changes should only be made by editing YAML files directly.
        
        Returns:
            Always True - configuration is read-only at runtime
        """
        return True


# Global configuration manager instance
config_manager = ConfigManager()
config = config_manager.config


def get_xgboost_config() -> Dict[str, Any]:
    """Get XGBoost configuration parameters."""
    return OmegaConf.to_container(config.model.xgboost, resolve=True)


def get_contrastive_learning_config() -> Dict[str, Any]:
    """Get contrastive learning configuration parameters."""
    return OmegaConf.to_container(config.training.contrastive_learning, resolve=True)


def get_hyperparameter_search_space() -> Dict[str, Any]:
    """Get hyperparameter search space for optimization."""
    return OmegaConf.to_container(config.hyperparameter_optimization.search_space, resolve=True)


def get_api_config() -> Dict[str, Any]:
    """Get API configuration parameters."""
    return OmegaConf.to_container(config.api, resolve=True)


def get_search_config() -> Dict[str, Any]:
    """Get search engine configuration parameters."""
    return OmegaConf.to_container(config.search, resolve=True)


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration parameters."""
    return OmegaConf.to_container(config.logging, resolve=True)


def get_training_config() -> Dict[str, Any]:
    """Get training configuration parameters."""
    return OmegaConf.to_container(config.training, resolve=True)


def get_normalization_config() -> Dict[str, Any]:
    """Get normalization configuration parameters."""
    return OmegaConf.to_container(config.normalization, resolve=True)


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration parameters."""
    return OmegaConf.to_container(config.monitoring, resolve=True)


def get_circuit_breaker_config() -> Dict[str, Any]:
    """Get circuit breaker configuration parameters."""
    return OmegaConf.to_container(config.circuit_breaker, resolve=True)