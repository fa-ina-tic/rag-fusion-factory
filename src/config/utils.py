"""Configuration utilities for RAG Fusion Factory."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from .settings import config_manager, config


def validate_config() -> bool:
    """
    Validate the current configuration for required fields and valid values.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check required API configuration
        assert config.api.host is not None, "API host must be specified"
        assert config.api.port > 0, "API port must be positive"
        
        # Check model configuration
        assert config.model.cache_dir is not None, "Model cache directory must be specified"
        assert config.model.xgboost.n_estimators > 0, "XGBoost n_estimators must be positive"
        assert 0 < config.model.xgboost.learning_rate <= 1, "XGBoost learning rate must be between 0 and 1"
        
        # Check training configuration
        assert config.training.batch_size > 0, "Training batch size must be positive"
        assert 0 < config.training.validation_split < 1, "Validation split must be between 0 and 1"
        
        # Check normalization configuration
        valid_methods = ["min_max", "z_score", "quantile"]
        assert config.normalization.default_method in valid_methods, f"Normalization method must be one of {valid_methods}"
        
        return True
    except (AssertionError, AttributeError) as e:
        print(f"Configuration validation failed: {e}")
        return False


def get_environment() -> str:
    """Get the current environment name."""
    return config_manager.environment


def set_environment(environment: str) -> None:
    """
    Set the environment and reload configuration.
    
    Args:
        environment: Environment name (development, production, etc.)
    """
    config_manager.environment = environment
    config_manager.reload()


def create_user_config_from_template() -> None:
    """Create user.yaml from template if it doesn't exist."""
    config_dir = Path("config")
    user_config_path = config_dir / "user.yaml"
    template_path = config_dir / "user.yaml.template"
    
    if not user_config_path.exists() and template_path.exists():
        # Copy template to user config
        template_content = template_path.read_text()
        user_config_path.write_text(template_content)
        print(f"Created user configuration file: {user_config_path}")


def update_user_config(updates: Dict[str, Any]) -> None:
    """
    Update user configuration with new values.
    
    Args:
        updates: Dictionary of configuration updates
    """
    config_manager.save_user_config(updates)
    print("User configuration updated successfully")


def print_config_summary() -> None:
    """Print a summary of the current configuration."""
    print("=== RAG Fusion Factory Configuration Summary ===")
    print(f"Environment: {get_environment()}")
    print(f"API: {config.api.host}:{config.api.port} (debug: {config.api.debug})")
    print(f"Model Cache: {config.model.cache_dir}")
    print(f"XGBoost Estimators: {config.model.xgboost.n_estimators}")
    print(f"Training Batch Size: {config.training.batch_size}")
    print(f"Normalization Method: {config.normalization.default_method}")
    print(f"Log Level: {config.logging.level}")
    print("=" * 50)


def export_config_to_file(output_path: str, format: str = "yaml") -> None:
    """
    Export current configuration to a file.
    
    Args:
        output_path: Path to output file
        format: Output format ('yaml' or 'json')
    """
    output_path = Path(output_path)
    
    if format.lower() == "yaml":
        OmegaConf.save(config, output_path)
    elif format.lower() == "json":
        with open(output_path, 'w') as f:
            import json
            json.dump(OmegaConf.to_container(config, resolve=True), f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Configuration exported to: {output_path}")


def load_config_from_dict(config_dict: Dict[str, Any]) -> None:
    """
    Load configuration from a dictionary.
    
    Args:
        config_dict: Configuration dictionary
    """
    temp_config = OmegaConf.create(config_dict)
    merged_config = OmegaConf.merge(config, temp_config)
    
    # Update the global config
    config_manager._config = merged_config
    print("Configuration loaded from dictionary")


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        key_path: Configuration key path (e.g., 'api.host')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    return config_manager.get(key_path, default)


def set_config_value(key_path: str, value: Any) -> None:
    """
    Set a configuration value using dot notation.
    
    Args:
        key_path: Configuration key path (e.g., 'api.host')
        value: Value to set
    """
    config_manager.set(key_path, value)
    print(f"Configuration updated: {key_path} = {value}")