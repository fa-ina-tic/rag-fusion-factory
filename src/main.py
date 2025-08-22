"""Main application entry point for RAG Fusion Factory."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging import setup_logging, get_logger
from config.settings import config, get_api_config, get_logging_config

# Initialize logging
logger = setup_logging()


def main():
    """Main application entry point."""
    logger.info("Starting RAG Fusion Factory")
    
    # Log configuration information
    api_config = get_api_config()
    logger.info(f"Configuration loaded: API will run on {api_config['host']}:{api_config['port']}")
    logger.info(f"Debug mode: {api_config['debug']}")
    logger.info(f"Model cache directory: {config.model.cache_dir}")
    logger.info(f"Default normalization method: {config.normalization.default_method}")
    logger.info(f"Environment: {config.get('environment', 'default')}")
    
    # Log XGBoost configuration
    logger.info(f"XGBoost estimators: {config.model.xgboost.n_estimators}")
    logger.info(f"Training batch size: {config.training.batch_size}")


if __name__ == "__main__":
    main()