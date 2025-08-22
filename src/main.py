"""Main application entry point for RAG Fusion Factory."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .utils.logging import setup_logging, get_logger
from .config.settings import ConfigManager, get_api_config, get_logging_config
from .services.fusion_factory import FusionFactory
from .adapters.registry import get_adapter_registry


class RAGFusionFactory:
    """Main application orchestrator for RAG Fusion Factory."""
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize RAG Fusion Factory application.
        
        Args:
            config_path: Path to configuration file or directory
            environment: Environment name (development, production, etc.)
        """
        self.config_path = config_path
        self.environment = environment
        
        # Initialize configuration manager
        config_dir = "config"
        config_file = None
        
        if config_path:
            config_path_obj = Path(config_path)
            if config_path_obj.is_dir():
                config_dir = str(config_path_obj)
            elif config_path_obj.is_file():
                config_file = str(config_path_obj)
                config_dir = str(config_path_obj.parent)
        
        self.config_manager = ConfigManager(
            config_dir=config_dir, 
            environment=environment,
            config_file=config_file
        )
        self.config = self.config_manager.config
        
        # Initialize logging with configuration
        self.logger = setup_logging()
        
        # Initialize fusion factory
        self.fusion_factory = None
        
    def validate_configuration(self) -> bool:
        """
        Validate the loaded configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required configuration sections
            required_sections = ['api', 'search', 'model', 'normalization', 'logging', 'training']
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate API configuration
            api_config = self.config.api
            if not isinstance(api_config.port, int) or api_config.port <= 0:
                self.logger.error("Invalid API port configuration")
                return False
            
            # Validate model cache directory
            cache_dir = Path(self.config.model.cache_dir)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Cannot create model cache directory: {e}")
                return False
            
            # Validate normalization method
            valid_methods = ['min_max', 'z_score', 'quantile']
            if self.config.normalization.default_method not in valid_methods:
                self.logger.error(f"Invalid normalization method: {self.config.normalization.default_method}")
                return False
            
            self.logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def log_configuration_info(self) -> None:
        """Log important configuration information."""
        # Use instance-specific configuration instead of global
        api_config = self.config.api
        self.logger.info(f"Configuration loaded: API will run on {api_config.host}:{api_config.port}")
        self.logger.info(f"Debug mode: {api_config.debug}")
        self.logger.info(f"Model cache directory: {self.config.model.cache_dir}")
        self.logger.info(f"Default normalization method: {self.config.normalization.default_method}")
        self.logger.info(f"Environment: {self.environment or 'default'}")
        
        # Log XGBoost configuration
        self.logger.info(f"XGBoost estimators: {self.config.model.xgboost.n_estimators}")
        self.logger.info(f"Training batch size: {self.config.training.batch_size}")
        
        # Log search configuration
        self.logger.info(f"Search timeout: {self.config.search.timeout}s")
        self.logger.info(f"Max concurrent engines: {self.config.search.max_concurrent_engines}")
    
    async def process_sample_query(self, query: str = "sample query") -> None:
        """Process a sample query to demonstrate fusion functionality.
        
        Args:
            query: Sample query to process
        """
        if not self.fusion_factory:
            self.logger.error("Fusion factory not initialized")
            return
        
        try:
            # Get engine status
            engine_status = self.fusion_factory.get_engine_status()
            self.logger.info(f"Engine status: {engine_status}")
            
            # Check if any adapters are available
            if engine_status['total_instances'] == 0:
                self.logger.info("No search engine adapters configured - fusion factory ready but no engines available")
                return
            
            # Perform health check
            health_results = await self.fusion_factory.health_check_engines()
            self.logger.info(f"Engine health check results: {health_results}")
            
            # Process sample query if engines are available and healthy
            healthy_engines = [engine_id for engine_id, is_healthy in health_results.items() if is_healthy]
            if healthy_engines:
                self.logger.info(f"Processing sample query with {len(healthy_engines)} healthy engines")
                results = await self.fusion_factory.process_query(query)
                self.logger.info(f"Sample query processed: {results.total_results} results returned")
            else:
                self.logger.info("No healthy engines available for query processing")
                
        except Exception as e:
            self.logger.error(f"Sample query processing failed: {e}")
    
    async def run_async(self) -> int:
        """
        Run the RAG Fusion Factory application asynchronously.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            self.logger.info("Starting RAG Fusion Factory")
            
            # Validate configuration
            if not self.validate_configuration():
                self.logger.error("Configuration validation failed")
                return 1
            
            # Log configuration information
            self.log_configuration_info()
            
            # Initialize fusion factory core engine
            self.fusion_factory = FusionFactory()
            self.logger.info("Fusion factory core engine initialized")
            
            # Initialize adapters from configuration if available
            if hasattr(self.config, 'adapters') and self.config.adapters:
                try:
                    adapters = self.fusion_factory.initialize_adapters_from_config(self.config.adapters)
                    self.logger.info(f"Initialized {len(adapters)} search engine adapters")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize adapters from config: {e}")
            
            # TODO: This will be expanded in future tasks to include:
            # - FastAPI server startup
            # - Model loading
            # - Training pipeline setup
            
            self.logger.info("RAG Fusion Factory initialized successfully")
            self.logger.info("Core fusion engine ready for query processing")
            
            # Demonstrate fusion functionality with sample query
            await self.process_sample_query("machine learning information retrieval")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Application startup failed: {e}")
            return 1
    
    def run(self) -> int:
        """
        Run the RAG Fusion Factory application.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        return asyncio.run(self.run_async())


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="rag-fusion-factory",
        description="RAG Fusion Factory - Automated system for optimizing multiple information retrieval methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default configuration
  %(prog)s --config /path/to/config          # Use custom configuration directory
  %(prog)s --config config.yaml              # Use specific configuration file
  %(prog)s --environment production          # Run in production environment
  %(prog)s --config custom/ --env staging    # Custom config directory with staging environment

For more information, visit: https://github.com/your-org/rag-fusion-factory
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file or directory (default: ./config)"
    )
    
    parser.add_argument(
        "--environment", "--env", "-e",
        type=str,
        help="Environment name (development, production, etc.)"
    )
    
    # Future CLI commands (placeholders for upcoming tasks)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training mode (to be implemented in task 10.1)"
    )
    
    parser.add_argument(
        "--list-engines",
        action="store_true",
        help="List configured search engines and their status (to be implemented in task 10.2)"
    )
    
    parser.add_argument(
        "--metrics",
        action="store_true", 
        help="Display performance metrics (to be implemented in task 10.2)"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization (to be implemented in task 10.2)"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (overrides configuration)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (equivalent to --log-level DEBUG)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output (equivalent to --log-level ERROR)"
    )
    
    return parser


def main() -> int:
    """
    Main application entry point with CLI argument parsing.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle future CLI commands (placeholders)
    if args.train:
        print("Training mode will be implemented in task 10.1")
        return 0
    
    if args.list_engines:
        print("Engine listing will be implemented in task 10.2")
        return 0
    
    if args.metrics:
        print("Metrics display will be implemented in task 10.2")
        return 0
    
    if args.optimize:
        print("Hyperparameter optimization will be implemented in task 10.2")
        return 0
    
    # Handle logging level overrides
    if args.verbose:
        import os
        os.environ["RAG_LOG_LEVEL"] = "DEBUG"
    elif args.quiet:
        import os
        os.environ["RAG_LOG_LEVEL"] = "ERROR"
    elif args.log_level:
        import os
        os.environ["RAG_LOG_LEVEL"] = args.log_level
    
    try:
        # Initialize and run the application
        app = RAGFusionFactory(
            config_path=args.config,
            environment=args.environment
        )
        return app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())