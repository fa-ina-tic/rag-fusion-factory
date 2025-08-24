#!/usr/bin/env python3
"""
Startup script for RAG Fusion Factory with different configurations.

This script provides a convenient way to start the fusion factory with
various predefined configurations for different environments and use cases.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config.settings import load_config
from src.services.fusion_factory import FusionFactory
from src.adapters.registry import get_adapter_registry
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


class FusionFactoryLauncher:
    """Launcher for RAG Fusion Factory with different configurations."""
    
    def __init__(self):
        self.registry = get_adapter_registry()
        self.fusion_factory = None
    
    async def start_with_config(self, config_path: str, **kwargs) -> None:
        """Start fusion factory with specified configuration file.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional configuration overrides
        """
        try:
            # Load configuration
            config = load_config(config_path)
            
            # Apply any command-line overrides
            if kwargs:
                config.update(kwargs)
            
            # Setup logging
            log_level = config.get('logging', {}).get('level', 'INFO')
            setup_logging(level=log_level)
            
            logger.info(f"Starting RAG Fusion Factory with config: {config_path}")
            
            # Create adapters from configuration
            adapters_config = config.get('engines', [])
            if not adapters_config:
                logger.error("No engines configured. Please check your configuration file.")
                return
            
            adapters = []
            for adapter_config in adapters_config:
                try:
                    adapter = self.registry.create_adapter(
                        engine_type=adapter_config['engine_type'],
                        engine_id=adapter_config['engine_id'],
                        config=adapter_config['config'],
                        timeout=adapter_config.get('timeout', 30.0)
                    )
                    adapters.append(adapter)
                    logger.info(f"Created adapter: {adapter.engine_id} ({adapter_config['engine_type']})")
                except Exception as e:
                    logger.error(f"Failed to create adapter {adapter_config.get('engine_id', 'unknown')}: {str(e)}")
                    continue
            
            if not adapters:
                logger.error("No adapters were successfully created. Exiting.")
                return
            
            # Initialize fusion factory
            fusion_config = config.get('fusion', {})
            self.fusion_factory = FusionFactory(adapters, fusion_config)
            
            # Perform health checks
            logger.info("Performing initial health checks...")
            health_results = await self.registry.health_check_all()
            
            healthy_count = sum(1 for is_healthy in health_results.values() if is_healthy)
            total_count = len(health_results)
            
            logger.info(f"Health check results: {healthy_count}/{total_count} adapters healthy")
            
            if healthy_count == 0:
                logger.error("No adapters are healthy. Please check your configuration and engine availability.")
                return
            
            # Start the fusion factory
            logger.info("RAG Fusion Factory started successfully!")
            logger.info(f"Available adapters: {list(health_results.keys())}")
            
            # Keep running (in a real deployment, this would start the API server)
            await self._run_interactive_mode()
            
        except Exception as e:
            logger.error(f"Failed to start fusion factory: {str(e)}")
            raise
    
    async def start_development_mode(self) -> None:
        """Start fusion factory in development mode with mock adapters."""
        logger.info("Starting RAG Fusion Factory in development mode...")
        
        # Create mock adapters for development
        mock_configs = [
            {
                'engine_type': 'mock_inmemory',
                'engine_id': 'dev_mock_1',
                'config': {
                    'response_delay': 0.1,
                    'failure_rate': 0.1,
                    'max_results': 5
                },
                'timeout': 5.0
            },
            {
                'engine_type': 'mock_inmemory',
                'engine_id': 'dev_mock_2',
                'config': {
                    'response_delay': 0.2,
                    'failure_rate': 0.05,
                    'max_results': 8
                },
                'timeout': 5.0
            }
        ]
        
        adapters = self.registry.create_adapters_from_config(mock_configs)
        
        # Initialize fusion factory with default config
        self.fusion_factory = FusionFactory(adapters, {})
        
        logger.info("Development mode started with mock adapters")
        await self._run_interactive_mode()
    
    async def start_testing_mode(self) -> None:
        """Start fusion factory in testing mode with minimal configuration."""
        logger.info("Starting RAG Fusion Factory in testing mode...")
        
        # Create minimal test configuration
        test_config = {
            'engine_type': 'mock_inmemory',
            'engine_id': 'test_engine',
            'config': {
                'response_delay': 0.01,
                'failure_rate': 0.0,
                'max_results': 3,
                'documents': [
                    {
                        'id': 'test1',
                        'title': 'Test Document 1',
                        'content': 'This is a test document for testing purposes',
                        'category': 'test'
                    }
                ]
            },
            'timeout': 1.0
        }
        
        adapter = self.registry.create_adapter(
            engine_type=test_config['engine_type'],
            engine_id=test_config['engine_id'],
            config=test_config['config'],
            timeout=test_config['timeout']
        )
        
        # Initialize fusion factory
        self.fusion_factory = FusionFactory([adapter], {})
        
        logger.info("Testing mode started with minimal configuration")
        await self._run_interactive_mode()
    
    async def _run_interactive_mode(self) -> None:
        """Run interactive mode for testing queries."""
        logger.info("\nInteractive mode started. Type 'quit' to exit.")
        logger.info("Enter search queries to test the fusion factory:")
        
        while True:
            try:
                query = input("\n> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                logger.info(f"Searching for: '{query}'")
                
                # Execute search
                results = await self.fusion_factory.search(query, limit=5)
                
                # Display results
                if results.results:
                    logger.info(f"Found {len(results.results)} results:")
                    for i, result in enumerate(results.results, 1):
                        logger.info(f"  {i}. [{result.engine_source}] {result.document_id}")
                        logger.info(f"     Score: {result.relevance_score:.3f}")
                        logger.info(f"     Content: {result.content[:100]}...")
                else:
                    logger.info("No results found.")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Search failed: {str(e)}")
        
        logger.info("Interactive mode ended.")
    
    async def list_available_adapters(self) -> None:
        """List all available adapter types."""
        available = self.registry.get_available_adapters()
        
        print("Available adapter types:")
        for adapter_type in sorted(available):
            print(f"  - {adapter_type}")
        
        print(f"\nTotal: {len(available)} adapter types available")
    
    async def validate_config(self, config_path: str) -> bool:
        """Validate a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            config = load_config(config_path)
            
            # Check required sections
            if 'engines' not in config:
                logger.error("Configuration missing 'engines' section")
                return False
            
            engines_config = config['engines']
            if not isinstance(engines_config, list) or not engines_config:
                logger.error("'engines' section must be a non-empty list")
                return False
            
            # Validate each engine configuration
            for i, engine_config in enumerate(engines_config):
                if not isinstance(engine_config, dict):
                    logger.error(f"Engine {i} configuration must be a dictionary")
                    return False
                
                required_fields = ['engine_type', 'engine_id', 'config']
                for field in required_fields:
                    if field not in engine_config:
                        logger.error(f"Engine {i} missing required field: {field}")
                        return False
                
                # Check if engine type is available
                engine_type = engine_config['engine_type']
                available_types = self.registry.get_available_adapters()
                if engine_type not in available_types:
                    logger.error(f"Engine {i} uses unknown type: {engine_type}")
                    logger.error(f"Available types: {', '.join(available_types)}")
                    return False
            
            logger.info(f"Configuration file '{config_path}' is valid")
            logger.info(f"Found {len(engines_config)} engine configurations")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False


async def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description="RAG Fusion Factory Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with custom configuration
  python start_fusion_factory.py --config config/production.yaml
  
  # Start in development mode
  python start_fusion_factory.py --dev
  
  # Start in testing mode
  python start_fusion_factory.py --test
  
  # List available adapters
  python start_fusion_factory.py --list-adapters
  
  # Validate configuration
  python start_fusion_factory.py --validate config/my_config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dev', '--development',
        action='store_true',
        help='Start in development mode with mock adapters'
    )
    
    parser.add_argument(
        '--test', '--testing',
        action='store_true',
        help='Start in testing mode with minimal configuration'
    )
    
    parser.add_argument(
        '--list-adapters',
        action='store_true',
        help='List all available adapter types'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        help='Validate a configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Setup basic logging
    setup_logging(level=args.log_level)
    
    launcher = FusionFactoryLauncher()
    
    try:
        if args.list_adapters:
            await launcher.list_available_adapters()
        elif args.validate:
            is_valid = await launcher.validate_config(args.validate)
            sys.exit(0 if is_valid else 1)
        elif args.dev:
            await launcher.start_development_mode()
        elif args.test:
            await launcher.start_testing_mode()
        elif args.config:
            await launcher.start_with_config(args.config)
        else:
            # Default to minimal configuration if no options specified
            logger.info("No configuration specified, starting with minimal configuration")
            await launcher.start_with_config('config/minimal.yaml')
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Launcher failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())