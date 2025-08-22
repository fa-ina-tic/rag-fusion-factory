"""Tests for CLI argument parser and main entry point."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main import create_argument_parser, RAGFusionFactory, main


class TestArgumentParser:
    """Test CLI argument parser functionality."""
    
    def test_argument_parser_creation(self):
        """Test that argument parser is created successfully."""
        parser = create_argument_parser()
        assert parser is not None
        assert parser.prog == "rag-fusion-factory"
    
    def test_config_argument(self):
        """Test --config argument parsing."""
        parser = create_argument_parser()
        
        # Test short form
        args = parser.parse_args(["-c", "test.yaml"])
        assert args.config == "test.yaml"
        
        # Test long form
        args = parser.parse_args(["--config", "/path/to/config"])
        assert args.config == "/path/to/config"
    
    def test_environment_argument(self):
        """Test --environment argument parsing."""
        parser = create_argument_parser()
        
        # Test long form
        args = parser.parse_args(["--environment", "production"])
        assert args.environment == "production"
        
        # Test short form
        args = parser.parse_args(["--env", "staging"])
        assert args.environment == "staging"
        
        # Test shortest form
        args = parser.parse_args(["-e", "development"])
        assert args.environment == "development"
    
    def test_future_cli_commands(self):
        """Test placeholder CLI commands for future implementation."""
        parser = create_argument_parser()
        
        # Test training command
        args = parser.parse_args(["--train"])
        assert args.train is True
        
        # Test list engines command
        args = parser.parse_args(["--list-engines"])
        assert args.list_engines is True
        
        # Test metrics command
        args = parser.parse_args(["--metrics"])
        assert args.metrics is True
        
        # Test optimize command
        args = parser.parse_args(["--optimize"])
        assert args.optimize is True
    
    def test_logging_arguments(self):
        """Test logging-related arguments."""
        parser = create_argument_parser()
        
        # Test log level
        args = parser.parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"
        
        # Test verbose
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True
        
        # Test quiet
        args = parser.parse_args(["--quiet"])
        assert args.quiet is True


class TestRAGFusionFactory:
    """Test RAG Fusion Factory application orchestrator."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        app = RAGFusionFactory()
        assert app.config_path is None
        assert app.environment is None
        assert app.config_manager is not None
        assert app.config is not None
        assert app.logger is not None
    
    def test_initialization_with_config_path(self):
        """Test initialization with custom config path."""
        app = RAGFusionFactory(config_path="config/development.yaml")
        assert app.config_path == "config/development.yaml"
        assert app.config_manager is not None
    
    def test_initialization_with_environment(self):
        """Test initialization with custom environment."""
        app = RAGFusionFactory(environment="production")
        assert app.environment == "production"
        assert app.config_manager is not None
    
    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        app = RAGFusionFactory()
        assert app.validate_configuration() is True
    
    def test_configuration_validation_with_development_config(self):
        """Test configuration validation with development config."""
        app = RAGFusionFactory(config_path="config/development.yaml")
        assert app.validate_configuration() is True
        
        # Verify development-specific settings are loaded
        assert app.config.api.debug is True
        assert app.config.api.port == 8001
        assert app.config.model.xgboost.n_estimators == 50


class TestMainFunction:
    """Test main function and CLI integration."""
    
    @patch('sys.argv', ['rag-fusion-factory', '--help'])
    def test_help_command(self):
        """Test that help command works without errors."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse exits with code 0 for help
        assert exc_info.value.code == 0
    
    @patch('sys.argv', ['rag-fusion-factory', '--train'])
    def test_train_placeholder(self, capsys):
        """Test train command placeholder."""
        result = main()
        captured = capsys.readouterr()
        assert result == 0
        assert "Training mode will be implemented in task 10.1" in captured.out
    
    @patch('sys.argv', ['rag-fusion-factory', '--list-engines'])
    def test_list_engines_placeholder(self, capsys):
        """Test list-engines command placeholder."""
        result = main()
        captured = capsys.readouterr()
        assert result == 0
        assert "Engine listing will be implemented in task 10.2" in captured.out
    
    @patch('sys.argv', ['rag-fusion-factory', '--metrics'])
    def test_metrics_placeholder(self, capsys):
        """Test metrics command placeholder."""
        result = main()
        captured = capsys.readouterr()
        assert result == 0
        assert "Metrics display will be implemented in task 10.2" in captured.out
    
    @patch('sys.argv', ['rag-fusion-factory', '--optimize'])
    def test_optimize_placeholder(self, capsys):
        """Test optimize command placeholder."""
        result = main()
        captured = capsys.readouterr()
        assert result == 0
        assert "Hyperparameter optimization will be implemented in task 10.2" in captured.out
    
    @patch('sys.argv', ['rag-fusion-factory'])
    def test_default_execution(self):
        """Test default application execution."""
        result = main()
        assert result == 0
    
    @patch('sys.argv', ['rag-fusion-factory', '--config', 'config/development.yaml'])
    def test_config_file_loading(self):
        """Test configuration file loading."""
        result = main()
        assert result == 0
    
    @patch('sys.argv', ['rag-fusion-factory', '--environment', 'development'])
    def test_environment_loading(self):
        """Test environment-specific configuration loading."""
        result = main()
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__])