"""Tests for ModelTrainingPipeline."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import numpy as np

from src.services.training_pipeline import ModelTrainingPipeline
from src.models.core import (
    SearchResult, SearchResults, TrainingExample, 
    ModelConfiguration
)
from src.models.weight_predictor import WeightPredictorModel


class TestModelTrainingPipeline:
    """Test cases for ModelTrainingPipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.backup_dir = Path(self.temp_dir) / "backups"
        
        self.pipeline = ModelTrainingPipeline(
            models_dir=str(self.models_dir),
            backup_dir=str(self.backup_dir)
        )
        
        # Create sample training data
        self.sample_training_data = [
            TrainingExample(
                query="test query 1",
                engine_results={
                    'engine1': SearchResults(
                        query="test query 1",
                        results=[
                            SearchResult("doc1", 0.9, "content1", {}, "engine1"),
                            SearchResult("doc2", 0.7, "content2", {}, "engine1")
                        ],
                        engine_id="engine1",
                        timestamp=datetime.now(),
                        total_results=2
                    ),
                    'engine2': SearchResults(
                        query="test query 1",
                        results=[
                            SearchResult("doc1", 0.8, "content1", {}, "engine2"),
                            SearchResult("doc3", 0.6, "content3", {}, "engine2")
                        ],
                        engine_id="engine2",
                        timestamp=datetime.now(),
                        total_results=2
                    )
                },
                ground_truth_labels={"doc1": 1.0, "doc2": 0.8, "doc3": 0.6},
                optimal_weights=np.array([0.6, 0.4])
            ),
            TrainingExample(
                query="test query 2",
                engine_results={
                    'engine1': SearchResults(
                        query="test query 2",
                        results=[
                            SearchResult("doc4", 0.8, "content4", {}, "engine1"),
                            SearchResult("doc5", 0.6, "content5", {}, "engine1")
                        ],
                        engine_id="engine1",
                        timestamp=datetime.now(),
                        total_results=2
                    ),
                    'engine2': SearchResults(
                        query="test query 2",
                        results=[
                            SearchResult("doc4", 0.7, "content4", {}, "engine2"),
                            SearchResult("doc6", 0.5, "content6", {}, "engine2")
                        ],
                        engine_id="engine2",
                        timestamp=datetime.now(),
                        total_results=2
                    )
                },
                ground_truth_labels={"doc4": 0.9, "doc5": 0.7, "doc6": 0.5},
                optimal_weights=np.array([0.7, 0.3])
            )
        ]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline.models_dir.exists()
        assert self.pipeline.backup_dir.exists()
        assert self.pipeline.current_model is None
    
    def test_validate_training_data_valid(self):
        """Test validation of valid training data."""
        validation_results = self.pipeline.validate_training_data(self.sample_training_data)
        
        assert validation_results['is_valid']
        assert validation_results['error'] is None
        assert validation_results['statistics']['valid_examples'] == 2
        assert validation_results['statistics']['invalid_examples'] == 0
        assert validation_results['statistics']['unique_queries'] == 2
        assert len(validation_results['statistics']['engine_coverage']) == 2
    
    def test_validate_training_data_empty(self):
        """Test validation of empty training data."""
        validation_results = self.pipeline.validate_training_data([])
        
        assert not validation_results['is_valid']
        assert 'empty' in validation_results['error'].lower()
    
    def test_validate_training_data_invalid_examples(self):
        """Test validation with invalid examples."""
        invalid_data = [
            TrainingExample(
                query="",  # Empty query
                engine_results={},
                ground_truth_labels={},
                optimal_weights=None
            ),
            TrainingExample(
                query="valid query",
                engine_results={
                    'engine1': SearchResults(
                        query="valid query",
                        results=[],  # No results
                        engine_id="engine1",
                        timestamp=datetime.now(),
                        total_results=0
                    )
                },
                ground_truth_labels={"doc1": 1.0},
                optimal_weights=None
            )
        ]
        
        validation_results = self.pipeline.validate_training_data(invalid_data)
        
        assert not validation_results['is_valid']
        assert validation_results['statistics']['valid_examples'] == 0
        assert validation_results['statistics']['invalid_examples'] == 2
    
    def test_prepare_training_data(self):
        """Test training data preparation and splitting."""
        train_data, test_data = self.pipeline.prepare_training_data(
            self.sample_training_data, test_size=0.5
        )
        
        assert len(train_data) >= 1
        assert len(test_data) >= 1
        assert len(train_data) + len(test_data) >= len(self.sample_training_data)
    
    def test_prepare_training_data_single_example(self):
        """Test data preparation with single example."""
        single_example = [self.sample_training_data[0]]
        train_data, test_data = self.pipeline.prepare_training_data(single_example)
        
        # With single example, both train and test should contain it
        assert len(train_data) == 1
        assert len(test_data) == 1
    
    def test_prepare_training_data_invalid(self):
        """Test data preparation with invalid data."""
        invalid_data = [
            TrainingExample(
                query="",
                engine_results={},
                ground_truth_labels={},
                optimal_weights=None
            )
        ]
        
        with pytest.raises(ValueError, match="Training data validation failed"):
            self.pipeline.prepare_training_data(invalid_data)
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        model = WeightPredictorModel()
        
        cv_results = self.pipeline.perform_cross_validation(
            model, self.sample_training_data, cv_folds=2
        )
        
        assert 'cv_mse_mean' in cv_results
        assert 'cv_mae_mean' in cv_results
        assert 'cv_r2_mean' in cv_results
        assert cv_results['cv_folds'] == 2
        assert cv_results['cv_samples'] > 0
    
    def test_cross_validation_limited_data(self):
        """Test cross-validation with limited data."""
        model = WeightPredictorModel()
        single_example = [self.sample_training_data[0]]
        
        cv_results = self.pipeline.perform_cross_validation(
            model, single_example, cv_folds=5
        )
        
        # Should skip cross-validation for insufficient data
        assert cv_results['cv_folds'] == 0
        assert cv_results['cv_samples'] == 1
    
    def test_hyperparameter_search(self):
        """Test hyperparameter search functionality."""
        # Use a small parameter grid for faster testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
        
        best_params, best_scores = self.pipeline.hyperparameter_search(
            self.sample_training_data, param_grid, cv_folds=2
        )
        
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert 'learning_rate' in best_params
        
        assert isinstance(best_scores, dict)
        assert 'best_mse' in best_scores
        assert 'best_mae' in best_scores
        assert 'best_r2' in best_scores
    
    def test_train_model_with_validation_basic(self):
        """Test basic model training with validation."""
        results = self.pipeline.train_model_with_validation(
            self.sample_training_data,
            perform_cv=True,
            perform_hyperparameter_search=False
        )
        
        assert 'training_metrics' in results
        assert 'cv_metrics' in results
        assert 'test_metrics' in results
        assert 'feature_importance' in results
        assert 'data_statistics' in results
        assert 'timestamp' in results
        
        assert self.pipeline.current_model is not None
        assert self.pipeline.current_model.is_trained
    
    def test_train_model_with_hyperparameter_search(self):
        """Test model training with hyperparameter search."""
        results = self.pipeline.train_model_with_validation(
            self.sample_training_data,
            perform_cv=False,
            perform_hyperparameter_search=True
        )
        
        assert 'best_hyperparameters' in results
        assert results['best_hyperparameters'] is not None
        assert self.pipeline.current_model is not None
    
    def test_save_model_version(self):
        """Test saving a model version."""
        # Train a model first
        self.pipeline.train_model_with_validation(self.sample_training_data)
        
        # Save the model
        metadata = {'description': 'Test model', 'accuracy': 0.95}
        model_path = self.pipeline.save_model_version(
            self.pipeline.current_model, 
            'test_v1', 
            metadata
        )
        
        assert Path(model_path).exists()
        
        # Check metadata file
        metadata_path = self.models_dir / 'test_v1' / 'metadata.json'
        assert metadata_path.exists()
        
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata['version_name'] == 'test_v1'
        assert saved_metadata['description'] == 'Test model'
        assert saved_metadata['accuracy'] == 0.95
    
    def test_save_untrained_model(self):
        """Test saving an untrained model."""
        untrained_model = WeightPredictorModel()
        
        with pytest.raises(ValueError, match="Cannot save untrained model"):
            self.pipeline.save_model_version(untrained_model, 'test_v1')
    
    def test_load_model_version(self):
        """Test loading a model version."""
        # Train and save a model first
        self.pipeline.train_model_with_validation(self.sample_training_data)
        original_model = self.pipeline.current_model
        
        metadata = {'description': 'Test model'}
        self.pipeline.save_model_version(original_model, 'test_v1', metadata)
        
        # Load the model
        loaded_model, loaded_metadata = self.pipeline.load_model_version('test_v1')
        
        assert loaded_model.is_trained
        assert loaded_metadata['version_name'] == 'test_v1'
        assert loaded_metadata['description'] == 'Test model'
    
    def test_load_nonexistent_model(self):
        """Test loading a non-existent model version."""
        with pytest.raises(FileNotFoundError, match="Model version 'nonexistent' not found"):
            self.pipeline.load_model_version('nonexistent')
    
    def test_list_model_versions_empty(self):
        """Test listing model versions when none exist."""
        versions = self.pipeline.list_model_versions()
        assert len(versions) == 0
    
    def test_list_model_versions_with_models(self):
        """Test listing model versions with saved models."""
        # Train and save multiple models
        self.pipeline.train_model_with_validation(self.sample_training_data)
        
        self.pipeline.save_model_version(
            self.pipeline.current_model, 'v1', {'accuracy': 0.9}
        )
        self.pipeline.save_model_version(
            self.pipeline.current_model, 'v2', {'accuracy': 0.95}
        )
        
        versions = self.pipeline.list_model_versions()
        
        assert len(versions) == 2
        version_names = [v['version_name'] for v in versions]
        assert 'v1' in version_names
        assert 'v2' in version_names
        
        # Check that versions have required fields
        for version in versions:
            assert 'version_name' in version
            assert 'has_model' in version
            assert 'has_metadata' in version
            assert 'created_at' in version
    
    def test_backup_model_version(self):
        """Test backing up a model version."""
        # Train and save a model
        self.pipeline.train_model_with_validation(self.sample_training_data)
        self.pipeline.save_model_version(self.pipeline.current_model, 'test_v1')
        
        # Create backup
        backup_path = self.pipeline.backup_model_version('test_v1')
        
        assert Path(backup_path).exists()
        assert Path(backup_path).is_dir()
        
        # Check that backup contains model and metadata
        backup_model_path = Path(backup_path) / 'model.pkl'
        backup_metadata_path = Path(backup_path) / 'metadata.json'
        
        assert backup_model_path.exists()
        assert backup_metadata_path.exists()
    
    def test_backup_nonexistent_model(self):
        """Test backing up a non-existent model version."""
        with pytest.raises(FileNotFoundError, match="Model version 'nonexistent' not found"):
            self.pipeline.backup_model_version('nonexistent')
    
    def test_delete_model_version_with_backup(self):
        """Test deleting a model version with backup."""
        # Train and save a model
        self.pipeline.train_model_with_validation(self.sample_training_data)
        self.pipeline.save_model_version(self.pipeline.current_model, 'test_v1')
        
        original_path = self.models_dir / 'test_v1'
        assert original_path.exists()
        
        # Delete with backup
        self.pipeline.delete_model_version('test_v1', create_backup=True)
        
        # Original should be deleted
        assert not original_path.exists()
        
        # Backup should exist
        backup_files = list(self.backup_dir.glob('test_v1_*'))
        assert len(backup_files) > 0
    
    def test_delete_model_version_without_backup(self):
        """Test deleting a model version without backup."""
        # Train and save a model
        self.pipeline.train_model_with_validation(self.sample_training_data)
        self.pipeline.save_model_version(self.pipeline.current_model, 'test_v1')
        
        original_path = self.models_dir / 'test_v1'
        assert original_path.exists()
        
        # Delete without backup
        self.pipeline.delete_model_version('test_v1', create_backup=False)
        
        # Original should be deleted
        assert not original_path.exists()
        
        # No backup should exist
        backup_files = list(self.backup_dir.glob('test_v1_*'))
        assert len(backup_files) == 0
    
    def test_get_best_model_version(self):
        """Test getting the best model version."""
        # Create multiple model versions with different metrics
        self.pipeline.train_model_with_validation(self.sample_training_data)
        
        # Save models with different metadata
        metadata1 = {
            'cv_metrics': {'cv_mse_mean': 0.5, 'cv_mae_mean': 0.3},
            'test_metrics': {'mse': 0.6, 'mae': 0.4}
        }
        metadata2 = {
            'cv_metrics': {'cv_mse_mean': 0.3, 'cv_mae_mean': 0.2},  # Better
            'test_metrics': {'mse': 0.4, 'mae': 0.3}
        }
        
        self.pipeline.save_model_version(self.pipeline.current_model, 'v1', metadata1)
        self.pipeline.save_model_version(self.pipeline.current_model, 'v2', metadata2)
        
        best_version = self.pipeline.get_best_model_version('cv_mse_mean')
        assert best_version == 'v2'  # Should pick the one with lower MSE
    
    def test_get_best_model_version_no_models(self):
        """Test getting best model version when no models exist."""
        best_version = self.pipeline.get_best_model_version()
        assert best_version is None
    
    def test_get_best_model_version_no_metric(self):
        """Test getting best model version when metric doesn't exist."""
        self.pipeline.train_model_with_validation(self.sample_training_data)
        self.pipeline.save_model_version(self.pipeline.current_model, 'v1', {})
        
        best_version = self.pipeline.get_best_model_version('nonexistent_metric')
        assert best_version is None