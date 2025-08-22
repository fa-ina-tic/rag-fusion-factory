"""Tests for WeightPredictorModel."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from src.models.weight_predictor import WeightPredictorModel
from src.models.core import (
    SearchResult, SearchResults, TrainingExample, 
    ModelConfiguration, NormalizedResults
)


class TestWeightPredictorModel:
    """Test cases for WeightPredictorModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = WeightPredictorModel()
        
        # Create sample search results
        self.sample_results = {
            'engine1': NormalizedResults(
                query="test query",
                results=[
                    SearchResult("doc1", 0.9, "content1", {}, "engine1"),
                    SearchResult("doc2", 0.7, "content2", {}, "engine1"),
                    SearchResult("doc3", 0.5, "content3", {}, "engine1")
                ],
                engine_id="engine1",
                normalization_method="min_max",
                timestamp=datetime.now()
            ),
            'engine2': NormalizedResults(
                query="test query",
                results=[
                    SearchResult("doc1", 0.8, "content1", {}, "engine2"),
                    SearchResult("doc4", 0.6, "content4", {}, "engine2"),
                    SearchResult("doc5", 0.4, "content5", {}, "engine2")
                ],
                engine_id="engine2",
                normalization_method="min_max",
                timestamp=datetime.now()
            )
        }
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.model is not None
        assert not self.model.is_trained
        assert self.model.num_engines == 0
        assert len(self.model.feature_names) == 0
    
    def test_default_config(self):
        """Test default configuration."""
        config = self.model._get_default_config()
        assert isinstance(config, ModelConfiguration)
        assert 'n_estimators' in config.xgboost_params
        assert config.normalization_method == 'min_max'
    
    def test_feature_extraction(self):
        """Test feature extraction from search results."""
        features = self.model.extract_features(self.sample_results)
        
        # Should have features for 2 engines + cross-engine features
        expected_features = 2 * 7 + 6  # 7 per engine + 6 cross-engine
        assert len(features) == expected_features
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
    
    def test_feature_extraction_empty_results(self):
        """Test feature extraction with empty results."""
        features = self.model.extract_features({})
        assert len(features) == 0
    
    def test_engine_features_extraction(self):
        """Test extraction of per-engine features."""
        results = self.sample_results['engine1']
        features = self.model._extract_engine_features(results)
        
        assert len(features) == 7
        assert features[0] == 3  # num_results
        assert 0 <= features[1] <= 1  # mean_score
        assert features[2] >= 0  # std_score
    
    def test_cross_engine_features_extraction(self):
        """Test extraction of cross-engine features."""
        features = self.model._extract_cross_engine_features(self.sample_results)
        
        assert len(features) == 6
        assert features[0] >= 0  # total_unique_documents
        assert 0 <= features[1] <= 1  # document_overlap_ratio
    
    def test_predict_weights_untrained(self):
        """Test weight prediction with untrained model."""
        weights = self.model.predict_weights(self.sample_results)
        
        # Should return equal weights for untrained model
        expected_weights = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(weights, expected_weights)
    
    def test_predict_weights_empty_results(self):
        """Test weight prediction with empty results."""
        weights = self.model.predict_weights({})
        assert len(weights) == 0
    
    def test_training_data_preparation(self):
        """Test preparation of training data."""
        # Create sample training data
        training_data = [
            TrainingExample(
                query="test query",
                engine_results={
                    'engine1': SearchResults(
                        query="test query",
                        results=[
                            SearchResult("doc1", 0.9, "content1", {}, "engine1"),
                            SearchResult("doc2", 0.7, "content2", {}, "engine1")
                        ],
                        engine_id="engine1",
                        timestamp=datetime.now(),
                        total_results=2
                    ),
                    'engine2': SearchResults(
                        query="test query",
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
            )
        ]
        
        X, y = self.model._prepare_training_data(training_data)
        
        assert len(X) == 1
        assert len(y) == 1
        assert X.shape[1] > 0  # Should have features
        assert y.shape[1] == 2  # Should have weights for 2 engines
    
    def test_compute_optimal_weights_from_ground_truth(self):
        """Test computation of optimal weights from ground truth."""
        ground_truth = {"doc1": 1.0, "doc2": 0.8, "doc3": 0.6}
        
        weights = self.model._compute_optimal_weights_from_ground_truth(
            self.sample_results, ground_truth
        )
        
        assert weights is not None
        assert len(weights) == 2
        assert np.isclose(np.sum(weights), 1.0)  # Should sum to 1
        assert np.all(weights >= 0)  # Should be non-negative
    
    def test_model_training(self):
        """Test model training."""
        # Create training data with optimal weights
        training_data = [
            TrainingExample(
                query="test query",
                engine_results={
                    'engine1': SearchResults(
                        query="test query",
                        results=[
                            SearchResult("doc1", 0.9, "content1", {}, "engine1"),
                            SearchResult("doc2", 0.7, "content2", {}, "engine1")
                        ],
                        engine_id="engine1",
                        timestamp=datetime.now(),
                        total_results=2
                    ),
                    'engine2': SearchResults(
                        query="test query",
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
            )
        ]
        
        metrics = self.model.train_model(training_data)
        
        assert self.model.is_trained
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert metrics['training_samples'] == 1
    
    def test_model_training_empty_data(self):
        """Test model training with empty data."""
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            self.model.train_model([])
    
    def test_predict_weights_after_training(self):
        """Test weight prediction after training."""
        # Train with minimal data
        training_data = [
            TrainingExample(
                query="test query",
                engine_results={
                    'engine1': SearchResults(
                        query="test query",
                        results=[SearchResult("doc1", 0.9, "content1", {}, "engine1")],
                        engine_id="engine1",
                        timestamp=datetime.now(),
                        total_results=1
                    ),
                    'engine2': SearchResults(
                        query="test query",
                        results=[SearchResult("doc1", 0.8, "content1", {}, "engine2")],
                        engine_id="engine2",
                        timestamp=datetime.now(),
                        total_results=1
                    )
                },
                ground_truth_labels={"doc1": 1.0},
                optimal_weights=np.array([0.7, 0.3])
            )
        ]
        
        self.model.train_model(training_data)
        weights = self.model.predict_weights(self.sample_results)
        
        assert len(weights) == 2
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    
    def test_model_evaluation_untrained(self):
        """Test model evaluation with untrained model."""
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            self.model.evaluate_model([])
    
    def test_feature_importance_untrained(self):
        """Test feature importance with untrained model."""
        with pytest.raises(ValueError, match="Model must be trained to get feature importance"):
            self.model.get_feature_importance()
    
    def test_save_load_model_untrained(self):
        """Test saving untrained model."""
        with pytest.raises(ValueError, match="Cannot save untrained model"):
            self.model.save_model("test_model.pkl")