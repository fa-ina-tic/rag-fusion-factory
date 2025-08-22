"""Tests for contrastive learning module."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List

from src.models.contrastive_learning import ContrastiveLearningModule
from src.models.core import (
    SearchResult, SearchResults, TrainingExample, 
    NormalizedResults, TrainingPair
)


class TestContrastiveLearningModule:
    """Test cases for ContrastiveLearningModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'margin': 1.0,
            'temperature': 0.1,
            'negative_sampling_ratio': 3,
            'loss_function': 'triplet'
        }
        self.module = ContrastiveLearningModule(self.config)
        
        # Create sample search results
        self.sample_results = self._create_sample_search_results()
        self.sample_labels = {
            'doc1': 1.0,  # Relevant
            'doc2': 0.8,  # Relevant
            'doc3': 0.2,  # Not relevant
            'doc4': 0.0,  # Not relevant
            'doc5': 0.9   # Relevant
        }
    
    def _create_sample_search_results(self) -> SearchResults:
        """Create sample search results for testing."""
        results = [
            SearchResult(
                document_id='doc1',
                relevance_score=0.9,
                content='Relevant document 1',
                metadata={'source': 'test'},
                engine_source='engine1'
            ),
            SearchResult(
                document_id='doc2',
                relevance_score=0.8,
                content='Relevant document 2',
                metadata={'source': 'test'},
                engine_source='engine1'
            ),
            SearchResult(
                document_id='doc3',
                relevance_score=0.3,
                content='Irrelevant document 3',
                metadata={'source': 'test'},
                engine_source='engine1'
            ),
            SearchResult(
                document_id='doc4',
                relevance_score=0.1,
                content='Irrelevant document 4',
                metadata={'source': 'test'},
                engine_source='engine1'
            ),
            SearchResult(
                document_id='doc5',
                relevance_score=0.85,
                content='Relevant document 5',
                metadata={'source': 'test'},
                engine_source='engine1'
            )
        ]
        
        return SearchResults(
            query='test query',
            results=results,
            engine_id='engine1',
            timestamp=datetime.now(),
            total_results=5
        )
    
    def test_initialization(self):
        """Test module initialization."""
        assert self.module.margin == 1.0
        assert self.module.temperature == 0.1
        assert self.module.negative_sampling_ratio == 3
        assert self.module.loss_function == 'triplet'
    
    def test_default_config(self):
        """Test default configuration."""
        module = ContrastiveLearningModule()
        assert module.margin == 1.0
        assert module.temperature == 0.1
        assert module.negative_sampling_ratio == 5
        assert module.loss_function == 'triplet'
    
    def test_contrastive_loss_triplet(self):
        """Test triplet loss computation."""
        predicted_scores = np.array([0.9, 0.8, 0.3, 0.1, 0.85])
        ground_truth = np.array([1.0, 0.8, 0.2, 0.0, 0.9])
        
        loss = self.module.contrastive_loss(predicted_scores, ground_truth)
        assert isinstance(loss, float)
        assert loss >= 0.0
    
    def test_contrastive_loss_pairwise(self):
        """Test pairwise contrastive loss computation."""
        self.module.loss_function = 'contrastive'
        
        predicted_scores = np.array([0.9, 0.8, 0.3, 0.1])
        ground_truth = np.array([1.0, 0.8, 0.2, 0.0])
        
        loss = self.module.contrastive_loss(predicted_scores, ground_truth)
        assert isinstance(loss, float)
        assert loss >= 0.0
    
    def test_contrastive_loss_ranking(self):
        """Test ranking loss computation."""
        self.module.loss_function = 'ranking'
        
        predicted_scores = np.array([0.9, 0.8, 0.3, 0.1])
        ground_truth = np.array([1.0, 0.8, 0.2, 0.0])
        
        loss = self.module.contrastive_loss(predicted_scores, ground_truth)
        assert isinstance(loss, float)
        assert loss >= 0.0
    
    def test_contrastive_loss_empty_input(self):
        """Test contrastive loss with empty input."""
        predicted_scores = np.array([])
        ground_truth = np.array([])
        
        loss = self.module.contrastive_loss(predicted_scores, ground_truth)
        assert loss == 0.0
    
    def test_contrastive_loss_mismatched_length(self):
        """Test contrastive loss with mismatched input lengths."""
        predicted_scores = np.array([0.9, 0.8])
        ground_truth = np.array([1.0, 0.8, 0.2])
        
        with pytest.raises(ValueError):
            self.module.contrastive_loss(predicted_scores, ground_truth)
    
    def test_generate_training_pairs(self):
        """Test training pair generation."""
        pairs = self.module.generate_training_pairs(self.sample_results, self.sample_labels)
        
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        
        # Check pair structure
        for pair in pairs:
            assert isinstance(pair, TrainingPair)
            assert hasattr(pair, 'positive_result')
            assert hasattr(pair, 'negative_result')
            assert hasattr(pair, 'query')
            assert hasattr(pair, 'label')
            assert pair.query == 'test query'
            assert pair.label in [0.0, 1.0]
    
    def test_generate_training_pairs_empty_input(self):
        """Test training pair generation with empty input."""
        empty_results = SearchResults(
            query='empty query',
            results=[],
            engine_id='engine1',
            timestamp=datetime.now(),
            total_results=0
        )
        
        pairs = self.module.generate_training_pairs(empty_results, {})
        assert pairs == []
    
    def test_generate_training_pairs_from_multiple_engines(self):
        """Test training pair generation from multiple engines."""
        # Create results for multiple engines
        engine_results = {
            'engine1': self.sample_results,
            'engine2': self.sample_results  # Reuse for simplicity
        }
        
        pairs = self.module.generate_training_pairs_from_multiple_engines(
            engine_results, self.sample_labels
        )
        
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        
        # Should have pairs from both engines
        engine1_pairs = [p for p in pairs if p.positive_result.engine_source == 'engine1']
        assert len(engine1_pairs) > 0
    
    def test_process_ground_truth_labels(self):
        """Test ground truth label processing."""
        raw_labels = {
            'doc1': 1,
            'doc2': 0.8,
            'doc3': 'relevant',
            'doc4': 'irrelevant',
            'doc5': True,
            'doc6': False,
            'doc7': '0.5',
            'doc8': 5  # 5-point scale
        }
        
        processed = self.module.process_ground_truth_labels(raw_labels)
        
        assert processed['doc1'] == 1.0
        assert processed['doc2'] == 0.8
        assert processed['doc3'] == 1.0
        assert processed['doc4'] == 0.0
        assert processed['doc5'] == 1.0
        assert processed['doc6'] == 0.0
        assert processed['doc7'] == 0.5
        assert processed['doc8'] == 1.0  # Normalized from 5-point scale
    
    def test_create_batch_training_data(self):
        """Test batch training data creation."""
        # Create training examples
        training_examples = [
            TrainingExample(
                query='test query 1',
                engine_results={'engine1': self.sample_results},
                ground_truth_labels=self.sample_labels
            ),
            TrainingExample(
                query='test query 2',
                engine_results={'engine1': self.sample_results},
                ground_truth_labels=self.sample_labels
            )
        ]
        
        batches = self.module.create_batch_training_data(training_examples, batch_size=5)
        
        assert isinstance(batches, list)
        assert len(batches) > 0
        
        # Check batch structure
        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) <= 5  # Batch size
            for pair in batch:
                assert isinstance(pair, TrainingPair)
    
    def test_validate_training_pairs(self):
        """Test training pair validation."""
        # Create valid pairs
        valid_pairs = self.module.generate_training_pairs(self.sample_results, self.sample_labels)
        
        # Create some invalid pairs
        invalid_pair = TrainingPair(
            positive_result=None,  # Invalid
            negative_result=self.sample_results.results[1],
            query='test',
            label=1.0
        )
        
        all_pairs = valid_pairs + [invalid_pair]
        validated_pairs = self.module.validate_training_pairs(all_pairs)
        
        # Should filter out invalid pairs
        assert len(validated_pairs) == len(valid_pairs)
        assert invalid_pair not in validated_pairs
    
    def test_compute_fused_scores(self):
        """Test fused score computation."""
        # Create normalized results
        normalized_results = {
            'engine1': NormalizedResults(
                query='test query',
                results=self.sample_results.results,
                engine_id='engine1',
                normalization_method='min_max',
                timestamp=datetime.now()
            )
        }
        
        weights = np.array([1.0])  # Single engine
        fused_scores = self.module._compute_fused_scores(weights, normalized_results)
        
        assert isinstance(fused_scores, dict)
        assert len(fused_scores) == len(self.sample_results.results)
        
        # Check that all document IDs are present
        for result in self.sample_results.results:
            assert result.document_id in fused_scores
            assert isinstance(fused_scores[result.document_id], float)
    
    def test_compute_contrastive_loss_for_training(self):
        """Test contrastive loss computation for training."""
        # Create normalized results
        normalized_results = {
            'engine1': NormalizedResults(
                query='test query',
                results=self.sample_results.results,
                engine_id='engine1',
                normalization_method='min_max',
                timestamp=datetime.now()
            )
        }
        
        weights = np.array([1.0])
        loss = self.module.compute_contrastive_loss_for_training(
            weights, normalized_results, self.sample_labels
        )
        
        assert isinstance(loss, float)
        assert loss >= 0.0
    
    def test_compute_optimal_weights_from_ground_truth(self):
        """Test optimal weight computation."""
        # Create normalized results for multiple engines
        normalized_results = {
            'engine1': NormalizedResults(
                query='test query',
                results=self.sample_results.results,
                engine_id='engine1',
                normalization_method='min_max',
                timestamp=datetime.now()
            ),
            'engine2': NormalizedResults(
                query='test query',
                results=self.sample_results.results,
                engine_id='engine2',
                normalization_method='min_max',
                timestamp=datetime.now()
            )
        }
        
        optimal_weights, loss_history = self.module.compute_optimal_weights_from_ground_truth(
            normalized_results, self.sample_labels, max_iterations=10
        )
        
        assert isinstance(optimal_weights, np.ndarray)
        assert len(optimal_weights) == 2  # Two engines
        assert np.isclose(np.sum(optimal_weights), 1.0)  # Weights should sum to 1
        assert np.all(optimal_weights >= 0.0)  # Weights should be non-negative
        
        assert isinstance(loss_history, list)
        assert len(loss_history) > 0
    
    def test_evaluate_ranking_quality(self):
        """Test ranking quality evaluation."""
        predicted_scores = np.array([0.9, 0.8, 0.3, 0.1, 0.85])
        ground_truth = np.array([1.0, 0.8, 0.2, 0.0, 0.9])
        
        metrics = self.module.evaluate_ranking_quality(predicted_scores, ground_truth)
        
        assert isinstance(metrics, dict)
        assert 'spearman_correlation' in metrics
        assert 'kendall_tau' in metrics
        assert 'map' in metrics
        
        # Check for NDCG metrics
        ndcg_keys = [k for k in metrics.keys() if k.startswith('ndcg@')]
        assert len(ndcg_keys) > 0
        
        # All metrics should be floats
        for value in metrics.values():
            assert isinstance(value, float)
    
    def test_evaluate_ranking_quality_empty_input(self):
        """Test ranking quality evaluation with empty input."""
        predicted_scores = np.array([])
        ground_truth = np.array([])
        
        metrics = self.module.evaluate_ranking_quality(predicted_scores, ground_truth)
        assert metrics == {}
    
    def test_evaluate_ranking_quality_mismatched_length(self):
        """Test ranking quality evaluation with mismatched lengths."""
        predicted_scores = np.array([0.9, 0.8])
        ground_truth = np.array([1.0, 0.8, 0.2])
        
        with pytest.raises(ValueError):
            self.module.evaluate_ranking_quality(predicted_scores, ground_truth)


if __name__ == '__main__':
    pytest.main([__file__])