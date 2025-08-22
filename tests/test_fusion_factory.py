"""Integration tests for the FusionFactory core engine."""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, List
import numpy as np

from src.models.core import SearchResult, SearchResults, NormalizedResults, RankedResults
from src.services.fusion_factory import (
    FusionFactory, QueryDispatcher, ScoreNormalizer, ResultFusionEngine
)
from src.adapters.base import SearchEngineAdapter
from src.adapters.registry import AdapterRegistry


class MockSearchEngineAdapter(SearchEngineAdapter):
    """Mock search engine adapter for testing."""
    
    def __init__(self, engine_id: str, mock_results: List[SearchResult], 
                 should_fail: bool = False, delay: float = 0.0):
        """Initialize mock adapter.
        
        Args:
            engine_id: Engine identifier
            mock_results: Predefined results to return
            should_fail: Whether to simulate failure
            delay: Artificial delay in seconds
        """
        super().__init__(engine_id, {}, timeout=30.0)
        self.mock_results = mock_results
        self.should_fail = should_fail
        self.delay = delay
        self.search_call_count = 0
        self.health_check_call_count = 0
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Mock search implementation."""
        self.search_call_count += 1
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise Exception(f"Mock failure for engine {self.engine_id}")
        
        # Return limited results
        limited_results = self.mock_results[:limit]
        
        return SearchResults(
            query=query,
            results=limited_results,
            engine_id=self.engine_id,
            timestamp=datetime.now(),
            total_results=len(limited_results)
        )
    
    async def health_check(self) -> bool:
        """Mock health check implementation."""
        self.health_check_call_count += 1
        return not self.should_fail
    
    def get_configuration(self) -> Dict:
        """Mock configuration."""
        return {"engine_id": self.engine_id, "mock": True}


@pytest.fixture
def mock_adapters():
    """Create mock adapters for testing."""
    # Engine A results
    engine_a_results = [
        SearchResult("doc1", 0.9, "Content 1", {"source": "A"}, "engine_a"),
        SearchResult("doc2", 0.8, "Content 2", {"source": "A"}, "engine_a"),
        SearchResult("doc3", 0.7, "Content 3", {"source": "A"}, "engine_a"),
    ]
    
    # Engine B results (some overlap)
    engine_b_results = [
        SearchResult("doc1", 0.85, "Content 1", {"source": "B"}, "engine_b"),
        SearchResult("doc4", 0.75, "Content 4", {"source": "B"}, "engine_b"),
        SearchResult("doc5", 0.65, "Content 5", {"source": "B"}, "engine_b"),
    ]
    
    # Engine C results (different scoring scale)
    engine_c_results = [
        SearchResult("doc2", 95.0, "Content 2", {"source": "C"}, "engine_c"),
        SearchResult("doc6", 90.0, "Content 6", {"source": "C"}, "engine_c"),
        SearchResult("doc7", 85.0, "Content 7", {"source": "C"}, "engine_c"),
    ]
    
    return {
        "engine_a": MockSearchEngineAdapter("engine_a", engine_a_results),
        "engine_b": MockSearchEngineAdapter("engine_b", engine_b_results),
        "engine_c": MockSearchEngineAdapter("engine_c", engine_c_results),
        "failing_engine": MockSearchEngineAdapter("failing_engine", [], should_fail=True),
        "slow_engine": MockSearchEngineAdapter("slow_engine", engine_a_results, delay=0.1)
    }


@pytest.fixture
def adapter_registry(mock_adapters):
    """Create adapter registry with mock adapters."""
    registry = AdapterRegistry()
    
    # Clear existing instances
    registry._instances.clear()
    
    # Add mock adapters
    for engine_id, adapter in mock_adapters.items():
        registry._instances[engine_id] = adapter
    
    return registry


@pytest.fixture
def fusion_factory(adapter_registry):
    """Create fusion factory with mock adapter registry."""
    return FusionFactory(adapter_registry)


class TestQueryDispatcher:
    """Test cases for QueryDispatcher."""
    
    @pytest.mark.asyncio
    async def test_dispatch_query_success(self, mock_adapters):
        """Test successful query dispatch to multiple engines."""
        dispatcher = QueryDispatcher(max_concurrent_engines=5, timeout=10.0)
        
        adapters = [mock_adapters["engine_a"], mock_adapters["engine_b"]]
        results = await dispatcher.dispatch_query("test query", adapters, limit=5)
        
        assert len(results) == 2
        assert "engine_a" in results
        assert "engine_b" in results
        
        # Verify results structure
        for engine_id, search_results in results.items():
            assert isinstance(search_results, SearchResults)
            assert search_results.query == "test query"
            assert len(search_results.results) <= 5
    
    @pytest.mark.asyncio
    async def test_dispatch_query_with_failures(self, mock_adapters):
        """Test query dispatch with some engines failing."""
        dispatcher = QueryDispatcher(max_concurrent_engines=5, timeout=10.0)
        
        adapters = [
            mock_adapters["engine_a"], 
            mock_adapters["failing_engine"],
            mock_adapters["engine_b"]
        ]
        results = await dispatcher.dispatch_query("test query", adapters)
        
        # Should get results from 2 engines (failing_engine should be excluded)
        assert len(results) == 2
        assert "engine_a" in results
        assert "engine_b" in results
        assert "failing_engine" not in results
    
    @pytest.mark.asyncio
    async def test_dispatch_query_empty_adapters(self):
        """Test query dispatch with no adapters."""
        dispatcher = QueryDispatcher()
        results = await dispatcher.dispatch_query("test query", [])
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_dispatch_query_timeout_handling(self, mock_adapters):
        """Test query dispatch with timeout handling."""
        dispatcher = QueryDispatcher(timeout=0.05)  # Very short timeout
        
        adapters = [mock_adapters["slow_engine"]]
        results = await dispatcher.dispatch_query("test query", adapters)
        
        # Slow engine should timeout and not return results
        assert len(results) == 0


class TestScoreNormalizer:
    """Test cases for ScoreNormalizer."""
    
    def test_min_max_normalization(self):
        """Test min-max score normalization."""
        normalizer = ScoreNormalizer(method="min_max")
        
        # Create test results with different score ranges
        results = {
            "engine_a": SearchResults(
                query="test",
                results=[
                    SearchResult("doc1", 0.9, "Content 1", {}, "engine_a"),
                    SearchResult("doc2", 0.5, "Content 2", {}, "engine_a"),
                    SearchResult("doc3", 0.1, "Content 3", {}, "engine_a"),
                ],
                engine_id="engine_a",
                timestamp=datetime.now(),
                total_results=3
            )
        }
        
        normalized = normalizer.normalize_scores(results)
        
        assert len(normalized) == 1
        assert "engine_a" in normalized
        
        scores = [r.relevance_score for r in normalized["engine_a"].results]
        assert max(scores) == 1.0  # Max should be normalized to 1
        assert min(scores) == 0.0  # Min should be normalized to 0
        assert 0.0 <= scores[1] <= 1.0  # Middle value should be between 0 and 1
    
    def test_z_score_normalization(self):
        """Test z-score normalization."""
        normalizer = ScoreNormalizer(method="z_score")
        
        results = {
            "engine_a": SearchResults(
                query="test",
                results=[
                    SearchResult("doc1", 10.0, "Content 1", {}, "engine_a"),
                    SearchResult("doc2", 5.0, "Content 2", {}, "engine_a"),
                    SearchResult("doc3", 0.0, "Content 3", {}, "engine_a"),
                ],
                engine_id="engine_a",
                timestamp=datetime.now(),
                total_results=3
            )
        }
        
        normalized = normalizer.normalize_scores(results)
        scores = [r.relevance_score for r in normalized["engine_a"].results]
        
        # All scores should be between 0 and 1 after sigmoid transformation
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    def test_normalize_empty_results(self):
        """Test normalization with empty results."""
        normalizer = ScoreNormalizer()
        
        results = {}
        normalized = normalizer.normalize_scores(results)
        
        assert len(normalized) == 0
    
    def test_normalize_single_result(self):
        """Test normalization with single result."""
        normalizer = ScoreNormalizer(method="min_max")
        
        results = {
            "engine_a": SearchResults(
                query="test",
                results=[SearchResult("doc1", 0.5, "Content 1", {}, "engine_a")],
                engine_id="engine_a",
                timestamp=datetime.now(),
                total_results=1
            )
        }
        
        normalized = normalizer.normalize_scores(results)
        
        # Single result should be normalized to 1.0
        assert len(normalized["engine_a"].results) == 1
        assert normalized["engine_a"].results[0].relevance_score == 1.0


class TestResultFusionEngine:
    """Test cases for ResultFusionEngine."""
    
    def test_fuse_results_equal_weights(self):
        """Test result fusion with equal weights."""
        fusion_engine = ResultFusionEngine()
        
        # Create normalized results from two engines
        normalized_results = {
            "engine_a": NormalizedResults(
                query="test",
                results=[
                    SearchResult("doc1", 1.0, "Content 1", {}, "engine_a"),
                    SearchResult("doc2", 0.5, "Content 2", {}, "engine_a"),
                ],
                engine_id="engine_a",
                normalization_method="min_max",
                timestamp=datetime.now()
            ),
            "engine_b": NormalizedResults(
                query="test",
                results=[
                    SearchResult("doc1", 0.8, "Content 1", {}, "engine_b"),
                    SearchResult("doc3", 0.6, "Content 3", {}, "engine_b"),
                ],
                engine_id="engine_b",
                normalization_method="min_max",
                timestamp=datetime.now()
            )
        }
        
        fused_results = fusion_engine.fuse_results(normalized_results)
        
        assert isinstance(fused_results, RankedResults)
        assert fused_results.query == "test"
        assert len(fused_results.results) == 3  # doc1, doc2, doc3
        assert len(fused_results.fusion_weights) == 2
        assert np.allclose(fused_results.fusion_weights, [0.5, 0.5])  # Equal weights
        
        # Results should be sorted by fused score (descending)
        scores = [r.relevance_score for r in fused_results.results]
        assert scores == sorted(scores, reverse=True)
        
        # doc1 should have highest score (appears in both engines)
        assert fused_results.results[0].document_id == "doc1"
    
    def test_fuse_results_custom_weights(self):
        """Test result fusion with custom weights."""
        fusion_engine = ResultFusionEngine()
        
        normalized_results = {
            "engine_a": NormalizedResults(
                query="test",
                results=[SearchResult("doc1", 1.0, "Content 1", {}, "engine_a")],
                engine_id="engine_a",
                normalization_method="min_max",
                timestamp=datetime.now()
            ),
            "engine_b": NormalizedResults(
                query="test",
                results=[SearchResult("doc1", 0.5, "Content 1", {}, "engine_b")],
                engine_id="engine_b",
                normalization_method="min_max",
                timestamp=datetime.now()
            )
        }
        
        # Give more weight to engine_a
        weights = np.array([0.8, 0.2])
        fused_results = fusion_engine.fuse_results(normalized_results, weights)
        
        # Fused score should be closer to engine_a's score
        fused_score = fused_results.results[0].relevance_score
        expected_score = 0.8 * 1.0 + 0.2 * 0.5  # 0.9
        assert abs(fused_score - expected_score) < 0.001
    
    def test_fuse_results_empty_input(self):
        """Test result fusion with empty input."""
        fusion_engine = ResultFusionEngine()
        
        fused_results = fusion_engine.fuse_results({})
        
        assert isinstance(fused_results, RankedResults)
        assert len(fused_results.results) == 0
        assert fused_results.total_results == 0


class TestFusionFactory:
    """Integration tests for FusionFactory."""
    
    @pytest.mark.asyncio
    async def test_process_query_success(self, fusion_factory, mock_adapters):
        """Test successful query processing through complete pipeline."""
        adapters = [mock_adapters["engine_a"], mock_adapters["engine_b"]]
        
        results = await fusion_factory.process_query("test query", adapters, limit=5)
        
        assert isinstance(results, RankedResults)
        assert results.query == "test query"
        assert results.total_results > 0
        assert len(results.fusion_weights) == 2
        assert len(results.confidence_scores) == results.total_results
        
        # Verify all results have fusion engine source
        for result in results.results:
            assert result.engine_source == "fusion"
    
    @pytest.mark.asyncio
    async def test_process_query_with_failures(self, fusion_factory, mock_adapters):
        """Test query processing with some engine failures."""
        adapters = [
            mock_adapters["engine_a"], 
            mock_adapters["failing_engine"],
            mock_adapters["engine_b"]
        ]
        
        results = await fusion_factory.process_query("test query", adapters)
        
        # Should still get results from working engines
        assert isinstance(results, RankedResults)
        assert results.total_results > 0
    
    @pytest.mark.asyncio
    async def test_process_query_empty_query(self, fusion_factory, mock_adapters):
        """Test query processing with empty query."""
        adapters = [mock_adapters["engine_a"]]
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await fusion_factory.process_query("", adapters)
    
    @pytest.mark.asyncio
    async def test_process_query_no_adapters(self, fusion_factory):
        """Test query processing with no adapters."""
        with pytest.raises(ValueError, match="No search engine adapters available"):
            await fusion_factory.process_query("test query", [])
    
    @pytest.mark.asyncio
    async def test_process_query_all_engines_fail(self, fusion_factory, mock_adapters):
        """Test query processing when all engines fail."""
        adapters = [mock_adapters["failing_engine"]]
        
        results = await fusion_factory.process_query("test query", adapters)
        
        # Should return empty results
        assert isinstance(results, RankedResults)
        assert results.total_results == 0
        assert len(results.results) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_engines(self, fusion_factory, mock_adapters):
        """Test engine health checking."""
        health_results = await fusion_factory.health_check_engines()
        
        assert isinstance(health_results, dict)
        assert len(health_results) == len(mock_adapters)
        
        # Check specific engine health
        assert health_results["engine_a"] is True
        assert health_results["engine_b"] is True
        assert health_results["failing_engine"] is False
    
    def test_get_engine_status(self, fusion_factory, mock_adapters):
        """Test getting engine status information."""
        status = fusion_factory.get_engine_status()
        
        assert isinstance(status, dict)
        assert "total_instances" in status
        assert "registered_types" in status
        assert status["total_instances"] == len(mock_adapters)
    
    def test_get_available_engines(self, fusion_factory):
        """Test getting available engine types."""
        engines = fusion_factory.get_available_engines()
        
        assert isinstance(engines, list)
        # Should include built-in engine types
        assert "elasticsearch" in engines
        assert "solr" in engines
    
    def test_initialize_adapters_from_config(self, fusion_factory):
        """Test initializing adapters from configuration."""
        config = [
            {
                "engine_type": "elasticsearch",
                "engine_id": "test_es",
                "config": {"host": "localhost", "port": 9200},
                "timeout": 30.0
            }
        ]
        
        adapters = fusion_factory.initialize_adapters_from_config(config)
        
        assert len(adapters) == 1
        assert adapters[0].engine_id == "test_es"
    
    def test_initialize_adapters_invalid_config(self, fusion_factory):
        """Test initializing adapters with invalid configuration."""
        config = [
            {
                "engine_type": "nonexistent",
                "engine_id": "test",
                "config": {}
            }
        ]
        
        with pytest.raises(ValueError):
            fusion_factory.initialize_adapters_from_config(config)


class TestFusionFactoryIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_fusion_pipeline(self, mock_adapters):
        """Test complete fusion pipeline from query to results."""
        # Create fresh registry and factory
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Add mock adapters
        for engine_id, adapter in mock_adapters.items():
            if not adapter.should_fail:  # Only add working engines
                registry._instances[engine_id] = adapter
        
        factory = FusionFactory(registry)
        
        # Process query
        results = await factory.process_query("machine learning", limit=3)
        
        # Verify complete pipeline execution
        assert isinstance(results, RankedResults)
        assert results.query == "machine learning"
        assert results.total_results > 0
        
        # Verify fusion weights sum to 1
        assert abs(np.sum(results.fusion_weights) - 1.0) < 0.001
        
        # Verify results are properly ranked
        scores = [r.relevance_score for r in results.results]
        assert scores == sorted(scores, reverse=True)
        
        # Verify confidence scores
        assert len(results.confidence_scores) == len(results.results)
        assert all(0.0 <= conf <= 1.0 for conf in results.confidence_scores)
    
    @pytest.mark.asyncio
    async def test_fusion_with_different_score_scales(self, mock_adapters):
        """Test fusion with engines using different scoring scales."""
        registry = AdapterRegistry()
        registry._instances.clear()
        
        # Add engines with different score scales
        registry._instances["engine_a"] = mock_adapters["engine_a"]  # 0-1 scale
        registry._instances["engine_c"] = mock_adapters["engine_c"]  # 0-100 scale
        
        factory = FusionFactory(registry)
        results = await factory.process_query("test query")
        
        # Should successfully normalize and fuse different scales
        assert isinstance(results, RankedResults)
        assert results.total_results > 0
        
        # All fused scores should be in reasonable range
        scores = [r.relevance_score for r in results.results]
        assert all(0.0 <= score <= 1.0 for score in scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])