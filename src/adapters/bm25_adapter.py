"""BM25 search engine adapter implementation."""

import asyncio
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from .base import SearchEngineAdapter, SearchEngineError
from ..models.core import SearchResult, SearchResults
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BM25SearchEngine:
    """BM25 (Best Matching 25) search engine implementation."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 engine.
        
        Args:
            k1: Controls term frequency saturation (default: 1.2)
            b: Controls length normalization (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_frequencies: Dict[str, int] = defaultdict(int)  # Document frequency for each term
        self.doc_lengths: List[int] = []  # Length of each document
        self.avg_doc_length: float = 0.0
        self.term_doc_mapping: Dict[str, List[int]] = defaultdict(list)  # term -> list of doc indices
        self.doc_term_counts: List[Dict[str, int]] = []  # term counts for each document
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if not text:
            return []
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the search index."""
        self.documents = documents
        self.doc_lengths = []
        self.doc_term_counts = []
        self.doc_frequencies = defaultdict(int)
        self.term_doc_mapping = defaultdict(list)
        
        # Process each document
        for doc_idx, doc in enumerate(documents):
            # Combine title and content for indexing
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            tokens = self._tokenize(text)
            
            # Count terms in this document
            term_counts = Counter(tokens)
            self.doc_term_counts.append(term_counts)
            self.doc_lengths.append(len(tokens))
            
            # Update document frequencies and term-document mapping
            for term in term_counts.keys():
                if doc_idx not in self.term_doc_mapping[term]:
                    self.term_doc_mapping[term].append(doc_idx)
                    self.doc_frequencies[term] += 1
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        logger.info(f"BM25 index built with {len(documents)} documents, "
                   f"avg length: {self.avg_doc_length:.1f} tokens")
    
    def search(self, query: str, limit: int = 10) -> List[tuple]:
        """Search documents using BM25 scoring.
        
        Returns:
            List of (doc_index, score) tuples sorted by score descending
        """
        if not query or not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # Calculate BM25 scores for all documents
        doc_scores = defaultdict(float)
        N = len(self.documents)  # Total number of documents
        
        for term in query_terms:
            if term not in self.doc_frequencies:
                continue
                
            df = self.doc_frequencies[term]  # Document frequency
            idf = math.log((N - df + 0.5) / (df + 0.5))  # Inverse document frequency
            
            # Score each document containing this term
            for doc_idx in self.term_doc_mapping[term]:
                tf = self.doc_term_counts[doc_idx][term]  # Term frequency
                doc_length = self.doc_lengths[doc_idx]
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                doc_scores[doc_idx] += idf * (numerator / denominator)
        
        # Sort by score and return top results
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]


class BM25Adapter(SearchEngineAdapter):
    """BM25 search engine adapter."""
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize BM25 adapter.
        
        Args:
            engine_id: Unique identifier for this engine
            config: Configuration dictionary containing:
                - documents: List of documents to index
                - k1: BM25 k1 parameter (optional, default: 1.2)
                - b: BM25 b parameter (optional, default: 0.75)
                - response_delay: Simulated response delay (optional)
            timeout: Search timeout in seconds
        """
        super().__init__(engine_id, timeout)
        
        self.k1 = config.get('k1', 1.2)
        self.b = config.get('b', 0.75)
        self.response_delay = config.get('response_delay', 0.0)
        
        # Initialize BM25 engine
        self.bm25_engine = BM25SearchEngine(k1=self.k1, b=self.b)
        
        # Add documents to index
        documents = config.get('documents', [])
        if documents:
            self.bm25_engine.add_documents(documents)
            logger.info(f"BM25 adapter '{engine_id}' initialized with {len(documents)} documents")
        else:
            logger.warning(f"BM25 adapter '{engine_id}' initialized with no documents")
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Search using BM25 algorithm."""
        start_time = datetime.now()
        
        try:
            # Simulate response delay if configured
            if self.response_delay > 0:
                await asyncio.sleep(self.response_delay)
            
            # Perform BM25 search
            search_results = self.bm25_engine.search(query, limit)
            
            # Convert to SearchResult objects
            results = []
            for doc_idx, score in search_results:
                doc = self.bm25_engine.documents[doc_idx]
                
                result = SearchResult(
                    document_id=doc.get('id', f'doc_{doc_idx}'),
                    relevance_score=float(score),
                    content=doc.get('content', ''),
                    metadata={
                        'title': doc.get('title', ''),
                        'category': doc.get('category', ''),
                        'tags': doc.get('tags', []),
                        'bm25_score': score,
                        'algorithm': 'BM25',
                        'k1': self.k1,
                        'b': self.b
                    },
                    engine_source=self.engine_id
                )
                results.append(result)
            
            return SearchResults(
                query=query,
                results=results,
                engine_id=self.engine_id,
                timestamp=start_time,
                total_results=len(results)
            )
            
        except Exception as e:
            logger.error(f"BM25 search failed for query '{query}': {e}")
            raise SearchEngineError(f"BM25 search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the BM25 engine is healthy."""
        try:
            # Simple health check - verify we have documents and can perform a basic search
            if not self.bm25_engine.documents:
                return False
            
            # Try a simple search
            test_results = self.bm25_engine.search("test", limit=1)
            return True
            
        except Exception as e:
            logger.error(f"BM25 health check failed: {e}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get adapter configuration."""
        return {
            'engine_type': 'bm25',
            'engine_id': self.engine_id,
            'k1': self.k1,
            'b': self.b,
            'document_count': len(self.bm25_engine.documents),
            'avg_doc_length': self.bm25_engine.avg_doc_length,
            'response_delay': self.response_delay
        }