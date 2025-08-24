"""TF-IDF search engine adapter implementation."""

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


class TFIDFSearchEngine:
    """TF-IDF (Term Frequency-Inverse Document Frequency) search engine implementation."""
    
    def __init__(self, use_log_tf: bool = True, use_cosine_similarity: bool = True):
        """Initialize TF-IDF engine.
        
        Args:
            use_log_tf: Whether to use log normalization for term frequency
            use_cosine_similarity: Whether to use cosine similarity for scoring
        """
        self.use_log_tf = use_log_tf
        self.use_cosine_similarity = use_cosine_similarity
        self.documents: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}  # term -> term_id mapping
        self.doc_frequencies: Dict[str, int] = defaultdict(int)  # Document frequency for each term
        self.tf_idf_matrix: List[Dict[int, float]] = []  # TF-IDF vectors for each document
        self.doc_norms: List[float] = []  # Document vector norms for cosine similarity
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if not text:
            return []
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _calculate_tf(self, term_count: int, total_terms: int) -> float:
        """Calculate term frequency."""
        if term_count == 0:
            return 0.0
        
        if self.use_log_tf:
            return 1.0 + math.log(term_count)
        else:
            return term_count / total_terms
    
    def _calculate_idf(self, doc_freq: int, total_docs: int) -> float:
        """Calculate inverse document frequency."""
        return math.log(total_docs / doc_freq)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the search index."""
        self.documents = documents
        self.vocabulary = {}
        self.doc_frequencies = defaultdict(int)
        self.tf_idf_matrix = []
        self.doc_norms = []
        
        # First pass: build vocabulary and document frequencies
        all_doc_terms = []
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            tokens = self._tokenize(text)
            doc_terms = Counter(tokens)
            all_doc_terms.append(doc_terms)
            
            # Update document frequencies
            for term in doc_terms.keys():
                self.doc_frequencies[term] += 1
        
        # Build vocabulary
        all_terms = set()
        for doc_terms in all_doc_terms:
            all_terms.update(doc_terms.keys())
        
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        
        # Second pass: calculate TF-IDF vectors
        N = len(documents)
        for doc_terms in all_doc_terms:
            tf_idf_vector = {}
            total_terms = sum(doc_terms.values())
            
            for term, count in doc_terms.items():
                if term in self.vocabulary:
                    term_id = self.vocabulary[term]
                    tf = self._calculate_tf(count, total_terms)
                    idf = self._calculate_idf(self.doc_frequencies[term], N)
                    tf_idf_vector[term_id] = tf * idf
            
            self.tf_idf_matrix.append(tf_idf_vector)
            
            # Calculate document norm for cosine similarity
            if self.use_cosine_similarity:
                norm = math.sqrt(sum(score ** 2 for score in tf_idf_vector.values()))
                self.doc_norms.append(norm if norm > 0 else 1.0)
            else:
                self.doc_norms.append(1.0)
        
        logger.info(f"TF-IDF index built with {len(documents)} documents, "
                   f"vocabulary size: {len(self.vocabulary)}")
    
    def search(self, query: str, limit: int = 10) -> List[tuple]:
        """Search documents using TF-IDF scoring.
        
        Returns:
            List of (doc_index, score) tuples sorted by score descending
        """
        if not query or not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # Calculate query TF-IDF vector
        query_term_counts = Counter(query_terms)
        query_vector = {}
        total_query_terms = len(query_terms)
        N = len(self.documents)
        
        for term, count in query_term_counts.items():
            if term in self.vocabulary:
                term_id = self.vocabulary[term]
                tf = self._calculate_tf(count, total_query_terms)
                idf = self._calculate_idf(self.doc_frequencies[term], N)
                query_vector[term_id] = tf * idf
        
        if not query_vector:
            return []
        
        # Calculate query norm for cosine similarity
        if self.use_cosine_similarity:
            query_norm = math.sqrt(sum(score ** 2 for score in query_vector.values()))
            if query_norm == 0:
                return []
        else:
            query_norm = 1.0
        
        # Calculate similarity scores with all documents
        doc_scores = []
        for doc_idx, doc_vector in enumerate(self.tf_idf_matrix):
            # Calculate dot product
            dot_product = 0.0
            for term_id, query_score in query_vector.items():
                if term_id in doc_vector:
                    dot_product += query_score * doc_vector[term_id]
            
            if dot_product > 0:
                if self.use_cosine_similarity:
                    # Cosine similarity
                    similarity = dot_product / (query_norm * self.doc_norms[doc_idx])
                else:
                    # Simple dot product
                    similarity = dot_product
                
                doc_scores.append((doc_idx, similarity))
        
        # Sort by score and return top results
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:limit]


class TFIDFAdapter(SearchEngineAdapter):
    """TF-IDF search engine adapter."""
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize TF-IDF adapter.
        
        Args:
            engine_id: Unique identifier for this engine
            config: Configuration dictionary containing:
                - documents: List of documents to index
                - use_log_tf: Whether to use log TF normalization (optional, default: True)
                - use_cosine_similarity: Whether to use cosine similarity (optional, default: True)
                - response_delay: Simulated response delay (optional)
            timeout: Search timeout in seconds
        """
        super().__init__(engine_id, timeout)
        
        self.use_log_tf = config.get('use_log_tf', True)
        self.use_cosine_similarity = config.get('use_cosine_similarity', True)
        self.response_delay = config.get('response_delay', 0.0)
        
        # Initialize TF-IDF engine
        self.tfidf_engine = TFIDFSearchEngine(
            use_log_tf=self.use_log_tf,
            use_cosine_similarity=self.use_cosine_similarity
        )
        
        # Add documents to index
        documents = config.get('documents', [])
        if documents:
            self.tfidf_engine.add_documents(documents)
            logger.info(f"TF-IDF adapter '{engine_id}' initialized with {len(documents)} documents")
        else:
            logger.warning(f"TF-IDF adapter '{engine_id}' initialized with no documents")
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Search using TF-IDF algorithm."""
        start_time = datetime.now()
        
        try:
            # Simulate response delay if configured
            if self.response_delay > 0:
                await asyncio.sleep(self.response_delay)
            
            # Perform TF-IDF search
            search_results = self.tfidf_engine.search(query, limit)
            
            # Convert to SearchResult objects
            results = []
            for doc_idx, score in search_results:
                doc = self.tfidf_engine.documents[doc_idx]
                
                result = SearchResult(
                    document_id=doc.get('id', f'doc_{doc_idx}'),
                    relevance_score=float(score),
                    content=doc.get('content', ''),
                    metadata={
                        'title': doc.get('title', ''),
                        'category': doc.get('category', ''),
                        'tags': doc.get('tags', []),
                        'tfidf_score': score,
                        'algorithm': 'TF-IDF',
                        'use_log_tf': self.use_log_tf,
                        'use_cosine_similarity': self.use_cosine_similarity
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
            logger.error(f"TF-IDF search failed for query '{query}': {e}")
            raise SearchEngineError(f"TF-IDF search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the TF-IDF engine is healthy."""
        try:
            # Simple health check - verify we have documents and can perform a basic search
            if not self.tfidf_engine.documents:
                return False
            
            # Try a simple search
            test_results = self.tfidf_engine.search("test", limit=1)
            return True
            
        except Exception as e:
            logger.error(f"TF-IDF health check failed: {e}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get adapter configuration."""
        return {
            'engine_type': 'tfidf',
            'engine_id': self.engine_id,
            'use_log_tf': self.use_log_tf,
            'use_cosine_similarity': self.use_cosine_similarity,
            'document_count': len(self.tfidf_engine.documents),
            'vocabulary_size': len(self.tfidf_engine.vocabulary),
            'response_delay': self.response_delay
        }