"""Mock search engine adapters for testing and development."""

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import SearchEngineAdapter, SearchEngineError
from ..models.core import SearchResults


logger = logging.getLogger(__name__)


class InMemoryMockAdapter(SearchEngineAdapter):
    """In-memory mock search engine adapter for testing.
    
    Provides a simple in-memory search capability with configurable data
    and response patterns for testing purposes.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the in-memory mock adapter.
        
        Args:
            engine_id: Unique identifier for this mock instance
            config: Configuration dictionary containing:
                - documents: List of mock documents to search
                - response_delay: Artificial delay in seconds (default: 0.1)
                - failure_rate: Probability of random failures (default: 0.0)
                - max_results: Maximum results to return (default: 10)
            timeout: Default timeout for operations in seconds
        """
        super().__init__(engine_id, config, timeout)
        
        # Extract configuration
        self.documents = config.get('documents', self._generate_default_documents())
        self.response_delay = config.get('response_delay', 0.1)
        self.failure_rate = config.get('failure_rate', 0.0)
        self.max_results = config.get('max_results', 10)
        
        # Build search index
        self._build_search_index()
    
    def _generate_default_documents(self) -> List[Dict[str, Any]]:
        """Generate default mock documents for testing.
        
        Returns:
            List of mock documents
        """
        return [
            {
                'id': 'doc1',
                'title': 'Introduction to Machine Learning',
                'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
                'category': 'technology',
                'author': 'John Doe',
                'tags': ['ml', 'ai', 'algorithms']
            },
            {
                'id': 'doc2',
                'title': 'Python Programming Guide',
                'content': 'Python is a versatile programming language used for web development and data science.',
                'category': 'programming',
                'author': 'Jane Smith',
                'tags': ['python', 'programming', 'development']
            },
            {
                'id': 'doc3',
                'title': 'Data Science Fundamentals',
                'content': 'Data science combines statistics, programming, and domain expertise to extract insights.',
                'category': 'data-science',
                'author': 'Bob Johnson',
                'tags': ['data-science', 'statistics', 'analytics']
            },
            {
                'id': 'doc4',
                'title': 'Web Development with FastAPI',
                'content': 'FastAPI is a modern web framework for building APIs with Python.',
                'category': 'web-development',
                'author': 'Alice Brown',
                'tags': ['fastapi', 'web', 'api', 'python']
            },
            {
                'id': 'doc5',
                'title': 'Database Design Principles',
                'content': 'Good database design is crucial for application performance and maintainability.',
                'category': 'database',
                'author': 'Charlie Wilson',
                'tags': ['database', 'design', 'sql']
            }
        ]
    
    def _build_search_index(self) -> None:
        """Build a simple search index from documents."""
        self.search_index = {}
        
        for doc in self.documents:
            doc_id = doc['id']
            # Create searchable text from all string fields
            searchable_text = []
            
            for key, value in doc.items():
                if isinstance(value, str):
                    searchable_text.append(value.lower())
                elif isinstance(value, list):
                    searchable_text.extend([str(item).lower() for item in value])
            
            self.search_index[doc_id] = ' '.join(searchable_text)
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query against the in-memory documents.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters (ignored for mock)
                
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If configured to simulate failures
        """
        # Simulate response delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Simulate random failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            raise SearchEngineError(f"Simulated failure in mock adapter {self.engine_id}")
        
        try:
            # Simple text matching
            query_terms = query.lower().split()
            scored_results = []
            
            for doc in self.documents:
                doc_id = doc['id']
                searchable_text = self.search_index.get(doc_id, '')
                
                # Calculate simple relevance score
                score = self._calculate_relevance_score(query_terms, searchable_text, doc)
                
                if score > 0:
                    content = self._extract_content_from_doc(doc)
                    metadata = self._build_metadata(doc, score)
                    scored_results.append((doc_id, score, content, metadata))
            
            # Sort by score descending
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply limit
            limited_results = scored_results[:min(limit, self.max_results)]
            
            # Normalize results
            normalized_results = self._normalize_result_format(limited_results)
            
            return self._create_search_results(query, normalized_results)
            
        except Exception as e:
            raise SearchEngineError(f"Mock search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the mock adapter is healthy.
        
        Returns:
            True (mock adapters are always healthy unless configured otherwise)
        """
        # Simulate occasional health check failures
        if self.failure_rate > 0 and random.random() < self.failure_rate * 0.5:
            return False
        
        return True
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the mock adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        return {
            'engine_type': 'mock_inmemory',
            'engine_id': self.engine_id,
            'document_count': len(self.documents),
            'response_delay': self.response_delay,
            'failure_rate': self.failure_rate,
            'max_results': self.max_results,
            'timeout': self.timeout
        }
    
    def _calculate_relevance_score(self, query_terms: List[str], searchable_text: str, doc: Dict[str, Any]) -> float:
        """Calculate a simple relevance score for a document.
        
        Args:
            query_terms: List of query terms
            searchable_text: Searchable text for the document
            doc: Original document
            
        Returns:
            Relevance score between 0 and 1
        """
        if not query_terms:
            return 0.0
        
        score = 0.0
        total_terms = len(query_terms)
        
        for term in query_terms:
            if term in searchable_text:
                # Base score for term presence
                score += 1.0
                
                # Bonus for title matches
                if 'title' in doc and term in doc['title'].lower():
                    score += 0.5
                
                # Bonus for exact matches in content
                if 'content' in doc and term in doc['content'].lower():
                    score += 0.3
                
                # Bonus for tag matches
                if 'tags' in doc:
                    for tag in doc['tags']:
                        if term in str(tag).lower():
                            score += 0.2
        
        # Normalize score
        max_possible_score = total_terms * 1.8  # 1.0 + 0.5 + 0.3 bonus
        normalized_score = min(score / max_possible_score, 1.0)
        
        return normalized_score
    
    def _extract_content_from_doc(self, doc: Dict[str, Any]) -> str:
        """Extract content from a mock document.
        
        Args:
            doc: Mock document
            
        Returns:
            Extracted content string
        """
        # Prefer content field, then title, then concatenate fields
        if 'content' in doc and doc['content']:
            return str(doc['content'])
        elif 'title' in doc and doc['title']:
            return str(doc['title'])
        else:
            # Concatenate all string fields
            text_parts = []
            for key, value in doc.items():
                if key != 'id' and isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")
            return ' | '.join(text_parts) if text_parts else f"Document {doc.get('id', 'unknown')}"
    
    def _build_metadata(self, doc: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Build metadata for a search result.
        
        Args:
            doc: Original document
            score: Calculated relevance score
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'raw_score': score,
            'engine_type': 'mock_inmemory',
            'doc_fields': list(doc.keys())
        }
        
        # Add document fields to metadata (excluding content)
        for key, value in doc.items():
            if key not in ['content', 'id']:
                metadata[f'doc_{key}'] = value
        
        return metadata


class FileMockAdapter(SearchEngineAdapter):
    """File-based mock search engine adapter for testing.
    
    Loads documents from JSON files and provides search functionality.
    Useful for testing with larger datasets or specific test scenarios.
    """
    
    def __init__(self, engine_id: str, config: Dict[str, Any], timeout: float = 30.0):
        """Initialize the file-based mock adapter.
        
        Args:
            engine_id: Unique identifier for this mock instance
            config: Configuration dictionary containing:
                - data_file: Path to JSON file containing documents
                - response_delay: Artificial delay in seconds (default: 0.1)
                - failure_rate: Probability of random failures (default: 0.0)
                - max_results: Maximum results to return (default: 10)
            timeout: Default timeout for operations in seconds
        """
        super().__init__(engine_id, config, timeout)
        
        # Extract configuration
        self.data_file = config.get('data_file')
        if not self.data_file:
            raise ValueError("data_file is required for FileMockAdapter")
        
        self.response_delay = config.get('response_delay', 0.1)
        self.failure_rate = config.get('failure_rate', 0.0)
        self.max_results = config.get('max_results', 10)
        
        # Load documents from file
        self._load_documents()
        
        # Build search index
        self._build_search_index()
    
    def _load_documents(self) -> None:
        """Load documents from the specified JSON file."""
        try:
            data_path = Path(self.data_file)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                self.documents = data
            elif isinstance(data, dict) and 'documents' in data:
                self.documents = data['documents']
            else:
                raise ValueError("JSON file must contain a list of documents or a dict with 'documents' key")
            
            logger.info(f"Loaded {len(self.documents)} documents from {self.data_file}")
            
        except Exception as e:
            logger.error(f"Failed to load documents from {self.data_file}: {str(e)}")
            raise SearchEngineError(f"Failed to load mock data: {str(e)}") from e
    
    def _build_search_index(self) -> None:
        """Build a simple search index from loaded documents."""
        self.search_index = {}
        
        for doc in self.documents:
            doc_id = doc.get('id', str(hash(str(doc))))
            # Create searchable text from all string fields
            searchable_text = []
            
            for key, value in doc.items():
                if isinstance(value, str):
                    searchable_text.append(value.lower())
                elif isinstance(value, list):
                    searchable_text.extend([str(item).lower() for item in value])
            
            self.search_index[doc_id] = ' '.join(searchable_text)
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> SearchResults:
        """Execute a search query against the file-based documents.
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            **kwargs: Additional search parameters (ignored for mock)
                
        Returns:
            SearchResults object containing the search results
            
        Raises:
            SearchEngineError: If configured to simulate failures
        """
        # Simulate response delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Simulate random failures
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            raise SearchEngineError(f"Simulated failure in file mock adapter {self.engine_id}")
        
        try:
            # Simple text matching (similar to InMemoryMockAdapter)
            query_terms = query.lower().split()
            scored_results = []
            
            for doc in self.documents:
                doc_id = doc.get('id', str(hash(str(doc))))
                searchable_text = self.search_index.get(doc_id, '')
                
                # Calculate simple relevance score
                score = self._calculate_relevance_score(query_terms, searchable_text, doc)
                
                if score > 0:
                    content = self._extract_content_from_doc(doc)
                    metadata = self._build_metadata(doc, score)
                    scored_results.append((doc_id, score, content, metadata))
            
            # Sort by score descending
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply limit
            limited_results = scored_results[:min(limit, self.max_results)]
            
            # Normalize results
            normalized_results = self._normalize_result_format(limited_results)
            
            return self._create_search_results(query, normalized_results)
            
        except Exception as e:
            raise SearchEngineError(f"File mock search failed: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """Check if the file mock adapter is healthy.
        
        Returns:
            True if data file exists and is readable, False otherwise
        """
        try:
            # Check if data file still exists and is readable
            data_path = Path(self.data_file)
            if not data_path.exists():
                return False
            
            # Simulate occasional health check failures
            if self.failure_rate > 0 and random.random() < self.failure_rate * 0.5:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration of the file mock adapter.
        
        Returns:
            Dictionary containing the adapter configuration
        """
        return {
            'engine_type': 'mock_file',
            'engine_id': self.engine_id,
            'data_file': self.data_file,
            'document_count': len(self.documents),
            'response_delay': self.response_delay,
            'failure_rate': self.failure_rate,
            'max_results': self.max_results,
            'timeout': self.timeout
        }
    
    def _calculate_relevance_score(self, query_terms: List[str], searchable_text: str, doc: Dict[str, Any]) -> float:
        """Calculate a simple relevance score for a document."""
        # Same implementation as InMemoryMockAdapter
        if not query_terms:
            return 0.0
        
        score = 0.0
        total_terms = len(query_terms)
        
        for term in query_terms:
            if term in searchable_text:
                score += 1.0
                
                # Bonus for title matches
                if 'title' in doc and term in str(doc['title']).lower():
                    score += 0.5
                
                # Bonus for exact matches in content
                if 'content' in doc and term in str(doc['content']).lower():
                    score += 0.3
                
                # Bonus for tag matches
                if 'tags' in doc and isinstance(doc['tags'], list):
                    for tag in doc['tags']:
                        if term in str(tag).lower():
                            score += 0.2
        
        # Normalize score
        max_possible_score = total_terms * 1.8
        normalized_score = min(score / max_possible_score, 1.0)
        
        return normalized_score
    
    def _extract_content_from_doc(self, doc: Dict[str, Any]) -> str:
        """Extract content from a document."""
        # Same implementation as InMemoryMockAdapter
        if 'content' in doc and doc['content']:
            return str(doc['content'])
        elif 'title' in doc and doc['title']:
            return str(doc['title'])
        else:
            text_parts = []
            for key, value in doc.items():
                if key != 'id' and isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value}")
            return ' | '.join(text_parts) if text_parts else f"Document {doc.get('id', 'unknown')}"
    
    def _build_metadata(self, doc: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Build metadata for a search result."""
        metadata = {
            'raw_score': score,
            'engine_type': 'mock_file',
            'doc_fields': list(doc.keys()),
            'data_source': self.data_file
        }
        
        # Add document fields to metadata (excluding content)
        for key, value in doc.items():
            if key not in ['content', 'id']:
                metadata[f'doc_{key}'] = value
        
        return metadata