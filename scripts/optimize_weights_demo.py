#!/usr/bin/env python3
"""
Demo script to find optimal weight combinations for BM25 and TF-IDF engines
using dummy data and ground truth labels.
"""

import asyncio
import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.fusion_factory import FusionFactory
from src.adapters.registry import AdapterRegistry
from src.models.core import TrainingExample, SearchResults
from src.models.weight_predictor import WeightPredictorModel
from src.models.contrastive_learning import ContrastiveLearningModule
from src.services.training_pipeline import ModelTrainingPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_dummy_documents() -> List[Dict[str, Any]]:
    """Generate dummy documents for testing."""
    documents = [
        # Machine Learning documents
        {
            'id': 'ml_001',
            'title': 'Introduction to Machine Learning',
            'content': 'Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. It enables computers to learn and improve from experience without being explicitly programmed.',
            'category': 'machine_learning',
            'tags': ['ml', 'ai', 'algorithms', 'statistics']
        },
        {
            'id': 'ml_002', 
            'title': 'Supervised Learning Algorithms',
            'content': 'Supervised learning uses labeled training data to learn a mapping function from input to output. Common algorithms include linear regression, decision trees, and support vector machines.',
            'category': 'machine_learning',
            'tags': ['supervised', 'regression', 'classification', 'svm']
        },
        {
            'id': 'ml_003',
            'title': 'Deep Learning Neural Networks',
            'content': 'Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized computer vision and natural language processing.',
            'category': 'deep_learning',
            'tags': ['deep_learning', 'neural_networks', 'cnn', 'rnn']
        },
        {
            'id': 'ml_004',
            'title': 'Unsupervised Learning Techniques',
            'content': 'Unsupervised learning finds hidden patterns in data without labeled examples. Clustering, dimensionality reduction, and association rules are common techniques.',
            'category': 'machine_learning',
            'tags': ['unsupervised', 'clustering', 'pca', 'kmeans']
        },
        
        # Natural Language Processing documents
        {
            'id': 'nlp_001',
            'title': 'Natural Language Processing Fundamentals',
            'content': 'Natural language processing combines computational linguistics with machine learning to help computers understand human language. It involves tokenization, parsing, and semantic analysis.',
            'category': 'nlp',
            'tags': ['nlp', 'linguistics', 'tokenization', 'parsing']
        },
        {
            'id': 'nlp_002',
            'title': 'Text Classification and Sentiment Analysis',
            'content': 'Text classification assigns categories to documents while sentiment analysis determines emotional tone. Both use machine learning techniques like naive bayes and neural networks.',
            'category': 'nlp',
            'tags': ['classification', 'sentiment', 'naive_bayes', 'text_mining']
        },
        {
            'id': 'nlp_003',
            'title': 'Language Models and Transformers',
            'content': 'Modern language models like BERT and GPT use transformer architecture to understand context and generate human-like text. They have achieved state-of-the-art results in many NLP tasks.',
            'category': 'nlp',
            'tags': ['transformers', 'bert', 'gpt', 'language_models']
        },
        
        # Computer Vision documents
        {
            'id': 'cv_001',
            'title': 'Computer Vision and Image Processing',
            'content': 'Computer vision enables machines to interpret and understand visual information from images and videos. It includes image preprocessing, feature extraction, and object recognition.',
            'category': 'computer_vision',
            'tags': ['computer_vision', 'image_processing', 'opencv', 'features']
        },
        {
            'id': 'cv_002',
            'title': 'Convolutional Neural Networks for Images',
            'content': 'CNNs are specialized neural networks for processing grid-like data such as images. They use convolution operations to detect features like edges, shapes, and textures.',
            'category': 'computer_vision',
            'tags': ['cnn', 'convolution', 'image_recognition', 'deep_learning']
        },
        {
            'id': 'cv_003',
            'title': 'Object Detection and Recognition',
            'content': 'Object detection identifies and locates objects within images. Popular algorithms include YOLO, R-CNN, and SSD which combine classification and localization.',
            'category': 'computer_vision',
            'tags': ['object_detection', 'yolo', 'rcnn', 'localization']
        },
        
        # Data Science documents
        {
            'id': 'ds_001',
            'title': 'Data Science Methodology',
            'content': 'Data science combines statistics, programming, and domain expertise to extract insights from data. The process includes data collection, cleaning, analysis, and visualization.',
            'category': 'data_science',
            'tags': ['data_science', 'statistics', 'visualization', 'analytics']
        },
        {
            'id': 'ds_002',
            'title': 'Statistical Analysis and Hypothesis Testing',
            'content': 'Statistical analysis helps understand data patterns and relationships. Hypothesis testing provides a framework for making decisions based on sample data.',
            'category': 'data_science',
            'tags': ['statistics', 'hypothesis_testing', 'analysis', 'inference']
        },
        {
            'id': 'ds_003',
            'title': 'Data Visualization and Exploratory Analysis',
            'content': 'Data visualization transforms complex datasets into understandable graphics. Exploratory data analysis reveals patterns, outliers, and relationships in data.',
            'category': 'data_science',
            'tags': ['visualization', 'eda', 'matplotlib', 'seaborn']
        },
        
        # Programming documents
        {
            'id': 'prog_001',
            'title': 'Python Programming for Data Science',
            'content': 'Python is a popular programming language for data science with libraries like pandas, numpy, and scikit-learn. It provides powerful tools for data manipulation and analysis.',
            'category': 'programming',
            'tags': ['python', 'pandas', 'numpy', 'scikit_learn']
        },
        {
            'id': 'prog_002',
            'title': 'R Programming for Statistical Computing',
            'content': 'R is a programming language designed for statistical computing and graphics. It offers extensive packages for data analysis, visualization, and statistical modeling.',
            'category': 'programming',
            'tags': ['r', 'statistics', 'ggplot2', 'statistical_computing']
        },
        
        # Algorithms documents
        {
            'id': 'algo_001',
            'title': 'Algorithm Design and Analysis',
            'content': 'Algorithm design involves creating step-by-step procedures to solve computational problems. Analysis evaluates efficiency in terms of time and space complexity.',
            'category': 'algorithms',
            'tags': ['algorithms', 'complexity', 'design', 'analysis']
        },
        {
            'id': 'algo_002',
            'title': 'Sorting and Searching Algorithms',
            'content': 'Sorting algorithms arrange data in order while searching algorithms find specific elements. Common examples include quicksort, mergesort, and binary search.',
            'category': 'algorithms',
            'tags': ['sorting', 'searching', 'quicksort', 'binary_search']
        }
    ]
    
    return documents


def generate_test_queries_with_ground_truth() -> List[Tuple[str, Dict[str, float]]]:
    """Generate test queries with ground truth relevance scores."""
    queries_and_labels = [
        # Machine Learning queries
        (
            "machine learning algorithms",
            {
                'ml_001': 1.0,  # Perfect match
                'ml_002': 0.9,  # High relevance - supervised learning algorithms
                'ml_004': 0.8,  # High relevance - unsupervised learning algorithms
                'ml_003': 0.7,  # Medium relevance - deep learning is ML
                'nlp_002': 0.6,  # Medium relevance - uses ML techniques
                'cv_002': 0.5,  # Medium relevance - CNNs are ML
                'ds_001': 0.4,  # Low relevance - mentions ML indirectly
                'algo_001': 0.3,  # Low relevance - algorithms in general
            }
        ),
        (
            "deep learning neural networks",
            {
                'ml_003': 1.0,  # Perfect match
                'cv_002': 0.9,  # High relevance - CNNs are neural networks
                'nlp_003': 0.8,  # High relevance - transformers use neural networks
                'ml_001': 0.6,  # Medium relevance - mentions AI/ML
                'ml_002': 0.4,  # Low relevance - different ML approach
                'nlp_002': 0.4,  # Low relevance - mentions neural networks
            }
        ),
        (
            "natural language processing",
            {
                'nlp_001': 1.0,  # Perfect match
                'nlp_002': 0.9,  # High relevance - NLP application
                'nlp_003': 0.9,  # High relevance - NLP models
                'ml_003': 0.6,  # Medium relevance - mentions NLP
                'ml_001': 0.4,  # Low relevance - AI/ML context
                'ds_001': 0.3,  # Low relevance - data science context
            }
        ),
        (
            "computer vision image processing",
            {
                'cv_001': 1.0,  # Perfect match
                'cv_002': 0.9,  # High relevance - CNNs for images
                'cv_003': 0.8,  # High relevance - object detection
                'ml_003': 0.6,  # Medium relevance - mentions computer vision
                'ds_003': 0.3,  # Low relevance - visualization (different context)
            }
        ),
        (
            "data science statistics",
            {
                'ds_001': 1.0,  # Perfect match
                'ds_002': 0.9,  # High relevance - statistical analysis
                'ds_003': 0.8,  # High relevance - data analysis
                'prog_002': 0.7,  # Medium relevance - R for statistics
                'ml_001': 0.5,  # Medium relevance - mentions statistics
                'prog_001': 0.4,  # Low relevance - Python for data science
            }
        ),
        (
            "python programming data analysis",
            {
                'prog_001': 1.0,  # Perfect match
                'ds_001': 0.7,  # High relevance - data science methodology
                'ds_003': 0.6,  # Medium relevance - data analysis
                'ml_001': 0.4,  # Low relevance - mentions programming context
                'prog_002': 0.3,  # Low relevance - different language
            }
        ),
        (
            "supervised learning classification",
            {
                'ml_002': 1.0,  # Perfect match
                'nlp_002': 0.8,  # High relevance - text classification
                'ml_001': 0.7,  # High relevance - introduces ML
                'cv_002': 0.6,  # Medium relevance - classification context
                'ml_004': 0.3,  # Low relevance - unsupervised (opposite)
            }
        ),
        (
            "algorithms complexity analysis",
            {
                'algo_001': 1.0,  # Perfect match
                'algo_002': 0.8,  # High relevance - specific algorithms
                'ml_002': 0.5,  # Medium relevance - mentions algorithms
                'ml_001': 0.4,  # Low relevance - mentions algorithms
                'ds_002': 0.3,  # Low relevance - analysis context
            }
        ),
        (
            "transformers language models",
            {
                'nlp_003': 1.0,  # Perfect match
                'nlp_001': 0.6,  # Medium relevance - NLP fundamentals
                'nlp_002': 0.4,  # Low relevance - different NLP technique
                'ml_003': 0.4,  # Low relevance - deep learning context
            }
        ),
        (
            "object detection computer vision",
            {
                'cv_003': 1.0,  # Perfect match
                'cv_001': 0.8,  # High relevance - computer vision fundamentals
                'cv_002': 0.7,  # High relevance - CNNs for vision
                'ml_003': 0.5,  # Medium relevance - mentions computer vision
            }
        )
    ]
    
    return queries_and_labels


async def setup_search_engines(documents: List[Dict[str, Any]]) -> Tuple[AdapterRegistry, FusionFactory]:
    """Set up BM25 and TF-IDF search engines."""
    registry = AdapterRegistry()
    registry._instances.clear()
    
    # Create BM25 adapter
    bm25_config = {
        'documents': documents,
        'k1': 1.2,
        'b': 0.75,
        'response_delay': 0.01
    }
    bm25_adapter = registry.create_adapter('bm25', 'bm25_engine', bm25_config)
    
    # Create TF-IDF adapter
    tfidf_config = {
        'documents': documents,
        'use_log_tf': True,
        'use_cosine_similarity': True,
        'response_delay': 0.01
    }
    tfidf_adapter = registry.create_adapter('tfidf', 'tfidf_engine', tfidf_config)
    
    # Create fusion factory
    factory = FusionFactory(registry)
    
    logger.info(f"Set up search engines with {len(documents)} documents")
    return registry, factory


async def collect_search_results(factory: FusionFactory, registry: AdapterRegistry, 
                                queries_and_labels: List[Tuple[str, Dict[str, float]]]) -> List[TrainingExample]:
    """Collect search results from both engines for training."""
    adapters = list(registry.get_all_adapters().values())
    training_examples = []
    
    for query, ground_truth in queries_and_labels:
        logger.info(f"Processing query: '{query}'")
        
        # Get results from each engine separately
        engine_results = {}
        
        for adapter in adapters:
            try:
                results = await adapter.search(query, limit=10)
                engine_results[adapter.engine_id] = results
                logger.info(f"  {adapter.engine_id}: {len(results.results)} results")
            except Exception as e:
                logger.error(f"Error searching with {adapter.engine_id}: {e}")
                continue
        
        if len(engine_results) >= 2:  # Need at least 2 engines
            training_example = TrainingExample(
                query=query,
                engine_results=engine_results,
                ground_truth_labels=ground_truth
            )
            training_examples.append(training_example)
    
    logger.info(f"Collected {len(training_examples)} training examples")
    return training_examples


def evaluate_weight_combination(weights: np.ndarray, training_examples: List[TrainingExample]) -> float:
    """Evaluate a weight combination using NDCG score."""
    total_ndcg = 0.0
    valid_examples = 0
    
    for example in training_examples:
        # Simulate fusion with given weights
        all_results = {}
        engine_ids = list(example.engine_results.keys())
        
        # Collect all results with weighted scores
        for i, (engine_id, search_results) in enumerate(example.engine_results.items()):
            weight = weights[i] if i < len(weights) else 0.0
            
            for result in search_results.results:
                doc_id = result.document_id
                weighted_score = result.relevance_score * weight
                
                if doc_id not in all_results:
                    all_results[doc_id] = weighted_score
                else:
                    all_results[doc_id] += weighted_score
        
        # Sort by fused score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate NDCG
        ndcg = calculate_ndcg(sorted_results, example.ground_truth_labels)
        if ndcg > 0:
            total_ndcg += ndcg
            valid_examples += 1
    
    return total_ndcg / valid_examples if valid_examples > 0 else 0.0


def calculate_ndcg(ranked_results: List[Tuple[str, float]], ground_truth: Dict[str, float], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG)."""
    if not ranked_results or not ground_truth:
        return 0.0
    
    # Calculate DCG
    dcg = 0.0
    for i, (doc_id, _) in enumerate(ranked_results[:k]):
        if doc_id in ground_truth:
            relevance = ground_truth[doc_id]
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    ideal_relevances = sorted(ground_truth.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances[:k]):
        idcg += relevance / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def grid_search_weights(training_examples: List[TrainingExample], num_engines: int = 2) -> Tuple[np.ndarray, float]:
    """Perform grid search to find optimal weights."""
    best_weights = None
    best_score = -1.0
    
    # Grid search with step size 0.1
    step = 0.1
    weight_combinations = []
    
    # Generate all combinations that sum to 1.0
    for w1 in np.arange(0.0, 1.0 + step, step):
        w2 = 1.0 - w1
        if abs(w2) < 1e-10:  # Handle floating point precision
            w2 = 0.0
        if w2 >= 0.0:
            weights = np.array([w1, w2])
            weight_combinations.append(weights)
    
    logger.info(f"Testing {len(weight_combinations)} weight combinations...")
    
    results = []
    for i, weights in enumerate(weight_combinations):
        score = evaluate_weight_combination(weights, training_examples)
        results.append((weights.copy(), score))
        
        if score > best_score:
            best_score = score
            best_weights = weights.copy()
        
        if i % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(weight_combinations)}, current best: {best_score:.4f}")
    
    logger.info(f"Grid search completed. Best score: {best_score:.4f}")
    return best_weights, best_score, results


async def train_ml_models(training_examples: List[TrainingExample]) -> Dict[str, Any]:
    """Train ML models for weight prediction."""
    logger.info("Training ML models for weight prediction...")
    
    # Train XGBoost weight predictor
    weight_predictor = WeightPredictorModel()
    weight_predictor.train_model(training_examples)
    
    # Train with contrastive learning
    cl_module = ContrastiveLearningModule()
    
    # Use training pipeline for comprehensive training
    pipeline = ModelTrainingPipeline()
    training_results = pipeline.train_model_with_validation(training_examples)
    
    return {
        'weight_predictor': weight_predictor,
        'contrastive_learning': cl_module,
        'training_results': training_results
    }


def visualize_results(grid_results: List[Tuple[np.ndarray, float]], best_weights: np.ndarray, best_score: float):
    """Visualize the weight optimization results."""
    if not PLOTTING_AVAILABLE:
        logger.info("Matplotlib/Seaborn not available, skipping visualization")
        return
    
    try:
        # Extract weights and scores
        w1_values = [weights[0] for weights, _ in grid_results]
        w2_values = [weights[1] for weights, _ in grid_results]
        scores = [score for _, score in grid_results]
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Weight combinations vs scores
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(w1_values, scores, c=scores, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='NDCG Score')
        plt.axvline(x=best_weights[0], color='red', linestyle='--', alpha=0.7, label=f'Best: {best_weights[0]:.2f}')
        plt.xlabel('BM25 Weight')
        plt.ylabel('NDCG Score')
        plt.title('Weight Optimization Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Heatmap of weight combinations
        plt.subplot(1, 2, 2)
        # Create a grid for heatmap
        w1_grid = np.linspace(0, 1, 11)
        w2_grid = np.linspace(0, 1, 11)
        score_grid = np.zeros((len(w2_grid), len(w1_grid)))
        
        for weights, score in grid_results:
            w1_idx = int(round(weights[0] * 10))
            w2_idx = int(round(weights[1] * 10))
            if 0 <= w1_idx < len(w1_grid) and 0 <= w2_idx < len(w2_grid):
                score_grid[w2_idx, w1_idx] = score
        
        sns.heatmap(score_grid, xticklabels=[f'{w:.1f}' for w in w1_grid], 
                   yticklabels=[f'{w:.1f}' for w in w2_grid[::-1]], 
                   annot=True, fmt='.3f', cmap='viridis')
        plt.xlabel('BM25 Weight')
        plt.ylabel('TF-IDF Weight')
        plt.title('Weight Combination Heatmap')
        
        plt.tight_layout()
        plt.savefig('weight_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualization saved as 'weight_optimization_results.png'")
        
    except ImportError:
        logger.warning("Matplotlib/Seaborn not available, skipping visualization")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")


async def main():
    """Main function to run the weight optimization demo."""
    logger.info("Starting BM25 vs TF-IDF weight optimization demo")
    
    # Generate dummy data
    logger.info("Generating dummy documents and queries...")
    documents = generate_dummy_documents()
    queries_and_labels = generate_test_queries_with_ground_truth()
    
    logger.info(f"Generated {len(documents)} documents and {len(queries_and_labels)} test queries")
    
    # Set up search engines
    registry, factory = await setup_search_engines(documents)
    
    # Collect search results for training
    training_examples = await collect_search_results(factory, registry, queries_and_labels)
    
    if not training_examples:
        logger.error("No training examples collected. Exiting.")
        return
    
    # Perform grid search for optimal weights
    logger.info("Performing grid search for optimal weights...")
    best_weights, best_score, grid_results = grid_search_weights(training_examples)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMAL WEIGHT COMBINATION FOUND:")
    logger.info(f"{'='*60}")
    logger.info(f"BM25 Weight:   {best_weights[0]:.3f}")
    logger.info(f"TF-IDF Weight: {best_weights[1]:.3f}")
    logger.info(f"NDCG Score:    {best_score:.4f}")
    logger.info(f"{'='*60}")
    
    # Train ML models for comparison
    ml_results = await train_ml_models(training_examples)
    
    # Test the optimal weights with a few example queries
    logger.info("\nTesting optimal weights with example queries:")
    adapters = list(registry.get_all_adapters().values())
    
    test_queries = ["machine learning algorithms", "deep learning neural networks", "natural language processing"]
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        
        # Get fusion results with optimal weights
        results = await factory.process_query(query, adapters=adapters, weights=best_weights, limit=5)
        
        logger.info(f"Fusion results (BM25: {best_weights[0]:.2f}, TF-IDF: {best_weights[1]:.2f}):")
        for i, result in enumerate(results.results[:3]):
            logger.info(f"  {i+1}. {result.document_id}: {result.relevance_score:.4f}")
    
    # Visualize results
    visualize_results(grid_results, best_weights, best_score)
    
    # Save results
    results_data = {
        'optimal_weights': {
            'bm25': float(best_weights[0]),
            'tfidf': float(best_weights[1])
        },
        'best_ndcg_score': float(best_score),
        'num_documents': len(documents),
        'num_queries': len(queries_and_labels),
        'grid_search_results': [
            {
                'bm25_weight': float(weights[0]),
                'tfidf_weight': float(weights[1]),
                'ndcg_score': float(score)
            }
            for weights, score in grid_results
        ]
    }
    
    with open('weight_optimization_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info("Results saved to 'weight_optimization_results.json'")
    logger.info("Weight optimization demo completed!")


if __name__ == "__main__":
    asyncio.run(main())