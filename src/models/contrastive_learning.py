"""Contrastive learning module for RAG Fusion Factory."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics.pairwise import cosine_similarity

from .core import SearchResult, SearchResults, TrainingPair, TrainingExample, NormalizedResults


logger = logging.getLogger(__name__)


class ContrastiveLearningModule:
    """Implements contrastive learning for XGBoost weight predictor optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the contrastive learning module.
        
        Args:
            config: Configuration parameters for contrastive learning
        """
        self.config = config or self._get_default_config()
        self.margin = self.config.get('margin', 1.0)
        self.temperature = self.config.get('temperature', 0.1)
        self.negative_sampling_ratio = self.config.get('negative_sampling_ratio', 5)
        self.loss_function = self.config.get('loss_function', 'triplet')
        
        logger.info("ContrastiveLearningModule initialized with config: %s", self.config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for contrastive learning.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'margin': 1.0,
            'temperature': 0.1,
            'negative_sampling_ratio': 5,
            'loss_function': 'triplet'
        }
    
    def contrastive_loss(self, predicted_scores: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute contrastive loss for ranking optimization.
        
        Args:
            predicted_scores: Predicted relevance scores for documents
            ground_truth: Ground truth relevance scores for documents
            
        Returns:
            Contrastive loss value
        """
        if len(predicted_scores) != len(ground_truth):
            raise ValueError("Predicted scores and ground truth must have same length")
        
        if len(predicted_scores) == 0:
            return 0.0
        
        try:
            if self.loss_function == 'triplet':
                return self._triplet_loss(predicted_scores, ground_truth)
            elif self.loss_function == 'contrastive':
                return self._pairwise_contrastive_loss(predicted_scores, ground_truth)
            elif self.loss_function == 'ranking':
                return self._ranking_loss(predicted_scores, ground_truth)
            else:
                logger.warning("Unknown loss function %s, using triplet loss", self.loss_function)
                return self._triplet_loss(predicted_scores, ground_truth)
                
        except Exception as e:
            logger.error("Error computing contrastive loss: %s", e)
            return float('inf')
    
    def _triplet_loss(self, predicted_scores: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute triplet loss for ranking optimization.
        
        Args:
            predicted_scores: Predicted relevance scores
            ground_truth: Ground truth relevance scores
            
        Returns:
            Triplet loss value
        """
        # Find positive and negative examples
        positive_mask = ground_truth > 0.5  # Relevant documents
        negative_mask = ground_truth <= 0.5  # Non-relevant documents
        
        if not np.any(positive_mask) or not np.any(negative_mask):
            return 0.0  # No valid triplets
        
        positive_scores = predicted_scores[positive_mask]
        negative_scores = predicted_scores[negative_mask]
        
        # Compute all possible triplet combinations
        losses = []
        for pos_score in positive_scores:
            for neg_score in negative_scores:
                # Triplet loss: max(0, margin - (pos_score - neg_score))
                loss = max(0.0, self.margin - (pos_score - neg_score))
                losses.append(loss)
        
        if not losses:
            return 0.0
        
        # Return mean triplet loss
        return float(np.mean(losses))
    
    def _pairwise_contrastive_loss(self, predicted_scores: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute pairwise contrastive loss.
        
        Args:
            predicted_scores: Predicted relevance scores
            ground_truth: Ground truth relevance scores
            
        Returns:
            Pairwise contrastive loss value
        """
        n = len(predicted_scores)
        total_loss = 0.0
        pair_count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                pred_i, pred_j = predicted_scores[i], predicted_scores[j]
                gt_i, gt_j = ground_truth[i], ground_truth[j]
                
                # Distance between predictions
                pred_dist = abs(pred_i - pred_j)
                
                # Label: 1 if both relevant or both non-relevant, 0 otherwise
                same_relevance = (gt_i > 0.5) == (gt_j > 0.5)
                
                if same_relevance:
                    # Similar pairs should have small distance
                    loss = pred_dist ** 2
                else:
                    # Dissimilar pairs should have large distance
                    loss = max(0.0, self.margin - pred_dist) ** 2
                
                total_loss += loss
                pair_count += 1
        
        return total_loss / pair_count if pair_count > 0 else 0.0
    
    def _ranking_loss(self, predicted_scores: np.ndarray, ground_truth: np.ndarray) -> float:
        """Compute ranking loss (listwise).
        
        Args:
            predicted_scores: Predicted relevance scores
            ground_truth: Ground truth relevance scores
            
        Returns:
            Ranking loss value
        """
        # Apply temperature scaling and compute softmax
        pred_scaled = predicted_scores / self.temperature
        gt_scaled = ground_truth / self.temperature
        
        # Softmax computation
        pred_exp = np.exp(pred_scaled - np.max(pred_scaled))  # Numerical stability
        pred_probs = pred_exp / np.sum(pred_exp)
        
        gt_exp = np.exp(gt_scaled - np.max(gt_scaled))
        gt_probs = gt_exp / np.sum(gt_exp)
        
        # KL divergence loss
        kl_loss = np.sum(gt_probs * np.log(gt_probs / (pred_probs + 1e-8) + 1e-8))
        
        return float(kl_loss)
    
    def generate_training_pairs(self, results: SearchResults, labels: Dict[str, float]) -> List[TrainingPair]:
        """Generate positive/negative training pairs from ground truth.
        
        Args:
            results: Search results from an engine
            labels: Ground truth labels mapping document_id to relevance score
            
        Returns:
            List of training pairs
        """
        if not results.results or not labels:
            return []
        
        try:
            # Separate relevant and non-relevant documents
            relevant_docs = []
            non_relevant_docs = []
            
            for result in results.results:
                if result.document_id in labels:
                    relevance = labels[result.document_id]
                    if relevance > 0.5:  # Threshold for relevance
                        relevant_docs.append(result)
                    else:
                        non_relevant_docs.append(result)
            
            # Generate positive pairs (relevant-relevant)
            training_pairs = []
            
            # Positive pairs: relevant documents paired together
            for i, doc1 in enumerate(relevant_docs):
                for doc2 in relevant_docs[i+1:]:
                    pair = TrainingPair(
                        positive_result=doc1,
                        negative_result=doc2,
                        query=results.query,
                        label=1.0  # Positive pair
                    )
                    training_pairs.append(pair)
            
            # Negative pairs: relevant vs non-relevant
            for relevant_doc in relevant_docs:
                # Sample negative documents
                num_negatives = min(
                    len(non_relevant_docs),
                    self.negative_sampling_ratio
                )
                
                if num_negatives > 0:
                    # Sample without replacement
                    sampled_negatives = np.random.choice(
                        non_relevant_docs,
                        size=num_negatives,
                        replace=False
                    )
                    
                    for neg_doc in sampled_negatives:
                        pair = TrainingPair(
                            positive_result=relevant_doc,
                            negative_result=neg_doc,
                            query=results.query,
                            label=0.0  # Negative pair
                        )
                        training_pairs.append(pair)
            
            logger.debug("Generated %d training pairs for query: %s", 
                        len(training_pairs), results.query)
            
            return training_pairs
            
        except Exception as e:
            logger.error("Error generating training pairs: %s", e)
            return []
    
    def generate_training_pairs_from_multiple_engines(self, 
                                                    search_results: Dict[str, SearchResults],
                                                    labels: Dict[str, float]) -> List[TrainingPair]:
        """Generate training pairs from multiple search engines.
        
        Args:
            search_results: Dictionary mapping engine_id to SearchResults
            labels: Ground truth labels mapping document_id to relevance score
            
        Returns:
            List of training pairs from all engines
        """
        all_pairs = []
        
        for engine_id, results in search_results.items():
            engine_pairs = self.generate_training_pairs(results, labels)
            all_pairs.extend(engine_pairs)
            
            logger.debug("Generated %d pairs from engine %s", len(engine_pairs), engine_id)
        
        logger.info("Generated total %d training pairs from %d engines", 
                   len(all_pairs), len(search_results))
        
        return all_pairs
    
    def process_ground_truth_labels(self, labels: Dict[str, Any]) -> Dict[str, float]:
        """Process and validate ground truth labels.
        
        Args:
            labels: Raw ground truth labels (may contain various formats)
            
        Returns:
            Processed labels with float relevance scores
        """
        processed_labels = {}
        
        for doc_id, label in labels.items():
            try:
                if isinstance(label, (int, float)):
                    # Normalize to 0-1 range if needed
                    if label > 1.0:
                        processed_labels[doc_id] = min(label / 5.0, 1.0)  # Assume 5-point scale
                    else:
                        processed_labels[doc_id] = float(label)
                elif isinstance(label, str):
                    # Handle string labels
                    label_lower = label.lower()
                    if label_lower in ['relevant', 'true', 'yes', '1']:
                        processed_labels[doc_id] = 1.0
                    elif label_lower in ['irrelevant', 'false', 'no', '0']:
                        processed_labels[doc_id] = 0.0
                    else:
                        # Try to parse as float
                        processed_labels[doc_id] = float(label)
                elif isinstance(label, bool):
                    processed_labels[doc_id] = 1.0 if label else 0.0
                else:
                    logger.warning("Unknown label format for document %s: %s", doc_id, label)
                    continue
                    
            except (ValueError, TypeError) as e:
                logger.warning("Error processing label for document %s: %s", doc_id, e)
                continue
        
        logger.debug("Processed %d ground truth labels", len(processed_labels))
        return processed_labels
    
    def create_batch_training_data(self, 
                                 training_examples: List[TrainingExample],
                                 batch_size: int = 32) -> List[List[TrainingPair]]:
        """Create batches of training pairs for efficient processing.
        
        Args:
            training_examples: List of training examples
            batch_size: Size of each batch
            
        Returns:
            List of batches, each containing training pairs
        """
        all_pairs = []
        
        # Generate pairs from all training examples
        for example in training_examples:
            try:
                # Process ground truth labels
                processed_labels = self.process_ground_truth_labels(example.ground_truth_labels)
                
                # Generate pairs from this example
                example_pairs = self.generate_training_pairs_from_multiple_engines(
                    example.engine_results, processed_labels
                )
                all_pairs.extend(example_pairs)
                
            except Exception as e:
                logger.error("Error processing training example: %s", e)
                continue
        
        # Create batches
        batches = []
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            batches.append(batch)
        
        logger.info("Created %d batches from %d training pairs", len(batches), len(all_pairs))
        return batches
    
    def validate_training_pairs(self, pairs: List[TrainingPair]) -> List[TrainingPair]:
        """Validate and filter training pairs.
        
        Args:
            pairs: List of training pairs to validate
            
        Returns:
            List of valid training pairs
        """
        valid_pairs = []
        
        for pair in pairs:
            try:
                # Check if pair has required attributes
                if not hasattr(pair, 'positive_result') or not hasattr(pair, 'negative_result'):
                    continue
                
                if not hasattr(pair, 'query') or not hasattr(pair, 'label'):
                    continue
                
                # Check if results have required attributes
                pos_result = pair.positive_result
                neg_result = pair.negative_result
                
                if not all(hasattr(pos_result, attr) for attr in ['document_id', 'relevance_score', 'content']):
                    continue
                
                if not all(hasattr(neg_result, attr) for attr in ['document_id', 'relevance_score', 'content']):
                    continue
                
                # Check for valid scores
                if not isinstance(pos_result.relevance_score, (int, float)):
                    continue
                
                if not isinstance(neg_result.relevance_score, (int, float)):
                    continue
                
                # Check for valid label
                if not isinstance(pair.label, (int, float)):
                    continue
                
                # Check for non-empty content
                if not pos_result.content.strip() or not neg_result.content.strip():
                    continue
                
                valid_pairs.append(pair)
                
            except Exception as e:
                logger.warning("Invalid training pair: %s", e)
                continue
        
        logger.debug("Validated %d out of %d training pairs", len(valid_pairs), len(pairs))
        return valid_pairs
    
    def compute_contrastive_loss_for_training(self, 
                                            predicted_weights: np.ndarray,
                                            search_results: Dict[str, NormalizedResults],
                                            ground_truth: Dict[str, float]) -> float:
        """Compute contrastive loss for XGBoost training data preparation.
        
        This method computes the contrastive loss that will be used as the target
        for XGBoost training. The XGBoost model learns to predict weights that
        minimize this contrastive loss.
        
        Args:
            predicted_weights: Predicted weight values for each engine
            search_results: Dictionary mapping engine_id to NormalizedResults
            ground_truth: Ground truth labels mapping document_id to relevance
            
        Returns:
            Contrastive loss value to be used as XGBoost target
        """
        if len(predicted_weights) != len(search_results):
            raise ValueError("Number of weights must match number of engines")
        
        if not ground_truth:
            return 0.0
        
        try:
            # Compute fused scores using predicted weights
            fused_scores = self._compute_fused_scores(predicted_weights, search_results)
            
            if not fused_scores:
                return 0.0
            
            # Extract ground truth for available documents
            available_docs = list(fused_scores.keys())
            gt_scores = np.array([ground_truth.get(doc_id, 0.0) for doc_id in available_docs])
            pred_scores = np.array([fused_scores[doc_id] for doc_id in available_docs])
            
            # Compute contrastive loss
            loss = self.contrastive_loss(pred_scores, gt_scores)
            
            logger.debug("Computed contrastive loss: %.6f for weights: %s", loss, predicted_weights)
            return loss
            
        except Exception as e:
            logger.error("Error computing contrastive loss: %s", e)
            return float('inf')
    
    def _compute_fused_scores(self, weights: np.ndarray, 
                            search_results: Dict[str, NormalizedResults]) -> Dict[str, float]:
        """Compute fused scores using weighted combination.
        
        Args:
            weights: Weight values for each engine
            search_results: Dictionary mapping engine_id to NormalizedResults
            
        Returns:
            Dictionary mapping document_id to fused score
        """
        engine_ids = sorted(search_results.keys())
        
        if len(weights) != len(engine_ids):
            raise ValueError("Weights length must match number of engines")
        
        # Collect all unique documents
        all_documents = set()
        for results in search_results.values():
            for result in results.results:
                all_documents.add(result.document_id)
        
        # Compute fused scores
        fused_scores = {}
        
        for doc_id in all_documents:
            weighted_score = 0.0
            total_weight = 0.0
            
            for i, engine_id in enumerate(engine_ids):
                results = search_results[engine_id]
                
                # Find document score in this engine
                doc_score = None
                for result in results.results:
                    if result.document_id == doc_id:
                        doc_score = result.relevance_score
                        break
                
                if doc_score is not None:
                    weighted_score += weights[i] * doc_score
                    total_weight += weights[i]
            
            # Normalize by total weight (handles missing documents)
            if total_weight > 0:
                fused_scores[doc_id] = weighted_score / total_weight
            else:
                fused_scores[doc_id] = 0.0
        
        return fused_scores
    
    def compute_optimal_weights_from_ground_truth(self, 
                                                search_results: Dict[str, NormalizedResults],
                                                ground_truth: Dict[str, float],
                                                learning_rate: float = 0.01,
                                                max_iterations: int = 100,
                                                tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """Compute optimal weights that minimize contrastive loss for training data generation.
        
        This method finds the optimal weights for a given query and ground truth,
        which will be used as target values for XGBoost training.
        
        Args:
            search_results: Dictionary mapping engine_id to NormalizedResults
            ground_truth: Ground truth labels
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (optimal_weights, loss_history)
        """
        if not search_results:
            return np.array([]), []
        
        # Initialize with equal weights
        num_engines = len(search_results)
        weights = np.ones(num_engines) / num_engines
        loss_history = []
        
        logger.debug("Computing optimal weights for %d engines", num_engines)
        
        for iteration in range(max_iterations):
            try:
                # Compute current loss
                current_loss = self.compute_contrastive_loss_for_training(
                    weights, search_results, ground_truth
                )
                loss_history.append(current_loss)
                
                # Numerical gradient computation
                gradients = np.zeros_like(weights)
                epsilon = 1e-6
                
                for i in range(len(weights)):
                    # Perturb weight slightly
                    weights_plus = weights.copy()
                    weights_plus[i] += epsilon
                    
                    # Normalize to maintain convex combination
                    weights_plus = weights_plus / np.sum(weights_plus)
                    
                    # Compute loss with perturbed weights
                    loss_plus = self.compute_contrastive_loss_for_training(
                        weights_plus, search_results, ground_truth
                    )
                    
                    # Compute gradient
                    gradients[i] = (loss_plus - current_loss) / epsilon
                
                # Update weights
                weights = weights - learning_rate * gradients
                
                # Project to simplex (ensure weights sum to 1 and are non-negative)
                weights = np.maximum(weights, 0.001)  # Minimum weight
                weights = weights / np.sum(weights)
                
                # Check convergence
                if iteration > 0 and abs(loss_history[-2] - current_loss) < tolerance:
                    logger.debug("Converged after %d iterations", iteration + 1)
                    break
                
            except Exception as e:
                logger.error("Error in weight optimization iteration %d: %s", iteration, e)
                break
        
        logger.debug("Optimal weight computation completed. Final loss: %.6f", 
                    loss_history[-1] if loss_history else 0.0)
        return weights, loss_history
    
    def evaluate_ranking_quality(self, predicted_scores: np.ndarray, 
                               ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate ranking quality using various metrics.
        
        Args:
            predicted_scores: Predicted relevance scores
            ground_truth: Ground truth relevance scores
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if len(predicted_scores) != len(ground_truth):
            raise ValueError("Predicted scores and ground truth must have same length")
        
        if len(predicted_scores) == 0:
            return {}
        
        try:
            # Convert to numpy arrays
            pred = np.array(predicted_scores)
            gt = np.array(ground_truth)
            
            # Ranking correlation (Spearman)
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(pred, gt)
            
            # Kendall's tau
            from scipy.stats import kendalltau
            kendall_tau, _ = kendalltau(pred, gt)
            
            # NDCG@k for different k values
            ndcg_scores = {}
            for k in [1, 3, 5, 10]:
                if len(pred) >= k:
                    ndcg_scores[f'ndcg@{k}'] = self._compute_ndcg(pred, gt, k)
            
            # Mean Average Precision (MAP)
            map_score = self._compute_map(pred, gt)
            
            metrics = {
                'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                'kendall_tau': float(kendall_tau) if not np.isnan(kendall_tau) else 0.0,
                'map': map_score,
                **ndcg_scores
            }
            
            return metrics
            
        except Exception as e:
            logger.error("Error evaluating ranking quality: %s", e)
            return {}
    
    def _compute_ndcg(self, predicted_scores: np.ndarray, 
                     ground_truth: np.ndarray, k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at k.
        
        Args:
            predicted_scores: Predicted relevance scores
            ground_truth: Ground truth relevance scores
            k: Number of top results to consider
            
        Returns:
            NDCG@k score
        """
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1][:k]
        
        # DCG calculation
        dcg = 0.0
        for i, idx in enumerate(sorted_indices):
            relevance = ground_truth[idx]
            dcg += (2**relevance - 1) / np.log2(i + 2)
        
        # IDCG calculation (ideal ranking by ground truth)
        ideal_indices = np.argsort(ground_truth)[::-1][:k]
        idcg = 0.0
        for i, idx in enumerate(ideal_indices):
            relevance = ground_truth[idx]
            idcg += (2**relevance - 1) / np.log2(i + 2)
        
        # NDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _compute_map(self, predicted_scores: np.ndarray, 
                    ground_truth: np.ndarray) -> float:
        """Compute Mean Average Precision.
        
        Args:
            predicted_scores: Predicted relevance scores
            ground_truth: Ground truth relevance scores
            
        Returns:
            MAP score
        """
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        
        # Binary relevance (threshold at 0.5)
        relevant_docs = ground_truth > 0.5
        
        if not np.any(relevant_docs):
            return 0.0
        
        # Calculate precision at each relevant document
        precisions = []
        num_relevant_found = 0
        
        for i, idx in enumerate(sorted_indices):
            if relevant_docs[idx]:
                num_relevant_found += 1
                precision = num_relevant_found / (i + 1)
                precisions.append(precision)
        
        # Average precision
        if not precisions:
            return 0.0
        
        return np.mean(precisions)