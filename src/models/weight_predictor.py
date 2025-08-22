"""XGBoost-based weight prediction model for RAG Fusion Factory."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .core import SearchResults, TrainingExample, ModelConfiguration, NormalizedResults


logger = logging.getLogger(__name__)


class WeightPredictorModel:
    """XGBoost-based model that predicts optimal weights for result fusion."""
    
    def __init__(self, config: Optional[ModelConfiguration] = None):
        """Initialize the weight predictor model.
        
        Args:
            config: Model configuration parameters
        """
        self.config = config or self._get_default_config()
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.num_engines: int = 0
        self.is_trained: bool = False
        
        # Initialize XGBoost model
        self._initialize_model()
        
        logger.info("WeightPredictorModel initialized")
    
    def _get_default_config(self) -> ModelConfiguration:
        """Get default model configuration.
        
        Returns:
            Default ModelConfiguration
        """
        return ModelConfiguration(
            xgboost_params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'reg:squarederror',
                'n_jobs': -1
            },
            normalization_method='min_max',
            contrastive_learning_params={
                'margin': 1.0,
                'temperature': 0.1
            },
            hyperparameter_search_space={
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        )
    
    def _initialize_model(self) -> None:
        """Initialize the XGBoost model with configuration parameters."""
        try:
            self.model = xgb.XGBRegressor(**self.config.xgboost_params)
            logger.debug("XGBoost model initialized with parameters: %s", self.config.xgboost_params)
        except Exception as e:
            logger.error("Failed to initialize XGBoost model: %s", e)
            raise
    
    def extract_features(self, search_results: Dict[str, NormalizedResults]) -> np.ndarray:
        """Extract features from search result patterns.
        
        Args:
            search_results: Dictionary mapping engine_id to NormalizedResults
            
        Returns:
            Feature vector as numpy array
        """
        if not search_results:
            return np.array([])
        
        engine_ids = sorted(search_results.keys())
        features = []
        
        # Update feature names if this is the first time or engines changed
        if not self.feature_names or len(engine_ids) != self.num_engines:
            self.num_engines = len(engine_ids)
            self._update_feature_names(engine_ids)
        
        # Extract per-engine features
        for engine_id in engine_ids:
            results = search_results[engine_id]
            engine_features = self._extract_engine_features(results)
            features.extend(engine_features)
        
        # Extract cross-engine features
        cross_engine_features = self._extract_cross_engine_features(search_results)
        features.extend(cross_engine_features)
        
        return np.array(features, dtype=np.float32)
    
    def _update_feature_names(self, engine_ids: List[str]) -> None:
        """Update feature names based on current engines.
        
        Args:
            engine_ids: List of engine identifiers
        """
        self.feature_names = []
        
        # Per-engine features
        for engine_id in engine_ids:
            self.feature_names.extend([
                f"{engine_id}_num_results",
                f"{engine_id}_mean_score",
                f"{engine_id}_std_score",
                f"{engine_id}_max_score",
                f"{engine_id}_min_score",
                f"{engine_id}_score_range",
                f"{engine_id}_score_variance"
            ])
        
        # Cross-engine features
        self.feature_names.extend([
            "total_unique_documents",
            "document_overlap_ratio",
            "score_correlation_mean",
            "score_diversity_index",
            "result_count_variance",
            "query_length"
        ])
        
        logger.debug("Updated feature names: %d features for %d engines", 
                    len(self.feature_names), len(engine_ids))
    
    def _extract_engine_features(self, results: NormalizedResults) -> List[float]:
        """Extract features for a single engine.
        
        Args:
            results: Normalized search results for one engine
            
        Returns:
            List of feature values for the engine
        """
        if not results.results:
            return [0.0] * 7  # Return zeros for all engine features
        
        scores = np.array([r.relevance_score for r in results.results])
        
        features = [
            len(results.results),           # num_results
            float(np.mean(scores)),         # mean_score
            float(np.std(scores)),          # std_score
            float(np.max(scores)),          # max_score
            float(np.min(scores)),          # min_score
            float(np.max(scores) - np.min(scores)),  # score_range
            float(np.var(scores))           # score_variance
        ]
        
        return features
    
    def _extract_cross_engine_features(self, search_results: Dict[str, NormalizedResults]) -> List[float]:
        """Extract features that compare across engines.
        
        Args:
            search_results: Dictionary mapping engine_id to NormalizedResults
            
        Returns:
            List of cross-engine feature values
        """
        if not search_results:
            return [0.0] * 6
        
        # Collect all documents and their scores per engine
        all_documents = set()
        engine_documents = {}
        engine_scores = {}
        result_counts = []
        
        for engine_id, results in search_results.items():
            docs = {r.document_id for r in results.results}
            scores = {r.document_id: r.relevance_score for r in results.results}
            
            all_documents.update(docs)
            engine_documents[engine_id] = docs
            engine_scores[engine_id] = scores
            result_counts.append(len(results.results))
        
        # Calculate cross-engine features
        total_unique_documents = len(all_documents)
        
        # Document overlap ratio
        if len(search_results) > 1 and total_unique_documents > 0:
            overlapping_docs = set.intersection(*engine_documents.values()) if engine_documents else set()
            document_overlap_ratio = len(overlapping_docs) / total_unique_documents
        else:
            document_overlap_ratio = 0.0
        
        # Score correlation (simplified - using overlap documents)
        score_correlations = []
        engine_ids = list(search_results.keys())
        for i in range(len(engine_ids)):
            for j in range(i + 1, len(engine_ids)):
                engine1, engine2 = engine_ids[i], engine_ids[j]
                common_docs = engine_documents[engine1] & engine_documents[engine2]
                
                if len(common_docs) > 1:
                    scores1 = [engine_scores[engine1][doc] for doc in common_docs]
                    scores2 = [engine_scores[engine2][doc] for doc in common_docs]
                    correlation = np.corrcoef(scores1, scores2)[0, 1]
                    if not np.isnan(correlation):
                        score_correlations.append(abs(correlation))
        
        score_correlation_mean = float(np.mean(score_correlations)) if score_correlations else 0.0
        
        # Score diversity index (coefficient of variation across engines)
        if result_counts and np.mean(result_counts) > 0:
            score_diversity_index = float(np.std(result_counts) / np.mean(result_counts))
        else:
            score_diversity_index = 0.0
        
        # Result count variance
        result_count_variance = float(np.var(result_counts)) if result_counts else 0.0
        
        # Query length (from first available result)
        query_length = 0.0
        for results in search_results.values():
            if results.query:
                query_length = float(len(results.query.split()))
                break
        
        features = [
            total_unique_documents,
            document_overlap_ratio,
            score_correlation_mean,
            score_diversity_index,
            result_count_variance,
            query_length
        ]
        
        return features
    
    def predict_weights(self, search_results: Dict[str, NormalizedResults]) -> np.ndarray:
        """Predict optimal weights for result fusion.
        
        Args:
            search_results: Dictionary mapping engine_id to NormalizedResults
            
        Returns:
            Predicted weights as numpy array (sums to 1)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning equal weights")
            num_engines = len(search_results)
            return np.ones(num_engines) / num_engines if num_engines > 0 else np.array([])
        
        if not search_results:
            return np.array([])
        
        try:
            # Extract features
            features = self.extract_features(search_results)
            
            if len(features) == 0:
                logger.warning("No features extracted, returning equal weights")
                num_engines = len(search_results)
                return np.ones(num_engines) / num_engines
            
            # Reshape for prediction
            features = features.reshape(1, -1)
            
            # Predict raw weights
            raw_weights = self.model.predict(features)[0]
            
            # Ensure weights are positive and sum to 1 (convex combination)
            weights = np.maximum(raw_weights, 0.001)  # Minimum weight to avoid zeros
            weights = weights / np.sum(weights)
            
            logger.debug("Predicted weights: %s", weights)
            return weights
            
        except Exception as e:
            logger.error("Weight prediction failed: %s", e)
            # Fallback to equal weights
            num_engines = len(search_results)
            return np.ones(num_engines) / num_engines
    
    def train_model(self, training_data: List[TrainingExample]) -> Dict[str, float]:
        """Train the XGBoost model on training data.
        
        Args:
            training_data: List of training examples
            
        Returns:
            Dictionary containing training metrics
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        logger.info("Training model with %d examples", len(training_data))
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                raise ValueError("No valid training features extracted")
            
            # Train the model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            metrics = {
                'mse': float(mean_squared_error(y, y_pred)),
                'mae': float(mean_absolute_error(y, y_pred)),
                'training_samples': len(X),
                'feature_count': X.shape[1]
            }
            
            logger.info("Model training completed. MSE: %.4f, MAE: %.4f", 
                       metrics['mse'], metrics['mae'])
            
            return metrics
            
        except Exception as e:
            logger.error("Model training failed: %s", e)
            raise
    
    def _prepare_training_data(self, training_data: List[TrainingExample]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for XGBoost.
        
        Args:
            training_data: List of training examples
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        features_list = []
        targets_list = []
        
        for example in training_data:
            try:
                # Convert SearchResults to NormalizedResults for feature extraction
                normalized_results = {}
                for engine_id, search_results in example.engine_results.items():
                    # Create a simple normalized version (assuming already normalized)
                    normalized_results[engine_id] = NormalizedResults(
                        query=search_results.query,
                        results=search_results.results,
                        engine_id=engine_id,
                        normalization_method=self.config.normalization_method,
                        timestamp=search_results.timestamp
                    )
                
                # Extract features
                features = self.extract_features(normalized_results)
                
                if len(features) == 0:
                    continue
                
                # Use optimal weights as target, or compute from ground truth
                if example.optimal_weights is not None:
                    target_weights = example.optimal_weights
                else:
                    # Compute optimal weights based on ground truth (simplified)
                    target_weights = self._compute_optimal_weights_from_ground_truth(
                        normalized_results, example.ground_truth_labels
                    )
                
                if target_weights is not None and len(target_weights) == len(normalized_results):
                    features_list.append(features)
                    targets_list.append(target_weights)
                
            except Exception as e:
                logger.warning("Skipping training example due to error: %s", e)
                continue
        
        if not features_list:
            return np.array([]), np.array([])
        
        X = np.vstack(features_list)
        y = np.vstack(targets_list)
        
        logger.debug("Prepared training data: %d samples, %d features", X.shape[0], X.shape[1])
        return X, y
    
    def _compute_optimal_weights_from_ground_truth(self, 
                                                 normalized_results: Dict[str, NormalizedResults],
                                                 ground_truth: Dict[str, float]) -> Optional[np.ndarray]:
        """Compute optimal weights based on ground truth labels.
        
        This is a simplified implementation that computes weights based on
        how well each engine's results correlate with ground truth.
        
        Args:
            normalized_results: Dictionary mapping engine_id to NormalizedResults
            ground_truth: Dictionary mapping document_id to relevance score
            
        Returns:
            Optimal weights array or None if computation fails
        """
        if not ground_truth:
            return None
        
        engine_ids = sorted(normalized_results.keys())
        engine_scores = []
        
        for engine_id in engine_ids:
            results = normalized_results[engine_id]
            
            # Calculate correlation with ground truth
            engine_relevance = []
            ground_truth_relevance = []
            
            for result in results.results:
                if result.document_id in ground_truth:
                    engine_relevance.append(result.relevance_score)
                    ground_truth_relevance.append(ground_truth[result.document_id])
            
            if len(engine_relevance) > 1:
                correlation = np.corrcoef(engine_relevance, ground_truth_relevance)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            engine_scores.append(max(0.0, correlation))  # Ensure non-negative
        
        # Convert correlations to weights
        if sum(engine_scores) > 0:
            weights = np.array(engine_scores) / sum(engine_scores)
        else:
            # Equal weights if no correlation found
            weights = np.ones(len(engine_ids)) / len(engine_ids)
        
        return weights
    
    def evaluate_model(self, test_data: List[TrainingExample]) -> Dict[str, float]:
        """Evaluate the trained model on test data.
        
        Args:
            test_data: List of test examples
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if not test_data:
            raise ValueError("Test data cannot be empty")
        
        logger.info("Evaluating model with %d test examples", len(test_data))
        
        try:
            # Prepare test data
            X_test, y_test = self._prepare_training_data(test_data)
            
            if len(X_test) == 0:
                raise ValueError("No valid test features extracted")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': float(mean_squared_error(y_test, y_pred)),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'test_samples': len(X_test),
                'feature_count': X_test.shape[1]
            }
            
            # Cross-validation score if enough data
            if len(X_test) >= 5:
                cv_scores = cross_val_score(self.model, X_test, y_test, 
                                          cv=min(5, len(X_test)), 
                                          scoring='neg_mean_squared_error')
                metrics['cv_mse_mean'] = float(-np.mean(cv_scores))
                metrics['cv_mse_std'] = float(np.std(cv_scores))
            
            logger.info("Model evaluation completed. Test MSE: %.4f, Test MAE: %.4f", 
                       metrics['mse'], metrics['mae'])
            
            return metrics
            
        except Exception as e:
            logger.error("Model evaluation failed: %s", e)
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            model_data = {
                'model': self.model,
                'config': self.config,
                'feature_names': self.feature_names,
                'num_engines': self.num_engines,
                'is_trained': self.is_trained
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved to %s", filepath)
            
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.num_engines = model_data['num_engines']
            self.is_trained = model_data['is_trained']
            
            logger.info("Model loaded from %s", filepath)
            
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        try:
            importance_scores = self.model.feature_importances_
            
            if len(self.feature_names) != len(importance_scores):
                logger.warning("Feature names and importance scores length mismatch")
                return {}
            
            importance_dict = dict(zip(self.feature_names, importance_scores))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error("Failed to get feature importance: %s", e)
            return {}