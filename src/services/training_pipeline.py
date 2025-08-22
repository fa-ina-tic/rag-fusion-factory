"""Training pipeline for the WeightPredictorModel."""

import logging
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..models.core import TrainingExample, ModelConfiguration
from ..models.weight_predictor import WeightPredictorModel


logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Pipeline for training and managing WeightPredictorModel instances."""
    
    def __init__(self, models_dir: str = "models", backup_dir: str = "models/backups"):
        """Initialize the training pipeline.
        
        Args:
            models_dir: Directory to store trained models
            backup_dir: Directory to store model backups
        """
        self.models_dir = Path(models_dir)
        self.backup_dir = Path(backup_dir)
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model: Optional[WeightPredictorModel] = None
        self.model_metadata: Dict[str, Any] = {}
        
        logger.info("ModelTrainingPipeline initialized")
    
    def validate_training_data(self, training_data: List[TrainingExample]) -> Dict[str, Any]:
        """Validate training data quality and completeness.
        
        Args:
            training_data: List of training examples to validate
            
        Returns:
            Dictionary containing validation results and statistics
        """
        if not training_data:
            return {
                'is_valid': False,
                'error': 'Training data is empty',
                'statistics': {}
            }
        
        logger.info("Validating %d training examples", len(training_data))
        
        validation_results = {
            'is_valid': True,
            'error': None,
            'statistics': {
                'total_examples': len(training_data),
                'valid_examples': 0,
                'invalid_examples': 0,
                'unique_queries': set(),
                'engine_coverage': {},
                'ground_truth_coverage': 0,
                'optimal_weights_coverage': 0
            }
        }
        
        invalid_reasons = []
        
        for i, example in enumerate(training_data):
            try:
                # Check basic structure
                if not example.query or not example.query.strip():
                    invalid_reasons.append(f"Example {i}: Empty query")
                    continue
                
                if not example.engine_results:
                    invalid_reasons.append(f"Example {i}: No engine results")
                    continue
                
                if not example.ground_truth_labels:
                    invalid_reasons.append(f"Example {i}: No ground truth labels")
                    continue
                
                # Check engine results
                valid_engines = 0
                for engine_id, search_results in example.engine_results.items():
                    if search_results.results:
                        valid_engines += 1
                        
                        # Track engine coverage
                        if engine_id not in validation_results['statistics']['engine_coverage']:
                            validation_results['statistics']['engine_coverage'][engine_id] = 0
                        validation_results['statistics']['engine_coverage'][engine_id] += 1
                
                if valid_engines < 2:
                    invalid_reasons.append(f"Example {i}: Less than 2 engines with results")
                    continue
                
                # Check ground truth quality
                if len(example.ground_truth_labels) == 0:
                    invalid_reasons.append(f"Example {i}: Empty ground truth labels")
                    continue
                
                # Valid example
                validation_results['statistics']['valid_examples'] += 1
                validation_results['statistics']['unique_queries'].add(example.query)
                
                if example.ground_truth_labels:
                    validation_results['statistics']['ground_truth_coverage'] += 1
                
                if example.optimal_weights is not None:
                    validation_results['statistics']['optimal_weights_coverage'] += 1
                
            except Exception as e:
                invalid_reasons.append(f"Example {i}: Exception during validation: {e}")
                continue
        
        validation_results['statistics']['invalid_examples'] = (
            len(training_data) - validation_results['statistics']['valid_examples']
        )
        validation_results['statistics']['unique_queries'] = len(validation_results['statistics']['unique_queries'])
        
        # Check if we have enough valid data
        min_examples = 1  # Minimum required examples (allow single example for testing)
        if validation_results['statistics']['valid_examples'] < min_examples:
            validation_results['is_valid'] = False
            validation_results['error'] = f"Insufficient valid examples: {validation_results['statistics']['valid_examples']} < {min_examples}"
        
        # Check engine diversity
        if len(validation_results['statistics']['engine_coverage']) < 2:
            validation_results['is_valid'] = False
            validation_results['error'] = "Need at least 2 different engines in training data"
        
        if invalid_reasons:
            validation_results['invalid_reasons'] = invalid_reasons[:10]  # Limit to first 10
        
        logger.info("Validation completed: %d valid, %d invalid examples", 
                   validation_results['statistics']['valid_examples'],
                   validation_results['statistics']['invalid_examples'])
        
        return validation_results
    
    def prepare_training_data(self, training_data: List[TrainingExample], 
                            test_size: float = 0.2, 
                            random_state: int = 42) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """Prepare and split training data.
        
        Args:
            training_data: List of training examples
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Validate data first
        validation_results = self.validate_training_data(training_data)
        if not validation_results['is_valid']:
            raise ValueError(f"Training data validation failed: {validation_results['error']}")
        
        # Filter out invalid examples
        valid_examples = []
        for example in training_data:
            try:
                # Basic validation
                if (example.query and example.engine_results and 
                    example.ground_truth_labels and 
                    len(example.engine_results) >= 2):
                    valid_examples.append(example)
            except Exception as e:
                logger.warning("Skipping invalid example: %s", e)
                continue
        
        if len(valid_examples) < 1:
            raise ValueError("Need at least 1 valid example for training")
        
        # Split data
        if len(valid_examples) == 1:
            # If only one example, use it for both train and test
            train_data = valid_examples
            test_data = valid_examples
        else:
            train_data, test_data = train_test_split(
                valid_examples, 
                test_size=test_size, 
                random_state=random_state
            )
        
        logger.info("Data split: %d train, %d test examples", len(train_data), len(test_data))
        return train_data, test_data
    
    def perform_cross_validation(self, model: WeightPredictorModel, 
                               training_data: List[TrainingExample],
                               cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on the model.
        
        Args:
            model: WeightPredictorModel instance
            training_data: List of training examples
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation metrics
        """
        if len(training_data) < cv_folds:
            cv_folds = max(2, len(training_data))  # Ensure at least 2 folds
            logger.warning("Reducing CV folds to %d due to limited data", cv_folds)
        
        # Skip cross-validation if we have too little data
        if len(training_data) < 2:
            logger.warning("Skipping cross-validation due to insufficient data")
            return {
                'cv_mse_mean': 0.0,
                'cv_mse_std': 0.0,
                'cv_mae_mean': 0.0,
                'cv_mae_std': 0.0,
                'cv_r2_mean': 0.0,
                'cv_r2_std': 0.0,
                'cv_folds': 0,
                'cv_samples': len(training_data)
            }
        
        logger.info("Performing %d-fold cross-validation", cv_folds)
        
        try:
            # Prepare data for cross-validation
            X, y = model._prepare_training_data(training_data)
            
            if len(X) == 0:
                raise ValueError("No valid features extracted for cross-validation")
            
            # Perform cross-validation
            cv_scores_mse = cross_val_score(
                model.model, X, y, 
                cv=cv_folds, 
                scoring='neg_mean_squared_error'
            )
            
            cv_scores_mae = cross_val_score(
                model.model, X, y, 
                cv=cv_folds, 
                scoring='neg_mean_absolute_error'
            )
            
            cv_scores_r2 = cross_val_score(
                model.model, X, y, 
                cv=cv_folds, 
                scoring='r2'
            )
            
            cv_results = {
                'cv_mse_mean': float(-np.mean(cv_scores_mse)),
                'cv_mse_std': float(np.std(cv_scores_mse)),
                'cv_mae_mean': float(-np.mean(cv_scores_mae)),
                'cv_mae_std': float(np.std(cv_scores_mae)),
                'cv_r2_mean': float(np.mean(cv_scores_r2)),
                'cv_r2_std': float(np.std(cv_scores_r2)),
                'cv_folds': cv_folds,
                'cv_samples': len(X)
            }
            
            logger.info("Cross-validation completed: MSE=%.4f±%.4f, MAE=%.4f±%.4f, R²=%.4f±%.4f",
                       cv_results['cv_mse_mean'], cv_results['cv_mse_std'],
                       cv_results['cv_mae_mean'], cv_results['cv_mae_std'],
                       cv_results['cv_r2_mean'], cv_results['cv_r2_std'])
            
            return cv_results
            
        except Exception as e:
            logger.error("Cross-validation failed: %s", e)
            raise
    
    def hyperparameter_search(self, training_data: List[TrainingExample],
                            param_grid: Optional[Dict[str, List[Any]]] = None,
                            cv_folds: int = 3) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Perform hyperparameter search using GridSearchCV.
        
        Args:
            training_data: List of training examples
            param_grid: Parameter grid for search (uses default if None)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (best_params, best_scores)
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        
        logger.info("Starting hyperparameter search with %d parameter combinations", 
                   np.prod([len(v) for v in param_grid.values()]))
        
        try:
            # Create a temporary model for hyperparameter search
            temp_model = WeightPredictorModel()
            X, y = temp_model._prepare_training_data(training_data)
            
            if len(X) == 0:
                raise ValueError("No valid features extracted for hyperparameter search")
            
            # Adjust CV folds if necessary
            if len(X) < cv_folds:
                cv_folds = max(2, len(X))
                logger.warning("Reducing CV folds to %d due to limited data", cv_folds)
            
            # Skip hyperparameter search if insufficient data
            if len(X) < 2:
                logger.warning("Skipping hyperparameter search due to insufficient data")
                return {}, {'best_mse': float('inf'), 'best_mae': float('inf'), 'best_r2': 0.0}
            
            # Perform grid search
            grid_search = GridSearchCV(
                temp_model.model,
                param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive MSE
            
            # Get additional metrics for best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X)
            
            best_scores = {
                'best_mse': float(best_score),
                'best_mae': float(mean_absolute_error(y, y_pred)),
                'best_r2': float(r2_score(y, y_pred)),
                'cv_folds': cv_folds,
                'total_combinations': len(grid_search.cv_results_['params'])
            }
            
            logger.info("Hyperparameter search completed. Best MSE: %.4f", best_score)
            logger.info("Best parameters: %s", best_params)
            
            return best_params, best_scores
            
        except Exception as e:
            logger.error("Hyperparameter search failed: %s", e)
            raise
    
    def train_model_with_validation(self, training_data: List[TrainingExample],
                                  config: Optional[ModelConfiguration] = None,
                                  perform_cv: bool = True,
                                  perform_hyperparameter_search: bool = False) -> Dict[str, Any]:
        """Train a model with comprehensive validation.
        
        Args:
            training_data: List of training examples
            config: Model configuration (uses default if None)
            perform_cv: Whether to perform cross-validation
            perform_hyperparameter_search: Whether to perform hyperparameter search
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting model training with validation")
        
        # Validate and prepare data
        train_data, test_data = self.prepare_training_data(training_data)
        
        # Perform hyperparameter search if requested
        best_params = None
        if perform_hyperparameter_search:
            best_params, search_scores = self.hyperparameter_search(train_data)
            
            # Update config with best parameters
            if config is None:
                config = ModelConfiguration(
                    xgboost_params=best_params,
                    normalization_method='min_max',
                    contrastive_learning_params={'margin': 1.0, 'temperature': 0.1},
                    hyperparameter_search_space={}
                )
            else:
                config.xgboost_params.update(best_params)
        
        # Create and train model
        model = WeightPredictorModel(config)
        training_metrics = model.train_model(train_data)
        
        # Perform cross-validation if requested
        cv_metrics = {}
        if perform_cv and len(train_data) > 1:
            try:
                cv_metrics = self.perform_cross_validation(model, train_data)
            except Exception as e:
                logger.warning("Cross-validation failed: %s", e)
        
        # Evaluate on test data
        test_metrics = {}
        if test_data and len(test_data) > 0:
            try:
                test_metrics = model.evaluate_model(test_data)
            except Exception as e:
                logger.warning("Test evaluation failed: %s", e)
        
        # Get feature importance
        feature_importance = {}
        try:
            feature_importance = model.get_feature_importance()
        except Exception as e:
            logger.warning("Feature importance extraction failed: %s", e)
        
        # Compile results
        results = {
            'training_metrics': training_metrics,
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model_config': config.__dict__ if config else None,
            'best_hyperparameters': best_params,
            'data_statistics': {
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'total_samples': len(training_data)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Store the trained model
        self.current_model = model
        
        logger.info("Model training completed successfully")
        return results
    
    def save_model_version(self, model: WeightPredictorModel, 
                          version_name: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a model version with metadata.
        
        Args:
            model: Trained WeightPredictorModel to save
            version_name: Name/identifier for this model version
            metadata: Optional metadata to store with the model
            
        Returns:
            Path to the saved model file
        """
        if not model.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create version directory
        version_dir = self.models_dir / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        model.save_model(str(model_path))
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'version_name': version_name,
            'saved_at': datetime.now().isoformat(),
            'model_type': 'WeightPredictorModel',
            'is_trained': model.is_trained,
            'num_engines': model.num_engines,
            'feature_count': len(model.feature_names)
        })
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model version '%s' saved to %s", version_name, version_dir)
        return str(model_path)
    
    def load_model_version(self, version_name: str) -> Tuple[WeightPredictorModel, Dict[str, Any]]:
        """Load a specific model version.
        
        Args:
            version_name: Name/identifier of the model version to load
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        version_dir = self.models_dir / version_name
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version '{version_name}' not found")
        
        # Load model
        model_path = version_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in version '{version_name}'")
        
        model = WeightPredictorModel()
        model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info("Model version '%s' loaded successfully", version_name)
        return model, metadata
    
    def list_model_versions(self) -> List[Dict[str, Any]]:
        """List all available model versions.
        
        Returns:
            List of dictionaries containing version information
        """
        versions = []
        
        if not self.models_dir.exists():
            return versions
        
        for version_dir in self.models_dir.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                model_path = version_dir / "model.pkl"
                
                version_info = {
                    'version_name': version_dir.name,
                    'has_model': model_path.exists(),
                    'has_metadata': metadata_path.exists(),
                    'created_at': datetime.fromtimestamp(version_dir.stat().st_ctime).isoformat()
                }
                
                # Load metadata if available
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        version_info.update(metadata)
                    except Exception as e:
                        logger.warning("Failed to load metadata for version %s: %s", 
                                     version_dir.name, e)
                
                versions.append(version_info)
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x.get('saved_at', x['created_at']), reverse=True)
        
        return versions
    
    def backup_model_version(self, version_name: str) -> str:
        """Create a backup of a model version.
        
        Args:
            version_name: Name of the version to backup
            
        Returns:
            Path to the backup directory
        """
        version_dir = self.models_dir / version_name
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version '{version_name}' not found")
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{version_name}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        # Copy the entire version directory
        shutil.copytree(version_dir, backup_path)
        
        logger.info("Model version '%s' backed up to %s", version_name, backup_path)
        return str(backup_path)
    
    def delete_model_version(self, version_name: str, create_backup: bool = True) -> None:
        """Delete a model version.
        
        Args:
            version_name: Name of the version to delete
            create_backup: Whether to create a backup before deletion
        """
        version_dir = self.models_dir / version_name
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version '{version_name}' not found")
        
        # Create backup if requested
        if create_backup:
            self.backup_model_version(version_name)
        
        # Delete the version directory
        shutil.rmtree(version_dir)
        
        logger.info("Model version '%s' deleted", version_name)
    
    def get_best_model_version(self, metric: str = 'cv_mse_mean') -> Optional[str]:
        """Get the name of the best model version based on a metric.
        
        Args:
            metric: Metric to use for comparison (lower is better for MSE/MAE)
            
        Returns:
            Name of the best model version or None if no versions found
        """
        versions = self.list_model_versions()
        
        if not versions:
            return None
        
        # Filter versions that have the requested metric
        valid_versions = []
        for version in versions:
            if 'cv_metrics' in version and metric in version['cv_metrics']:
                valid_versions.append(version)
            elif 'test_metrics' in version and metric in version['test_metrics']:
                valid_versions.append(version)
        
        if not valid_versions:
            logger.warning("No versions found with metric '%s'", metric)
            return None
        
        # Find best version (assuming lower is better for MSE/MAE)
        best_version = min(valid_versions, key=lambda x: (
            x.get('cv_metrics', {}).get(metric) or 
            x.get('test_metrics', {}).get(metric) or 
            float('inf')
        ))
        
        return best_version['version_name']