"""
Coral Recovery Prediction - Model Implementations

This module provides machine learning model implementations for coral recovery prediction,
including XGBoost and Neural Network models with a common interface.
"""

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import joblib
import os

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# XGBoost imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost models will not work.")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural Network models will not work.")

from sklearn.preprocessing import StandardScaler


class BaseCoralModel(ABC):
    """
    Abstract base class for coral recovery prediction models.
    Defines the common interface that all models must implement.
    """
    
    def __init__(self, config: Dict, model_name: str):
        """
        Initialize the base model.
        
        Args:
            config: Configuration dictionary
            model_name: Name of the model
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.feature_importance_ = {}
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        logger.info(f"{self.model_name} model initialized")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        seeds = self.config.get('random_seeds', {})
        np.random.seed(seeds.get('numpy_seed', 42))
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(seeds.get('tensorflow_seed', 42))
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseCoralModel':
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted probabilities
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        logger.info(f"Evaluating {self.model_name} model performance")
        
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            y_pred_proba = self.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Add AUC-ROC if probabilities are available
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                if y_pred_proba.ndim == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"{self.model_name} evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.model_name} model: {e}")
            raise
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Dictionary of cross-validation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        logger.info(f"Performing cross-validation for {self.model_name}")
        
        try:
            cv_config = self.config['analysis']
            cv_folds = cv_config.get('cv_folds', 5)
            cv_random_state = cv_config.get('cv_random_state', 42)
            
            # For Neural Networks, implement custom cross-validation
            if self.model_name == "Neural Network":
                return self._custom_cross_validate(X, y, cv_folds, cv_random_state)
            
            # Create cross-validation strategy for sklearn-compatible models
            cv = StratifiedKFold(
                n_splits=cv_folds, 
                shuffle=cv_config.get('cv_shuffle', True),
                random_state=cv_random_state
            )
            
            # Perform cross-validation for different metrics
            cv_scores = {}
            
            # Accuracy
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
            cv_scores['cv_accuracy_mean'] = scores.mean()
            cv_scores['cv_accuracy_std'] = scores.std()
            
            # F1 Score
            scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
            cv_scores['cv_f1_mean'] = scores.mean()
            cv_scores['cv_f1_std'] = scores.std()
            
            # ROC AUC (if binary classification)
            if len(np.unique(y)) == 2:
                scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
                cv_scores['cv_roc_auc_mean'] = scores.mean()
                cv_scores['cv_roc_auc_std'] = scores.std()
            
            logger.info(f"Cross-validation complete. Mean accuracy: {cv_scores['cv_accuracy_mean']:.4f} ± {cv_scores['cv_accuracy_std']:.4f}")
            return cv_scores
            
        except Exception as e:
            logger.error(f"Error in cross-validation for {self.model_name}: {e}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model using joblib for sklearn-compatible models
            if hasattr(self.model, 'save_model') and self.model_name == 'XGBoost':
                # XGBoost native save
                self.model.save_model(filepath + '.json')
            elif hasattr(self.model, 'save') and self.model_name == 'Neural Network':
                # Keras model save
                self.model.save(filepath + '.h5')
            else:
                # Generic joblib save
                joblib.dump(self.model, filepath + '.pkl')
            
            logger.info(f"{self.model_name} model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving {self.model_name} model: {e}")
            raise


class CoralXGBoostModel(BaseCoralModel):
    """
    XGBoost implementation for coral recovery prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize XGBoost model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
        
        super().__init__(config, "XGBoost")
        
        # Get XGBoost-specific configuration
        self.xgb_config = config['models']['xgboost']
        
        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=self.xgb_config.get('n_estimators', 100),
            max_depth=self.xgb_config.get('max_depth', 6),
            learning_rate=self.xgb_config.get('learning_rate', 0.1),
            subsample=self.xgb_config.get('subsample', 0.8),
            colsample_bytree=self.xgb_config.get('colsample_bytree', 0.8),
            random_state=self.xgb_config.get('random_state', 42),
            objective=self.xgb_config.get('objective', 'binary:logistic'),
            eval_metric=self.xgb_config.get('eval_metric', 'logloss'),
            enable_categorical=True,  # Enable categorical feature support
            verbosity=0  # Reduce output verbosity
        )
        
        logger.info("XGBoost model configured successfully")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'CoralXGBoostModel':
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Self for method chaining
        """
        logger.info("Training XGBoost model...")
        
        try:
            # Store feature names
            self.feature_names = X_train.columns.tolist()
            
            # Handle any remaining NaN values
            X_train_clean = X_train.copy()
            numeric_cols = X_train_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = X_train_clean.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric columns with median
            if len(numeric_cols) > 0:
                X_train_clean[numeric_cols] = X_train_clean[numeric_cols].fillna(X_train_clean[numeric_cols].median())
            
            # Fill categorical columns with mode or 'Unknown'
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = X_train_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    X_train_clean[col] = X_train_clean[col].fillna(fill_val)
            
            # Train the model
            self.model.fit(X_train_clean, y_train)
            self.is_fitted = True
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            logger.info("XGBoost model training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Handle any remaining NaN values
            X_test_clean = X_test.copy()
            numeric_cols = X_test_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = X_test_clean.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric columns with median
            if len(numeric_cols) > 0:
                X_test_clean[numeric_cols] = X_test_clean[numeric_cols].fillna(X_test_clean[numeric_cols].median())
            
            # Fill categorical columns with mode or 'Unknown'
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = X_test_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    X_test_clean[col] = X_test_clean[col].fillna(fill_val)
            
            # Make predictions
            predictions = self.model.predict(X_test_clean)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with XGBoost model: {e}")
            raise
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the trained XGBoost model.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Handle any remaining NaN values
            X_test_clean = X_test.copy()
            numeric_cols = X_test_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = X_test_clean.select_dtypes(exclude=[np.number]).columns
            
            # Fill numeric columns with median
            if len(numeric_cols) > 0:
                X_test_clean[numeric_cols] = X_test_clean[numeric_cols].fillna(X_test_clean[numeric_cols].median())
            
            # Fill categorical columns with mode or 'Unknown'
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = X_test_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                    X_test_clean[col] = X_test_clean[col].fillna(fill_val)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(X_test_clean)
            return probabilities
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with XGBoost model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained XGBoost model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.feature_importance_
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance scores."""
        try:
            # Get feature importance from XGBoost
            importance_scores = self.model.feature_importances_
            
            # Create feature importance dictionary
            self.feature_importance_ = dict(zip(self.feature_names, importance_scores))
            
            logger.info("Feature importance calculated for XGBoost model")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            raise


class CoralNeuralNetwork(BaseCoralModel):
    """
    Neural Network implementation for coral recovery prediction using TensorFlow/Keras.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Neural Network model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Please install it with: pip install tensorflow")
        
        super().__init__(config, "Neural Network")
        
        # Get Neural Network-specific configuration
        self.nn_config = config['models']['neural_network']
        
        # Initialize scaler for neural network
        self.scaler = StandardScaler()
        self.input_dim = None
        
        logger.info("Neural Network model initialized")
    
    def _build_model(self, input_dim: int):
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        try:
            # Get architecture parameters
            hidden_layers = self.nn_config.get('hidden_layers', [128, 64, 32])
            dropout_rate = self.nn_config.get('dropout_rate', 0.3)
            learning_rate = self.nn_config.get('learning_rate', 0.001)
            
            # Build model
            model = keras.Sequential()
            
            # Input layer
            model.add(layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
            model.add(layers.Dropout(dropout_rate))
            
            # Hidden layers
            for units in hidden_layers[1:]:
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.Dropout(dropout_rate))
            
            # Output layer (binary classification)
            model.add(layers.Dense(1, activation='sigmoid'))
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            logger.info(f"Neural network architecture built: {len(hidden_layers)} hidden layers")
            return model
            
        except Exception as e:
            logger.error(f"Error building neural network model: {e}")
            raise
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'CoralNeuralNetwork':
        """
        Train the Neural Network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Self for method chaining
        """
        logger.info("Training Neural Network model...")
        
        try:
            # Store feature names
            self.feature_names = X_train.columns.tolist()
            
            # Convert all data to numeric (Neural Networks need numeric input only)
            X_train_numeric = X_train.copy()
            
            # Handle categorical columns
            for col in X_train_numeric.columns:
                if X_train_numeric[col].dtype == 'category' or X_train_numeric[col].dtype == 'object':
                    # Convert categorical to numeric codes
                    X_train_numeric[col] = pd.Categorical(X_train_numeric[col]).codes
            
            # Handle any remaining NaN values in numeric data
            numeric_cols = X_train_numeric.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X_train_numeric[col].isnull().any():
                    median_val = X_train_numeric[col].median()
                    if pd.isna(median_val):  # If all values are NaN
                        median_val = 0
                    X_train_numeric[col] = X_train_numeric[col].fillna(median_val)
            
            # Ensure all data is numeric
            X_train_clean = X_train_numeric.astype(float)
            
            # Scale features for neural network
            X_train_scaled = self.scaler.fit_transform(X_train_clean)
            
            # Build model
            self.input_dim = X_train_scaled.shape[1]
            self.model = self._build_model(self.input_dim)
            
            # Set up callbacks
            callbacks = []
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.nn_config.get('early_stopping_patience', 10),
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stopping)
            
            # Learning rate reduction
            lr_reduction = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0
            )
            callbacks.append(lr_reduction)
            
            # Train the model
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=self.nn_config.get('epochs', 100),
                batch_size=self.nn_config.get('batch_size', 32),
                validation_split=self.nn_config.get('validation_split', 0.2),
                callbacks=callbacks,
                verbose=0  # Reduce output verbosity
            )
            
            self.is_fitted = True
            self.training_history = history
            
            # Calculate feature importance using permutation importance
            self._calculate_feature_importance(X_train_clean, y_train)
            
            logger.info("Neural Network model training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error training Neural Network model: {e}")
            raise
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained Neural Network model.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Convert all data to numeric (same as training)
            X_test_numeric = X_test.copy()
            
            # Handle categorical columns
            for col in X_test_numeric.columns:
                if X_test_numeric[col].dtype == 'category' or X_test_numeric[col].dtype == 'object':
                    # Convert categorical to numeric codes
                    X_test_numeric[col] = pd.Categorical(X_test_numeric[col]).codes
            
            # Handle any remaining NaN values in numeric data
            numeric_cols = X_test_numeric.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X_test_numeric[col].isnull().any():
                    median_val = X_test_numeric[col].median()
                    if pd.isna(median_val):  # If all values are NaN
                        median_val = 0
                    X_test_numeric[col] = X_test_numeric[col].fillna(median_val)
            
            # Ensure all data is numeric
            X_test_clean = X_test_numeric.astype(float)
            
            # Scale features
            X_test_scaled = self.scaler.transform(X_test_clean)
            
            # Make predictions
            predictions_proba = self.model.predict(X_test_scaled, verbose=0)
            predictions = (predictions_proba > 0.5).astype(int).flatten()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with Neural Network model: {e}")
            raise
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the trained Neural Network model.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Convert all data to numeric (same as training)
            X_test_numeric = X_test.copy()
            
            # Handle categorical columns
            for col in X_test_numeric.columns:
                if X_test_numeric[col].dtype == 'category' or X_test_numeric[col].dtype == 'object':
                    # Convert categorical to numeric codes
                    X_test_numeric[col] = pd.Categorical(X_test_numeric[col]).codes
            
            # Handle any remaining NaN values in numeric data
            numeric_cols = X_test_numeric.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X_test_numeric[col].isnull().any():
                    median_val = X_test_numeric[col].median()
                    if pd.isna(median_val):  # If all values are NaN
                        median_val = 0
                    X_test_numeric[col] = X_test_numeric[col].fillna(median_val)
            
            # Ensure all data is numeric
            X_test_clean = X_test_numeric.astype(float)
            
            # Scale features
            X_test_scaled = self.scaler.transform(X_test_clean)
            
            # Predict probabilities
            probabilities = self.model.predict(X_test_scaled, verbose=0)
            
            # Convert to binary classification format (negative class, positive class)
            prob_negative = 1 - probabilities.flatten()
            prob_positive = probabilities.flatten()
            
            return np.column_stack([prob_negative, prob_positive])
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with Neural Network model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained Neural Network model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.feature_importance_
    
    def _calculate_feature_importance(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Calculate feature importance using permutation importance.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        try:
            logger.info("Calculating feature importance for Neural Network using permutation method")
            
            # Scale training data
            X_train_scaled = self.scaler.transform(X_train)
            
            # Get baseline accuracy
            baseline_predictions = self.model.predict(X_train_scaled, verbose=0)
            baseline_accuracy = accuracy_score(y_train, (baseline_predictions > 0.5).astype(int))
            
            # Calculate permutation importance
            importance_scores = []
            
            for i, feature_name in enumerate(self.feature_names):
                # Create permuted version of the data
                X_permuted = X_train_scaled.copy()
                np.random.shuffle(X_permuted[:, i])  # Shuffle the i-th feature
                
                # Get predictions with permuted feature
                permuted_predictions = self.model.predict(X_permuted, verbose=0)
                permuted_accuracy = accuracy_score(y_train, (permuted_predictions > 0.5).astype(int))
                
                # Importance is the decrease in accuracy
                importance = baseline_accuracy - permuted_accuracy
                importance_scores.append(max(0, importance))  # Ensure non-negative
            
            # Normalize importance scores
            total_importance = sum(importance_scores)
            if total_importance > 0:
                importance_scores = [score / total_importance for score in importance_scores]
            
            # Create feature importance dictionary
            self.feature_importance_ = dict(zip(self.feature_names, importance_scores))
            
            logger.info("Feature importance calculated for Neural Network model")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance for Neural Network: {e}")
            # Fallback: use uniform importance
            uniform_importance = 1.0 / len(self.feature_names)
            self.feature_importance_ = {name: uniform_importance for name in self.feature_names}
    
    def _custom_cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int) -> Dict[str, float]:
        """
        Custom cross-validation implementation for Neural Networks.
        
        Args:
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary of cross-validation metrics
        """
        try:
            from sklearn.model_selection import StratifiedKFold
            
            # Create cross-validation strategy
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            # Store original model and scaler
            original_model = self.model
            original_scaler = self.scaler
            
            # Lists to store fold results
            fold_accuracies = []
            fold_f1_scores = []
            fold_roc_aucs = []
            
            # Perform cross-validation
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                logger.debug(f"Processing fold {fold + 1}/{cv_folds}")
                
                # Split data
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create new model and scaler for this fold
                self.scaler = StandardScaler()
                
                # Process training data
                X_fold_train_numeric = self._convert_to_numeric(X_fold_train)
                X_fold_train_scaled = self.scaler.fit_transform(X_fold_train_numeric)
                
                # Process validation data
                X_fold_val_numeric = self._convert_to_numeric(X_fold_val)
                X_fold_val_scaled = self.scaler.transform(X_fold_val_numeric)
                
                # Build and train model for this fold
                self.model = self._build_model(X_fold_train_scaled.shape[1])
                
                # Train with reduced epochs for CV
                self.model.fit(
                    X_fold_train_scaled, y_fold_train,
                    epochs=min(50, self.nn_config.get('epochs', 100)),  # Reduced epochs for CV
                    batch_size=self.nn_config.get('batch_size', 32),
                    validation_split=0.1,  # Small validation split
                    verbose=0
                )
                
                # Make predictions
                y_pred_proba = self.model.predict(X_fold_val_scaled, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                
                # Calculate metrics
                fold_accuracy = accuracy_score(y_fold_val, y_pred)
                fold_f1 = f1_score(y_fold_val, y_pred, average='weighted', zero_division=0)
                fold_roc_auc = roc_auc_score(y_fold_val, y_pred_proba.flatten())
                
                fold_accuracies.append(fold_accuracy)
                fold_f1_scores.append(fold_f1)
                fold_roc_aucs.append(fold_roc_auc)
            
            # Restore original model and scaler
            self.model = original_model
            self.scaler = original_scaler
            
            # Calculate mean and std
            cv_scores = {
                'cv_accuracy_mean': np.mean(fold_accuracies),
                'cv_accuracy_std': np.std(fold_accuracies),
                'cv_f1_mean': np.mean(fold_f1_scores),
                'cv_f1_std': np.std(fold_f1_scores),
                'cv_roc_auc_mean': np.mean(fold_roc_aucs),
                'cv_roc_auc_std': np.std(fold_roc_aucs)
            }
            
            logger.info(f"Neural Network cross-validation complete. Mean accuracy: {cv_scores['cv_accuracy_mean']:.4f} ± {cv_scores['cv_accuracy_std']:.4f}")
            return cv_scores
            
        except Exception as e:
            logger.error(f"Error in custom cross-validation for Neural Network: {e}")
            # Return default values if CV fails
            return {
                'cv_accuracy_mean': 0.0,
                'cv_accuracy_std': 0.0,
                'cv_f1_mean': 0.0,
                'cv_f1_std': 0.0,
                'cv_roc_auc_mean': 0.0,
                'cv_roc_auc_std': 0.0
            }
    
    def _convert_to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame to numeric format for Neural Network processing.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Numeric DataFrame
        """
        X_numeric = X.copy()
        
        # Handle categorical columns
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'category' or X_numeric[col].dtype == 'object':
                # Convert categorical to numeric codes
                X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        
        # Handle any remaining NaN values in numeric data
        numeric_cols = X_numeric.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_numeric[col].isnull().any():
                median_val = X_numeric[col].median()
                if pd.isna(median_val):  # If all values are NaN
                    median_val = 0
                X_numeric[col] = X_numeric[col].fillna(median_val)
        
        # Ensure all data is numeric
        return X_numeric.astype(float)


def create_model(model_type: str, config: Dict) -> BaseCoralModel:
    """
    Factory function to create model instances.
    
    Args:
        model_type: Type of model ('xgboost' or 'neural_network')
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    if model_type.lower() == 'xgboost':
        return CoralXGBoostModel(config)
    elif model_type.lower() == 'neural_network':
        return CoralNeuralNetwork(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'xgboost', 'neural_network'")


def compare_models(models: Dict[str, BaseCoralModel], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        models: Dictionary of model name to model instance
        X_test: Test features
        y_test: Test targets
        
    Returns:
        DataFrame with comparison results
    """
    logger.info("Comparing model performances")
    
    try:
        comparison_results = []
        
        for model_name, model in models.items():
            if not model.is_fitted:
                logger.warning(f"Model {model_name} is not fitted, skipping comparison")
                continue
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            metrics['model'] = model_name
            comparison_results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        if not comparison_df.empty:
            # Sort by accuracy (or another metric)
            comparison_df = comparison_df.sort_values('accuracy', ascending=False)
            logger.info("Model comparison completed")
        
        return comparison_df
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise