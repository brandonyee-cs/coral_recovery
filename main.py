#!/usr/bin/env python3
"""
Coral Recovery Prediction - Main Pipeline

This script orchestrates the complete machine learning pipeline for coral recovery prediction,
including data loading, preprocessing, model training, evaluation, and results generation.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from src.analysis import CoralDataProcessor, FeatureAnalyzer
from src.models import create_model, compare_models, BaseCoralModel, TENSORFLOW_AVAILABLE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')


class CoralPredictionPipeline:
    """
    Main pipeline class that orchestrates the complete coral recovery prediction workflow.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        self.setup_output_directories()
        
        # Initialize components
        self.data_processor = CoralDataProcessor(config_path)
        self.feature_analyzer = FeatureAnalyzer(self.config)
        
        # Pipeline state
        self.datasets = {}
        self.merged_data = None
        self.processed_data = None
        self.models = {}
        self.results = {}
        
        self.logger.info("Coral Prediction Pipeline initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def setup_logging(self) -> None:
        """Set up comprehensive logging."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up root logger
        self.logger = logging.getLogger('CoralPipeline')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        log_file = log_config.get('log_file', 'results/coral_prediction.log')
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info("Logging configured successfully")
    
    def setup_output_directories(self) -> None:
        """Create necessary output directories."""
        output_dir = self.config['analysis']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['models', 'plots', 'results', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        self.logger.info(f"Output directories created in {output_dir}")
    
    def load_and_process_data(self) -> None:
        """Load and process all coral datasets."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: DATA LOADING AND PROCESSING")
        self.logger.info("=" * 60)
        
        try:
            # Load primary dataset
            self.logger.info("Loading coral bleaching dataset...")
            raw_data = self.data_processor.load_coral_data()
            self.logger.info(f"Raw dataset shape: {raw_data.shape}")
            
            # Clean and prepare data
            self.logger.info("Cleaning and preparing data...")
            self.merged_data = self.data_processor.clean_and_prepare_data(raw_data)
            self.logger.info(f"Cleaned dataset shape: {self.merged_data.shape}")
            
            # Validate target variable
            self.logger.info("Validating target variable...")
            self.merged_data = self.data_processor.validate_target_variable(self.merged_data)
            
            # Feature engineering
            self.logger.info("Performing feature engineering...")
            self.merged_data = self.data_processor.perform_feature_engineering(self.merged_data)
            
            # Preprocess features
            self.logger.info("Preprocessing features...")
            self.processed_data = self.data_processor.preprocess_features(self.merged_data)
            
            self.logger.info(f"Final processed dataset shape: {self.processed_data.shape}")
            
            # Generate data distribution plots
            self._generate_data_visualizations()
            
        except Exception as e:
            self.logger.error(f"Error in data loading and processing: {e}")
            raise
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prepare data for model training with validation split."""
        self.logger.info("Preparing training data...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = self.data_processor.split_data(self.processed_data)
            
            # Further split training data for validation (for threshold optimization)
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, 
                stratify=y_train if len(y_train.unique()) > 1 else None
            )
            
            # Feature selection
            X_train_selected, selected_features = self.data_processor.perform_feature_selection(X_train_split, y_train_split)
            X_val_selected = X_val[selected_features]
            X_test_selected = X_test[selected_features]
            
            self.logger.info(f"Selected {len(selected_features)} features for training")
            self.logger.info(f"Training set: {X_train_selected.shape}, Validation set: {X_val_selected.shape}, Test set: {X_test_selected.shape}")
            
            # Log class distribution
            train_distribution = y_train_split.value_counts(normalize=True)
            val_distribution = y_val.value_counts(normalize=True)
            test_distribution = y_test.value_counts(normalize=True)
            
            self.logger.info(f"Training set class distribution: {dict(train_distribution)}")
            self.logger.info(f"Validation set class distribution: {dict(val_distribution)}")
            self.logger.info(f"Test set class distribution: {dict(test_distribution)}")
            
            return X_train_selected, X_val_selected, X_test_selected, y_train_split, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
    
    def train_models(self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series) -> None:
        """Train both XGBoost and Neural Network models with threshold optimization."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: MODEL TRAINING")
        self.logger.info("=" * 60)
        
        try:
            # Train XGBoost model
            self.logger.info("Training XGBoost model...")
            xgb_model = create_model('xgboost', self.config)
            xgb_model.fit(X_train, y_train)
            
            # Optimize threshold if configured
            if self.config['analysis']['preprocessing'].get('threshold_optimization', False):
                xgb_model.optimize_threshold(X_val, y_val)
            
            self.models['XGBoost'] = xgb_model
            self.logger.info("XGBoost model training completed")
            
            # Train Neural Network model (if TensorFlow is available)
            if TENSORFLOW_AVAILABLE:
                self.logger.info("Training Neural Network model...")
                nn_model = create_model('neural_network', self.config)
                nn_model.fit(X_train, y_train)
                
                # Optimize threshold if configured
                if self.config['analysis']['preprocessing'].get('threshold_optimization', False):
                    nn_model.optimize_threshold(X_val, y_val)
                
                self.models['Neural Network'] = nn_model
                self.logger.info("Neural Network model training completed")
                
                # Train Ensemble model
                self.logger.info("Training Ensemble model...")
                ensemble_model = create_model('ensemble', self.config)
                ensemble_model.fit(X_train, y_train)
                
                # Optimize threshold if configured
                if self.config['analysis']['preprocessing'].get('threshold_optimization', False):
                    ensemble_model.optimize_threshold(X_val, y_val)
                
                self.models['Ensemble'] = ensemble_model
                self.logger.info("Ensemble model training completed")
            else:
                self.logger.warning("Skipping Neural Network and Ensemble model training - TensorFlow not available")
            
            self.logger.info(f"Successfully trained {len(self.models)} models")
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluate all trained models."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: MODEL EVALUATION")
        self.logger.info("=" * 60)
        
        try:
            # Evaluate each model
            for model_name, model in self.models.items():
                self.logger.info(f"Evaluating {model_name} model...")
                
                # Basic evaluation
                metrics = model.evaluate(X_test, y_test)
                self.results[model_name] = metrics
                
                # Cross-validation
                cv_metrics = model.cross_validate(X_test, y_test)
                self.results[model_name].update(cv_metrics)
                
                # Log results
                self.logger.info(f"{model_name} Results:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            # Compare models
            self.logger.info("Comparing model performances...")
            comparison_df = compare_models(self.models, X_test, y_test)
            
            if not comparison_df.empty:
                self.logger.info("Model Comparison Results:")
                self.logger.info(f"\n{comparison_df.to_string(index=False)}")
                
                # Save comparison results
                comparison_path = os.path.join(self.config['analysis']['output_dir'], 'results', 'model_comparison.csv')
                comparison_df.to_csv(comparison_path, index=False)
                self.logger.info(f"Model comparison saved to {comparison_path}")
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {e}")
            raise
    
    def analyze_feature_importance(self) -> None:
        """Analyze and visualize feature importance for all models."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: FEATURE IMPORTANCE ANALYSIS")
        self.logger.info("=" * 60)
        
        try:
            feature_importance_results = {}
            
            for model_name, model in self.models.items():
                self.logger.info(f"Analyzing feature importance for {model_name}...")
                
                # Get feature importance
                feature_importance = model.get_feature_importance()
                
                # Analyze importance
                importance_df = self.feature_analyzer.analyze_feature_importance(
                    feature_importance, model_name
                )
                
                # Store results
                feature_importance_results[model_name] = importance_df
                
                # Generate visualization
                self.feature_analyzer.plot_feature_importance(importance_df, model_name)
                
                # Log top features
                top_n = self.config['analysis']['feature_importance_top_n']
                top_features = importance_df.head(top_n)
                
                self.logger.info(f"Top {top_n} features for {model_name}:")
                for _, row in top_features.iterrows():
                    self.logger.info(f"  {row['rank']:2d}. {row['feature']}: {row['importance']:.4f}")
            
            # Save feature importance results
            self._save_feature_importance_results(feature_importance_results)
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
            raise
    
    def generate_predictions_and_visualizations(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Generate predictions and create comprehensive visualizations."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 5: PREDICTIONS AND VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        try:
            predictions_results = {}
            
            for model_name, model in self.models.items():
                self.logger.info(f"Generating predictions for {model_name}...")
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Store predictions
                predictions_results[model_name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Generate visualizations
                self._create_model_visualizations(model_name, y_test, y_pred, y_pred_proba)
            
            # Save predictions
            self._save_predictions(predictions_results, X_test, y_test)
            
            # Create comparison visualizations
            self._create_comparison_visualizations(predictions_results, y_test)
            
        except Exception as e:
            self.logger.error(f"Error generating predictions and visualizations: {e}")
            raise
    
    def save_models_and_results(self) -> None:
        """Save trained models and comprehensive results."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 6: SAVING MODELS AND RESULTS")
        self.logger.info("=" * 60)
        
        try:
            output_config = self.config.get('output', {})
            output_dir = self.config['analysis']['output_dir']
            
            # Save models
            if output_config.get('save_models', True):
                models_dir = os.path.join(output_dir, 'models')
                
                for model_name, model in self.models.items():
                    model_path = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}_model")
                    model.save_model(model_path)
                    self.logger.info(f"Saved {model_name} model to {model_path}")
            
            # Save evaluation metrics
            if output_config.get('save_evaluation_metrics', True):
                metrics_path = os.path.join(output_dir, 'results', 'evaluation_metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(self.results, f, indent=2, default=str)
                self.logger.info(f"Evaluation metrics saved to {metrics_path}")
            
            # Save configuration used
            config_path = os.path.join(output_dir, 'results', 'pipeline_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Pipeline configuration saved to {config_path}")
            
            # Create summary report
            self._create_summary_report()
            
        except Exception as e:
            self.logger.error(f"Error saving models and results: {e}")
            raise
    
    def run_complete_pipeline(self) -> None:
        """Execute the complete coral recovery prediction pipeline."""
        start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("CORAL RECOVERY PREDICTION PIPELINE - STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {start_time}")
        self.logger.info(f"Configuration: {self.config_path}")
        
        try:
            # Phase 1: Data Loading and Processing
            self.load_and_process_data()
            
            # Phase 2: Prepare Training Data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data()
            
            # Phase 3: Model Training
            self.train_models(X_train, X_val, y_train, y_val)
            
            # Phase 4: Model Evaluation
            self.evaluate_models(X_test, y_test)
            
            # Phase 5: Feature Importance Analysis
            self.analyze_feature_importance()
            
            # Phase 6: Predictions and Visualizations
            self.generate_predictions_and_visualizations(X_test, y_test)
            
            # Phase 7: Save Models and Results
            self.save_models_and_results()
            
            # Pipeline completion
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 80)
            self.logger.info("CORAL RECOVERY PREDICTION PIPELINE - COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"End time: {end_time}")
            self.logger.info(f"Total duration: {duration}")
            self.logger.info(f"Results saved to: {self.config['analysis']['output_dir']}")
            
            # Print final summary
            self._print_final_summary()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error("Pipeline execution terminated due to error")
            raise
    
    def _generate_data_visualizations(self) -> None:
        """Generate data distribution and correlation visualizations."""
        try:
            self.logger.info("Generating data visualizations...")
            
            # Create data distribution plots
            self.feature_analyzer.plot_data_distribution(
                self.processed_data, 
                self.data_processor.target_column
            )
            
            self.logger.info("Data visualizations generated successfully")
            
        except Exception as e:
            self.logger.warning(f"Error generating data visualizations: {e}")
    
    def _create_model_visualizations(self, model_name: str, y_true: pd.Series, 
                                   y_pred: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Create visualizations for individual model performance."""
        try:
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            
            output_dir = self.config['analysis']['output_dir']
            plots_dir = os.path.join(output_dir, 'plots')
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            for fmt in self.config['output']['plot_formats']:
                plt.savefig(f"{plots_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.{fmt}", 
                           dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC Curve (for binary classification)
            if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
                plt.figure(figsize=(8, 6))
                
                if y_pred_proba.ndim == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                else:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                
                for fmt in self.config['output']['plot_formats']:
                    plt.savefig(f"{plots_dir}/roc_curve_{model_name.lower().replace(' ', '_')}.{fmt}", 
                               dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating visualizations for {model_name}: {e}")
    
    def _create_comparison_visualizations(self, predictions_results: Dict, y_true: pd.Series) -> None:
        """Create comparison visualizations between models."""
        try:
            from sklearn.metrics import accuracy_score, f1_score
            
            output_dir = self.config['analysis']['output_dir']
            plots_dir = os.path.join(output_dir, 'plots')
            
            # Model performance comparison
            model_names = list(predictions_results.keys())
            accuracies = []
            f1_scores = []
            
            for model_name in model_names:
                y_pred = predictions_results[model_name]['predictions']
                accuracies.append(accuracy_score(y_true, y_pred))
                f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
            
            # Create comparison bar plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Accuracy comparison
            ax1.bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # F1 Score comparison
            ax2.bar(model_names, f1_scores, color=['lightgreen', 'gold'])
            ax2.set_title('Model F1 Score Comparison')
            ax2.set_ylabel('F1 Score')
            ax2.set_ylim(0, 1)
            for i, v in enumerate(f1_scores):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            plt.tight_layout()
            
            for fmt in self.config['output']['plot_formats']:
                plt.savefig(f"{plots_dir}/model_comparison.{fmt}", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating comparison visualizations: {e}")
    
    def _save_feature_importance_results(self, feature_importance_results: Dict) -> None:
        """Save feature importance results to files."""
        try:
            output_dir = self.config['analysis']['output_dir']
            results_dir = os.path.join(output_dir, 'results')
            
            for model_name, importance_df in feature_importance_results.items():
                filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
                filepath = os.path.join(results_dir, filename)
                importance_df.to_csv(filepath, index=False)
                self.logger.info(f"Feature importance for {model_name} saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Error saving feature importance results: {e}")
    
    def _save_predictions(self, predictions_results: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Save model predictions to files."""
        try:
            if not self.config['output'].get('save_predictions', True):
                return
            
            output_dir = self.config['analysis']['output_dir']
            results_dir = os.path.join(output_dir, 'results')
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame()
            predictions_df['true_label'] = y_test.values
            
            for model_name, results in predictions_results.items():
                predictions_df[f'{model_name.lower().replace(" ", "_")}_prediction'] = results['predictions']
                
                if results['probabilities'] is not None:
                    if results['probabilities'].ndim == 2:
                        predictions_df[f'{model_name.lower().replace(" ", "_")}_probability'] = results['probabilities'][:, 1]
                    else:
                        predictions_df[f'{model_name.lower().replace(" ", "_")}_probability'] = results['probabilities']
            
            # Save predictions
            predictions_path = os.path.join(results_dir, 'model_predictions.csv')
            predictions_df.to_csv(predictions_path, index=False)
            self.logger.info(f"Model predictions saved to {predictions_path}")
            
        except Exception as e:
            self.logger.warning(f"Error saving predictions: {e}")
    
    def _create_summary_report(self) -> None:
        """Create a comprehensive summary report."""
        try:
            output_dir = self.config['analysis']['output_dir']
            report_path = os.path.join(output_dir, 'results', 'pipeline_summary_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("CORAL RECOVERY PREDICTION PIPELINE - SUMMARY REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Configuration: {self.config_path}\n\n")
                
                # Data summary
                f.write("DATA SUMMARY:\n")
                f.write("-" * 20 + "\n")
                if self.processed_data is not None:
                    f.write(f"Final dataset shape: {self.processed_data.shape}\n")
                    f.write(f"Number of features: {self.processed_data.shape[1] - 1}\n")
                    target_dist = self.processed_data[self.data_processor.target_column].value_counts(normalize=True)
                    f.write(f"Target distribution: {dict(target_dist)}\n\n")
                
                # Model results
                f.write("MODEL PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                for model_name, metrics in self.results.items():
                    f.write(f"\n{model_name}:\n")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")
                
                f.write(f"\nDetailed results and visualizations saved to: {output_dir}\n")
            
            self.logger.info(f"Summary report created: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Error creating summary report: {e}")
    
    def _print_final_summary(self) -> None:
        """Print final summary to console."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        if self.results:
            print("\nMODEL PERFORMANCE SUMMARY:")
            print("-" * 40)
            
            for model_name, metrics in self.results.items():
                print(f"\n{model_name}:")
                key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                for metric in key_metrics:
                    if metric in metrics:
                        print(f"  {metric.capitalize()}: {metrics[metric]:.4f}")
        
        print(f"\nResults saved to: {self.config['analysis']['output_dir']}")
        print("Pipeline completed successfully!")
        print("=" * 80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Coral Recovery Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with default config.yaml
  python main.py --config custom.yaml    # Run with custom configuration
  python main.py --verbose               # Run with verbose logging
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Override output directory from config'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the coral recovery prediction pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Check if config file exists
        if not os.path.exists(args.config):
            print(f"Error: Configuration file '{args.config}' not found.")
            sys.exit(1)
        
        # Override logging level if verbose
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Initialize and run pipeline
        pipeline = CoralPredictionPipeline(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            pipeline.config['analysis']['output_dir'] = args.output_dir
            pipeline.setup_output_directories()
        
        # Run the complete pipeline
        pipeline.run_complete_pipeline()
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()