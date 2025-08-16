# Design Document

## Overview

The coral recovery prediction system is designed as a modular machine learning pipeline that processes multiple coral-related datasets to predict recovery outcomes. The system implements two complementary modeling approaches: XGBoost (gradient boosting) for interpretable tree-based predictions and a Neural Network for capturing complex non-linear relationships. The architecture emphasizes reproducibility, configurability, and comprehensive analysis of feature importance.

## Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────┐
│    main.py      │  ← Entry point and orchestration
├─────────────────┤
│  src/analysis.py│  ← Data processing and analysis
├─────────────────┤
│  src/models.py  │  ← Model implementations
├─────────────────┤
│   config.yaml   │  ← Configuration management
└─────────────────┘
```

### Data Flow

1. **Configuration Loading**: System reads parameters from config.yaml
2. **Data Ingestion**: Multiple CSV files are loaded and validated
3. **Data Integration**: Datasets are merged on common keys (site, date, etc.)
4. **Feature Engineering**: Environmental and biological features are processed
5. **Model Training**: Both XGBoost and Neural Network models are trained
6. **Evaluation**: Models are evaluated using multiple metrics
7. **Analysis**: Feature importance and comparative analysis are performed
8. **Output Generation**: Results, visualizations, and model artifacts are saved

## Components and Interfaces

### Configuration Management (config.yaml)

```yaml
data:
  data_path: "data/"
  target_column: "recovery_status"
  test_size: 0.2
  random_state: 42

models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    random_state: 42
  
  neural_network:
    hidden_layers: [128, 64, 32]
    dropout_rate: 0.3
    learning_rate: 0.001
    epochs: 100
    batch_size: 32

analysis:
  cv_folds: 5
  feature_importance_top_n: 15
  save_plots: true
  output_dir: "results/"
```

### Data Analysis Module (src/analysis.py)

**Core Functions:**
- `load_coral_data()`: Loads and validates all CSV datasets
- `merge_datasets()`: Integrates datasets on common keys
- `preprocess_features()`: Handles missing values, scaling, encoding
- `create_target_variable()`: Derives recovery status from coral health data
- `perform_feature_selection()`: Identifies relevant features
- `generate_visualizations()`: Creates analysis plots and charts

**Key Classes:**
- `CoralDataProcessor`: Handles all data preprocessing operations
- `FeatureAnalyzer`: Performs feature importance and correlation analysis

### Models Module (src/models.py)

**XGBoost Implementation:**
- `CoralXGBoostModel`: Wrapper class for XGBoost with coral-specific preprocessing
- Handles categorical encoding, feature scaling
- Implements cross-validation and hyperparameter tuning
- Provides built-in feature importance extraction

**Neural Network Implementation:**
- `CoralNeuralNetwork`: TensorFlow/Keras-based neural network
- Multi-layer perceptron architecture with dropout regularization
- Custom loss functions for imbalanced coral recovery data
- Early stopping and learning rate scheduling

**Common Interface:**
```python
class BaseCoralModel:
    def fit(self, X_train, y_train)
    def predict(self, X_test)
    def predict_proba(self, X_test)
    def get_feature_importance(self)
    def evaluate(self, X_test, y_test)
```

### Main Pipeline (main.py)

Orchestrates the complete workflow:
1. Configuration loading and validation
2. Data loading and preprocessing
3. Model training and evaluation
4. Comparative analysis
5. Results generation and saving

## Data Models

### Input Data Structure

**Primary Datasets:**
- `bleaching.csv`: Coral bleaching events with DHW (Degree Heating Weeks) data
- `environmental_summary.csv`: Site-level environmental conditions
- `sedimentation_data.csv`: Sedimentation rates by site and time
- `wave_data.csv`: Wave energy and height measurements
- `metadata_coral_samples.csv`: Individual coral sample information
- `metadata_site_locations.csv`: Geographic and depth information

**Integrated Feature Set:**
```python
features = {
    'environmental': ['dhw', 'mean_dhw', 'max_dhw', 'depth_m'],
    'sedimentation': ['g/day1_mean', 'g/day2_mean', ..., 'g/day7_mean'],
    'wave_action': ['rms_mean', 'rms_std', 'height_mean', 'height_std'],
    'spatial': ['lat', 'lon', 'block', 'region'],
    'temporal': ['date', 'time'],
    'biological': ['growth_morph', 'length_cm', 'width_cm', 'color']
}
```

**Target Variable:**
- `recovery_status`: Binary classification (0=no recovery, 1=recovery)
- Derived from coral health status changes over time
- Considers bleaching severity and subsequent recovery patterns

### Feature Engineering Strategy

1. **Temporal Features**: Extract seasonal patterns, time since bleaching events
2. **Environmental Stress Indices**: Combine DHW, sedimentation, and wave data
3. **Spatial Clustering**: Group sites by geographic proximity and depth
4. **Interaction Features**: Cross-products of key environmental variables
5. **Lag Features**: Previous time period environmental conditions

## Error Handling

### Data Validation
- Missing file detection with clear error messages
- Schema validation for expected columns and data types
- Outlier detection and handling strategies
- Data quality checks (e.g., negative depths, invalid dates)

### Model Training Errors
- Convergence failure handling for neural networks
- Memory management for large datasets
- Cross-validation fold failures
- Hyperparameter validation

### Pipeline Robustness
- Graceful degradation when optional features are missing
- Checkpoint saving for long-running processes
- Comprehensive logging throughout the pipeline
- Configuration validation before execution

## Testing Strategy

### Unit Tests
- Data loading and preprocessing functions
- Feature engineering transformations
- Model training and prediction methods
- Configuration parsing and validation

### Integration Tests
- End-to-end pipeline execution
- Model performance benchmarks
- Cross-validation consistency
- Output file generation and format validation

### Data Quality Tests
- Dataset completeness and consistency checks
- Feature distribution validation
- Target variable balance assessment
- Temporal data continuity verification

### Model Validation
- Performance threshold testing (minimum accuracy requirements)
- Feature importance stability across runs
- Prediction consistency with different random seeds
- Overfitting detection through learning curves

## Performance Considerations

### Computational Efficiency
- Efficient data loading with pandas optimizations
- Memory-conscious processing for large datasets
- Parallel processing for cross-validation
- GPU utilization for neural network training (if available)

### Scalability
- Modular design allows for easy dataset expansion
- Configuration-driven approach supports different coral regions
- Model serialization for deployment and reuse
- Batch processing capabilities for large-scale predictions

### Monitoring and Logging
- Comprehensive logging of data processing steps
- Model training progress tracking
- Performance metrics logging
- Error tracking and debugging information