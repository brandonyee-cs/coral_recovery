# Coral Recovery Prediction System

A comprehensive machine learning pipeline for predicting coral recovery from bleaching events using multiple ML approaches and modern data science tools.

## 🌊 Overview

This system analyzes coral bleaching recovery patterns using environmental, biological, temporal, and location features to predict recovery outcomes (No Recovery, Partial Recovery, Full Recovery) using:

- **XGBoost**: Gradient boosting for structured data
- **Random Forest**: Ensemble method for robust predictions  
- **Neural Network**: Deep learning with PyTorch for complex patterns

## 🛠️ Tech Stack

- **Data Processing**: Polars (faster than pandas)
- **Visualization**: Plotly (interactive plots)
- **Deep Learning**: PyTorch (instead of TensorFlow)
- **ML Models**: XGBoost, Scikit-learn
- **Configuration**: YAML

## 📁 Project Structure

```
coral_recovery/
├── config.yaml                 # Configuration settings
├── main.py                     # Main pipeline script
├── requirements.txt            # Python dependencies
├── coral_recovery/
│   ├── __init__.py
│   ├── models.py              # ML model implementations
│   └── analysis.py            # Data processing & visualization
├── models/                    # Saved trained models
├── results/                   # Analysis results & reports
└── plots/                     # Generated visualizations
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd coral_recovery

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run with default synthetic data
python main.py

# The pipeline will:
# - Generate synthetic coral data (or load your CSV)
# - Preprocess and engineer features
# - Train 3 different ML models
# - Perform cross-validation
# - Generate visualizations and reports
```

### 3. Using Your Own Data

Replace the synthetic data generation in `main.py`:

```python
# Replace this line:
df = data_loader.generate_synthetic_data(n_samples=2000)

# With your data:
df = data_loader.load_data("path/to/your/coral_data.csv")
```

## 📊 Expected Data Format

Your CSV should contain these columns:

### Environmental Features
- `water_temperature` - Water temperature (°C)
- `ph_level` - pH level
- `nutrient_concentration` - Nutrient concentration
- `light_availability` - Light availability (0-1)
- `depth` - Depth (meters)
- `salinity` - Salinity (ppt)
- `current_strength` - Current strength (m/s)

### Biological Features
- `coral_species` - Coral species name
- `initial_bleaching_severity` - Initial bleaching severity (0-1)
- `colony_size` - Colony size (cm²)
- `zooxanthellae_density` - Zooxanthellae density (cells/cm²)
- `tissue_thickness` - Tissue thickness (mm)

### Temporal Features
- `days_since_bleaching` - Days since bleaching event
- `season` - Season (Spring/Summer/Autumn/Winter)
- `recovery_monitoring_period` - Monitoring period (days)

### Location Features
- `latitude` - Latitude
- `longitude` - Longitude
- `reef_zone` - Reef zone type

### Target Variable
- `recovery_success` - Recovery outcome (0: No Recovery, 1: Partial, 2: Full)

## ⚙️ Configuration

Customize the pipeline by editing `config.yaml`:

```yaml
# Model hyperparameters
models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  
  neural_network:
    hidden_layers: [128, 64, 32]
    learning_rate: 0.001
    epochs: 100

# Data processing
data:
  preprocessing:
    test_size: 0.2
    scale_features: true
```

## 📈 Outputs

The pipeline generates:

### 1. Trained Models
- `models/xgboost_model.json`
- `models/random_forest_model.pkl` 
- `models/neural_network_model.pth`

### 2. Analysis Results
- `results/feature_importance.csv` - Feature importance rankings
- `results/model_performance.csv` - Model comparison metrics
- `results/analysis_report.md` - Detailed markdown report
- `results/coral_recovery_analysis_report.html` - Interactive HTML report

### 3. Visualizations
- `plots/data_distribution.html` - Data distribution plots
- `plots/feature_importance.html` - Feature importance comparison
- `plots/model_performance.html` - Model performance comparison
- `plots/confusion_matrices.html` - Confusion matrices

## 🔬 Key Features

### Advanced Feature Engineering
- Temperature and pH stress indicators
- Light-depth interaction ratios
- Recovery potential scores
- Seasonal factors
- Colony health indices

### Model Comparison
- Cross-validation for robust evaluation
- Feature importance analysis across models
- Performance visualization
- Confusion matrix analysis

### Interactive Visualizations
- Plotly-based interactive charts
- Data distribution analysis
- Feature importance rankings
- Model performance comparison

## 🧪 Model Details

### XGBoost
- Gradient boosting optimized for structured data
- Built-in feature importance
- Handles missing values well

### Random Forest  
- Ensemble of decision trees
- Robust to overfitting
- Good baseline performance

### Neural Network (PyTorch)
- Multi-layer perceptron
- Batch normalization and dropout
- Early stopping and learning rate scheduling
- GPU support when available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Coral bleaching research community
- Marine biology datasets
- Open source ML libraries

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed description

---