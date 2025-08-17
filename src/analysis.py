"""
Coral Recovery Prediction - Data Processing and Analysis Module

This module provides comprehensive data processing, feature engineering, and analysis
capabilities for coral recovery prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
# Try to import imbalanced-learn, fall back gracefully if not available
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn not available. Resampling features will be disabled.")
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoralDataProcessor:
    """
    Main class for processing coral recovery data including loading, preprocessing,
    feature engineering, and target variable creation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the CoralDataProcessor with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.data_path = self.config['data']['data_path']
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = self.config['data']['target_column']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['analysis']['output_dir'], exist_ok=True)
        
        logger.info("CoralDataProcessor initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def load_coral_data(self) -> pd.DataFrame:
        """
        Load the primary coral bleaching dataset from CSV file.
        
        Returns:
            DataFrame containing the bleaching data
        """
        logger.info("Loading coral bleaching dataset...")
        
        try:
            # Load the primary bleaching dataset
            primary_file = self.config['data']['primary_file']
            file_path = os.path.join(self.data_path, primary_file)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Primary data file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"Loaded bleaching dataset: {df.shape}")
            
            # Basic data info
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Target column '{self.target_column}' distribution:")
            if self.target_column in df.columns:
                target_dist = df[self.target_column].value_counts()
                logger.info(f"{target_dist}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading coral data: {e}")
            raise
    
    def clean_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the bleaching dataset for analysis.
        
        Args:
            df: Input DataFrame from bleaching.csv
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning and preparing bleaching dataset...")
        
        try:
            df_clean = df.copy()
            
            # Remove rows with missing target values
            if self.target_column in df_clean.columns:
                initial_rows = len(df_clean)
                df_clean = df_clean.dropna(subset=[self.target_column])
                removed_rows = initial_rows - len(df_clean)
                if removed_rows > 0:
                    logger.info(f"Removed {removed_rows} rows with missing target values")
            
            # Clean column names to be more model-friendly
            df_clean.columns = [col.replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_').replace('-', '_') for col in df_clean.columns]
            
            # Update target column name if it was changed
            if 'status' in df.columns and 'status' not in df_clean.columns:
                # Find the new name for status column
                for col in df_clean.columns:
                    if 'status' in col.lower():
                        self.target_column = col
                        break
            
            # Remove completely empty columns
            empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
            if empty_cols:
                df_clean = df_clean.drop(columns=empty_cols)
                logger.info(f"Removed {len(empty_cols)} completely empty columns")
            
            # Remove columns with too many missing values (>80% missing)
            missing_threshold = 0.8
            high_missing_cols = []
            for col in df_clean.columns:
                if col != self.target_column:
                    missing_pct = df_clean[col].isnull().sum() / len(df_clean)
                    if missing_pct > missing_threshold:
                        high_missing_cols.append(col)
            
            if high_missing_cols:
                df_clean = df_clean.drop(columns=high_missing_cols)
                logger.info(f"Removed {len(high_missing_cols)} columns with >80% missing values")
            
            # Remove non-informative columns (single unique value)
            single_value_cols = []
            for col in df_clean.columns:
                if col != self.target_column and df_clean[col].nunique() <= 1:
                    single_value_cols.append(col)
            
            if single_value_cols:
                df_clean = df_clean.drop(columns=single_value_cols)
                logger.info(f"Removed {len(single_value_cols)} columns with single unique value")
            
            logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            raise
    
    def validate_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and ensure the target variable is properly formatted.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with validated target variable
        """
        logger.info("Validating target variable...")
        
        try:
            df = df.copy()
            
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataset")
            
            # Convert target to numeric if needed
            df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
            
            # Remove rows with missing target values
            initial_rows = len(df)
            df = df.dropna(subset=[self.target_column])
            removed_rows = initial_rows - len(df)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with missing target values")
            
            # Ensure target is binary (0 or 1)
            unique_values = df[self.target_column].unique()
            logger.info(f"Target variable unique values: {sorted(unique_values)}")
            
            # Convert to binary if needed
            if len(unique_values) == 2:
                # Map to 0 and 1
                min_val, max_val = min(unique_values), max(unique_values)
                df[self.target_column] = df[self.target_column].map({min_val: 0, max_val: 1})
            elif len(unique_values) > 2:
                # Convert to binary using median threshold
                threshold = df[self.target_column].median()
                df[self.target_column] = (df[self.target_column] > threshold).astype(int)
                logger.info(f"Converted multi-class target to binary using threshold {threshold}")
            
            recovery_rate = df[self.target_column].mean()
            logger.info(f"Target variable validated. Recovery rate: {recovery_rate:.2%}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating target variable: {e}")
            raise
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features including handling missing values, scaling, and encoding.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting feature preprocessing...")
        
        try:
            df_processed = df.copy()
            
            # Handle missing values
            missing_strategy = self.config['analysis']['preprocessing']['handle_missing']
            
            # Separate numeric and categorical columns
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target column from preprocessing
            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)
            if self.target_column in categorical_cols:
                categorical_cols.remove(self.target_column)
            
            # Handle missing values in numeric columns
            if numeric_cols:
                for col in numeric_cols:
                    if df_processed[col].isnull().any():
                        if missing_strategy == 'median':
                            fill_value = df_processed[col].median()
                            # If median is NaN (all values are NaN), use 0
                            if pd.isna(fill_value):
                                fill_value = 0
                        elif missing_strategy == 'mean':
                            fill_value = df_processed[col].mean()
                            # If mean is NaN (all values are NaN), use 0
                            if pd.isna(fill_value):
                                fill_value = 0
                        else:  # drop
                            continue
                        
                        df_processed[col] = df_processed[col].fillna(fill_value)
                        logger.debug(f"Filled missing values in {col} with {fill_value}")
            
            # Final check: ensure no NaN values remain in numeric columns
            remaining_nan_cols = df_processed[numeric_cols].columns[df_processed[numeric_cols].isnull().any()].tolist()
            if remaining_nan_cols:
                logger.warning(f"Still have NaN values in columns: {remaining_nan_cols}")
                for col in remaining_nan_cols:
                    df_processed[col] = df_processed[col].fillna(0)
                    logger.info(f"Force-filled remaining NaN values in {col} with 0")
            
            # Handle missing values in categorical columns
            if categorical_cols:
                for col in categorical_cols:
                    df_processed[col] = df_processed[col].fillna('Unknown')
            
            # Encode categorical variables
            encoding_method = self.config['analysis']['preprocessing']['encode_categorical']
            
            if categorical_cols and encoding_method == 'label':
                for col in categorical_cols:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
            
            elif categorical_cols and encoding_method == 'onehot':
                # For high-cardinality categorical variables, use label encoding instead
                high_cardinality_threshold = 50
                
                for col in categorical_cols:
                    unique_values = df_processed[col].nunique()
                    
                    if unique_values > high_cardinality_threshold:
                        # Use label encoding for high cardinality columns
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        self.label_encoders[col] = le
                        logger.info(f"Used label encoding for high-cardinality column '{col}' ({unique_values} unique values)")
                    else:
                        # Use one-hot encoding for low cardinality columns
                        dummies = pd.get_dummies(df_processed[col], prefix=col)
                        df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
                
                # Clean column names to remove invalid characters for XGBoost
                df_processed.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_').replace('-', '_') for col in df_processed.columns]
            
            # Scale features if requested
            if self.config['analysis']['preprocessing']['scale_features']:
                # Only scale numeric features after encoding
                numeric_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                if self.target_column in numeric_features:
                    numeric_features.remove(self.target_column)
                
                if numeric_features:
                    df_processed[numeric_features] = self.scaler.fit_transform(df_processed[numeric_features])
            
            self.feature_names = [col for col in df_processed.columns if col != self.target_column]
            
            logger.info(f"Feature preprocessing complete. Features: {len(self.feature_names)}")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error in feature preprocessing: {e}")
            raise
    
    def perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for coral recovery prediction from bleaching data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        try:
            df_engineered = df.copy()
            
            # Create depth-based features
            if 'depth_m' in df_engineered.columns:
                depth_bins = pd.cut(df_engineered['depth_m'], 
                                  bins=[0, 1.5, 3, 4, float('inf')], 
                                  labels=[0, 1, 2, 3])  # Use numeric labels
                df_engineered['depth_category'] = depth_bins.astype(float)
                df_engineered['is_deep_site'] = (df_engineered['depth_m'] > 3).astype(int)
            
            # Create thermal stress features from DHW data
            dhw_cols = ['dhw', 'mean_dhw', 'max_dhw']
            available_dhw_cols = [col for col in dhw_cols if col in df_engineered.columns]
            
            if available_dhw_cols:
                if len(available_dhw_cols) > 1:
                    if 'max_dhw' in df_engineered.columns and 'dhw' in df_engineered.columns:
                        df_engineered['dhw_range'] = df_engineered['max_dhw'] - df_engineered['dhw']
                
                # Create thermal stress categories
                if 'dhw' in df_engineered.columns:
                    thermal_bins = pd.cut(df_engineered['dhw'], 
                                        bins=[0, 2, 4, 8, float('inf')], 
                                        labels=[0, 1, 2, 3])  # Use numeric labels
                    df_engineered['thermal_stress_level'] = thermal_bins.astype(float)
            
            # Create sedimentation features from g/day columns
            sedimentation_cols = [col for col in df_engineered.columns if 'g_day' in col and '_mean' in col]
            if sedimentation_cols:
                # Average sedimentation across all days
                df_engineered['avg_sedimentation'] = df_engineered[sedimentation_cols].mean(axis=1)
                df_engineered['max_sedimentation'] = df_engineered[sedimentation_cols].max(axis=1)
                df_engineered['sedimentation_variability'] = df_engineered[sedimentation_cols].std(axis=1)
                
                # High sedimentation indicator
                sed_threshold = df_engineered['avg_sedimentation'].quantile(0.75)
                df_engineered['high_sedimentation'] = (df_engineered['avg_sedimentation'] > sed_threshold).astype(int)
            
            # Create wave energy features
            wave_cols = ['rms_mean', 'rms_std', 'height_mean', 'height_std']
            available_wave_cols = [col for col in wave_cols if col in df_engineered.columns]
            
            if available_wave_cols:
                if 'rms_mean' in df_engineered.columns and 'height_mean' in df_engineered.columns:
                    # Handle NaN values before multiplication
                    rms_clean = df_engineered['rms_mean'].fillna(0)
                    height_clean = df_engineered['height_mean'].fillna(0)
                    df_engineered['wave_energy_index'] = rms_clean * height_clean
                
                if 'rms_std' in df_engineered.columns and 'height_std' in df_engineered.columns:
                    rms_std_clean = df_engineered['rms_std'].fillna(0)
                    height_std_clean = df_engineered['height_std'].fillna(0)
                    df_engineered['wave_variability'] = rms_std_clean + height_std_clean
            
            # Create coral health features
            health_cols = [col for col in df_engineered.columns if 'coral_health' in col]
            if health_cols:
                # Average health across measurements
                numeric_health_cols = [col for col in health_cols if df_engineered[col].dtype in ['float64', 'int64']]
                if numeric_health_cols:
                    df_engineered['avg_coral_health'] = df_engineered[numeric_health_cols].mean(axis=1)
            
            # Create coral size features
            if 'coral_length_cm_mean' in df_engineered.columns and 'coral_width_cm_mean' in df_engineered.columns:
                length_clean = df_engineered['coral_length_cm_mean'].fillna(df_engineered['coral_length_cm_mean'].median())
                width_clean = df_engineered['coral_width_cm_mean'].fillna(df_engineered['coral_width_cm_mean'].median())
                
                df_engineered['coral_area_estimate'] = length_clean * width_clean
                df_engineered['coral_aspect_ratio'] = length_clean / (width_clean + 0.001)  # Avoid division by zero
            
            # Create spatial features if coordinates available
            if 'lat' in df_engineered.columns and 'lon' in df_engineered.columns:
                # Distance from center point
                center_lat = df_engineered['lat'].mean()
                center_lon = df_engineered['lon'].mean()
                df_engineered['distance_from_center'] = np.sqrt(
                    (df_engineered['lat'] - center_lat)**2 + 
                    (df_engineered['lon'] - center_lon)**2
                )
            
            # Create species-based features
            if 'sub' in df_engineered.columns:
                # Create binary features for common species
                species_counts = df_engineered['sub'].value_counts()
                common_species = species_counts.head(3).index.tolist()
                
                for species in common_species:
                    df_engineered[f'is_{species.replace(" ", "_").lower()}'] = (df_engineered['sub'] == species).astype(int)
            
            # Create advanced interaction features
            if 'depth_m' in df_engineered.columns and 'dhw' in df_engineered.columns:
                depth_clean = df_engineered['depth_m'].fillna(df_engineered['depth_m'].median())
                dhw_clean = df_engineered['dhw'].fillna(df_engineered['dhw'].median())
                df_engineered['depth_thermal_interaction'] = depth_clean * (1 / (dhw_clean + 1))
                df_engineered['depth_thermal_stress'] = depth_clean * dhw_clean
                df_engineered['thermal_protection_index'] = depth_clean / (dhw_clean + 0.1)
            
            if 'avg_sedimentation' in df_engineered.columns and 'depth_m' in df_engineered.columns:
                depth_clean = df_engineered['depth_m'].fillna(df_engineered['depth_m'].median())
                sed_clean = df_engineered['avg_sedimentation'].fillna(df_engineered['avg_sedimentation'].median())
                df_engineered['depth_sedimentation_interaction'] = depth_clean / (sed_clean + 0.001)
                df_engineered['sedimentation_stress_index'] = sed_clean * (1 / (depth_clean + 0.1))
            
            # Create coral resilience features
            if 'coral_length_cm_mean' in df_engineered.columns and 'coral_health_initial_mean' in df_engineered.columns:
                length_clean = df_engineered['coral_length_cm_mean'].fillna(df_engineered['coral_length_cm_mean'].median())
                health_clean = df_engineered['coral_health_initial_mean'].fillna(df_engineered['coral_health_initial_mean'].median())
                df_engineered['size_health_interaction'] = length_clean * health_clean
                df_engineered['resilience_index'] = (length_clean * health_clean) / (dhw_clean + 1) if 'dhw' in df_engineered.columns else length_clean * health_clean
            
            # Create temporal health change features
            health_cols = ['coral_health_initial_mean', 'coral_health_t1_mean', 'coral_health_t2_mean']
            available_health_cols = [col for col in health_cols if col in df_engineered.columns]
            
            if len(available_health_cols) >= 2:
                if 'coral_health_t1_mean' in df_engineered.columns and 'coral_health_initial_mean' in df_engineered.columns:
                    df_engineered['health_change_t1'] = df_engineered['coral_health_t1_mean'] - df_engineered['coral_health_initial_mean']
                
                if 'coral_health_t2_mean' in df_engineered.columns and 'coral_health_t1_mean' in df_engineered.columns:
                    df_engineered['health_change_t2'] = df_engineered['coral_health_t2_mean'] - df_engineered['coral_health_t1_mean']
                
                if 'coral_health_t2_mean' in df_engineered.columns and 'coral_health_initial_mean' in df_engineered.columns:
                    df_engineered['total_health_change'] = df_engineered['coral_health_t2_mean'] - df_engineered['coral_health_initial_mean']
            
            # Create environmental stress composite features
            stress_factors = []
            if 'dhw' in df_engineered.columns:
                stress_factors.append('dhw')
            if 'avg_sedimentation' in df_engineered.columns:
                stress_factors.append('avg_sedimentation')
            
            if len(stress_factors) >= 2:
                # Normalize stress factors and create composite
                for factor in stress_factors:
                    factor_clean = df_engineered[factor].fillna(df_engineered[factor].median())
                    factor_normalized = (factor_clean - factor_clean.min()) / (factor_clean.max() - factor_clean.min() + 0.001)
                    df_engineered[f'{factor}_normalized'] = factor_normalized
                
                df_engineered['composite_stress_index'] = df_engineered[[f'{f}_normalized' for f in stress_factors]].mean(axis=1)
            
            # Create polynomial features for key variables
            if 'dhw' in df_engineered.columns:
                dhw_clean = df_engineered['dhw'].fillna(df_engineered['dhw'].median())
                df_engineered['dhw_squared'] = dhw_clean ** 2
                df_engineered['dhw_log'] = np.log1p(dhw_clean)  # log(1+x) to handle zeros
            
            if 'depth_m' in df_engineered.columns:
                depth_clean = df_engineered['depth_m'].fillna(df_engineered['depth_m'].median())
                df_engineered['depth_squared'] = depth_clean ** 2
                df_engineered['depth_log'] = np.log1p(depth_clean)
            
            # Create ratio features
            if 'coral_length_cm_mean' in df_engineered.columns and 'coral_width_cm_mean' in df_engineered.columns:
                length_clean = df_engineered['coral_length_cm_mean'].fillna(df_engineered['coral_length_cm_mean'].median())
                width_clean = df_engineered['coral_width_cm_mean'].fillna(df_engineered['coral_width_cm_mean'].median())
                df_engineered['coral_size_ratio'] = length_clean / (width_clean + 0.001)
                df_engineered['coral_perimeter_estimate'] = 2 * (length_clean + width_clean)
            
            # Create binned features for continuous variables
            if 'dhw' in df_engineered.columns:
                dhw_clean = df_engineered['dhw'].fillna(df_engineered['dhw'].median())
                df_engineered['dhw_bin'] = pd.cut(dhw_clean, bins=5, labels=False).astype(float)
            
            if 'depth_m' in df_engineered.columns:
                depth_clean = df_engineered['depth_m'].fillna(df_engineered['depth_m'].median())
                df_engineered['depth_bin'] = pd.cut(depth_clean, bins=5, labels=False).astype(float)
            
            logger.info(f"Feature engineering complete. New shape: {df_engineered.shape}")
            return df_engineered
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def perform_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform feature selection using mutual information.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info("Starting feature selection...")
        
        try:
            # Get feature selection parameters
            method = self.config['analysis']['feature_selection']['method']
            k_best = self.config['analysis']['feature_selection']['k_best']
            
            if method == 'mutual_info':
                # Only select numeric features for feature selection
                numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_features) == 0:
                    logger.warning("No numeric features found for feature selection")
                    return X, X.columns.tolist()
                
                X_numeric = X[numeric_features]
                
                # Handle any remaining NaN values
                X_numeric = X_numeric.fillna(X_numeric.median())
                
                # Use mutual information for feature selection
                selector = SelectKBest(score_func=mutual_info_classif, k=min(k_best, X_numeric.shape[1]))
                X_selected_numeric = selector.fit_transform(X_numeric, y)
                
                # Get selected feature names
                selected_numeric_features = X_numeric.columns[selector.get_support()].tolist()
                
                # Combine selected numeric features with all categorical features
                categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
                all_selected_features = selected_numeric_features + categorical_features
                
                X_selected_df = X[all_selected_features].copy()
                
                logger.info(f"Feature selection complete. Selected {len(selected_numeric_features)} numeric + {len(categorical_features)} categorical = {len(all_selected_features)} total features")
                return X_selected_df, all_selected_features
            
            elif method == 'rfe':
                # Use Recursive Feature Elimination with Random Forest
                numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_features) == 0:
                    logger.warning("No numeric features found for RFE")
                    return X, X.columns.tolist()
                
                X_numeric = X[numeric_features].fillna(X[numeric_features].median())
                
                # Use Random Forest as base estimator
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                selector = RFE(rf, n_features_to_select=min(k_best, X_numeric.shape[1]))
                selector.fit(X_numeric, y)
                
                selected_numeric_features = X_numeric.columns[selector.support_].tolist()
                categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
                all_selected_features = selected_numeric_features + categorical_features
                
                X_selected_df = X[all_selected_features].copy()
                
                logger.info(f"RFE feature selection complete. Selected {len(selected_numeric_features)} numeric + {len(categorical_features)} categorical = {len(all_selected_features)} total features")
                return X_selected_df, all_selected_features
            
            elif method == 'tree_based':
                # Use tree-based feature importance
                numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_features) == 0:
                    logger.warning("No numeric features found for tree-based selection")
                    return X, X.columns.tolist()
                
                X_numeric = X[numeric_features].fillna(X[numeric_features].median())
                
                # Use Random Forest for feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                selector = SelectFromModel(rf, max_features=min(k_best, X_numeric.shape[1]))
                selector.fit(X_numeric, y)
                
                selected_numeric_features = X_numeric.columns[selector.get_support()].tolist()
                categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
                all_selected_features = selected_numeric_features + categorical_features
                
                X_selected_df = X[all_selected_features].copy()
                
                logger.info(f"Tree-based feature selection complete. Selected {len(selected_numeric_features)} numeric + {len(categorical_features)} categorical = {len(all_selected_features)} total features")
                return X_selected_df, all_selected_features
            
            else:
                logger.info("No feature selection applied")
                return X, X.columns.tolist()
                
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with optional resampling.
        
        Args:
            df: Input DataFrame with features and target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train/test sets...")
        
        try:
            # Separate features and target
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Split data
            test_size = self.config['data']['test_size']
            random_state = self.config['data']['random_state']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if len(y.unique()) > 1 else None
            )
            
            # Apply resampling to training data if configured and available
            resampling_method = self.config['analysis']['preprocessing'].get('resampling', None)
            
            if resampling_method and len(y_train.unique()) > 1 and IMBLEARN_AVAILABLE:
                logger.info(f"Applying {resampling_method} resampling to training data...")
                X_train, y_train = self._apply_resampling(X_train, y_train, resampling_method)
                logger.info(f"After resampling - Train: {X_train.shape}")
                logger.info(f"New class distribution: {y_train.value_counts(normalize=True).to_dict()}")
            elif resampling_method and not IMBLEARN_AVAILABLE:
                logger.warning("Resampling requested but imbalanced-learn not available. Skipping resampling.")
            
            logger.info(f"Data split complete. Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    def _apply_resampling(self, X_train: pd.DataFrame, y_train: pd.Series, method: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling techniques to handle class imbalance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            method: Resampling method ('smote', 'adasyn', 'smoteenn', 'undersample')
            
        Returns:
            Tuple of resampled (X_train, y_train)
        """
        try:
            random_state = self.config['data']['random_state']
            
            # Only use numeric features for resampling
            numeric_features = X_train.select_dtypes(include=[np.number]).columns
            X_numeric = X_train[numeric_features].copy()
            
            # Handle any remaining NaN values
            X_numeric = X_numeric.fillna(X_numeric.median())
            
            if method.lower() == 'smote':
                sampler = SMOTE(random_state=random_state, k_neighbors=min(5, len(y_train[y_train==1])-1))
            elif method.lower() == 'adasyn':
                sampler = ADASYN(random_state=random_state, n_neighbors=min(5, len(y_train[y_train==1])-1))
            elif method.lower() == 'smoteenn':
                sampler = SMOTEENN(random_state=random_state)
            elif method.lower() == 'undersample':
                sampler = RandomUnderSampler(random_state=random_state)
            else:
                logger.warning(f"Unknown resampling method: {method}")
                return X_train, y_train
            
            # Apply resampling
            X_resampled, y_resampled = sampler.fit_resample(X_numeric, y_train)
            
            # Convert back to DataFrame
            X_resampled_df = pd.DataFrame(X_resampled, columns=numeric_features)
            
            # Add back categorical features (replicate for new samples)
            categorical_features = X_train.select_dtypes(exclude=[np.number]).columns
            if len(categorical_features) > 0:
                # For new samples, use mode of minority class
                minority_class = y_train.value_counts().idxmin()
                minority_mask = y_train == minority_class
                minority_categorical = X_train.loc[minority_mask, categorical_features].mode().iloc[0]
                
                # Create categorical data for resampled dataset
                categorical_resampled = pd.DataFrame(
                    index=X_resampled_df.index,
                    columns=categorical_features
                )
                
                # Fill with original values for original samples and mode for new samples
                original_indices = X_resampled_df.index[:len(X_train)]
                new_indices = X_resampled_df.index[len(X_train):]
                
                if len(original_indices) > 0:
                    categorical_resampled.loc[original_indices] = X_train[categorical_features].values
                
                if len(new_indices) > 0:
                    for col in categorical_features:
                        categorical_resampled.loc[new_indices, col] = minority_categorical[col]
                
                # Combine numeric and categorical
                X_resampled_df = pd.concat([X_resampled_df, categorical_resampled], axis=1)
            
            # Ensure column order matches original
            X_resampled_df = X_resampled_df[X_train.columns]
            
            return X_resampled_df, pd.Series(y_resampled, name=y_train.name)
            
        except Exception as e:
            logger.error(f"Error applying resampling: {e}")
            return X_train, y_train


class FeatureAnalyzer:
    """
    Class for analyzing feature importance and generating visualizations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FeatureAnalyzer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config['analysis']['output_dir']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info("FeatureAnalyzer initialized")
    
    def analyze_feature_importance(self, feature_importance: Dict[str, float], 
                                 model_name: str) -> pd.DataFrame:
        """
        Analyze and rank feature importance scores.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            model_name: Name of the model
            
        Returns:
            DataFrame with ranked feature importance
        """
        logger.info(f"Analyzing feature importance for {model_name}")
        
        try:
            # Create DataFrame from feature importance
            importance_df = pd.DataFrame(
                list(feature_importance.items()),
                columns=['feature', 'importance']
            )
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df['rank'] = range(1, len(importance_df) + 1)
            importance_df['model'] = model_name
            
            # Get top N features
            top_n = self.config['analysis']['feature_importance_top_n']
            top_features = importance_df.head(top_n)
            
            logger.info(f"Top {len(top_features)} features identified for {model_name}")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            raise
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                              model_name: str, top_n: int = None) -> None:
        """
        Create feature importance visualization.
        
        Args:
            importance_df: DataFrame with feature importance scores
            model_name: Name of the model
            top_n: Number of top features to plot
        """
        logger.info(f"Creating feature importance plot for {model_name}")
        
        try:
            if top_n is None:
                top_n = self.config['analysis']['feature_importance_top_n']
            
            # Get top features
            top_features = importance_df.head(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            # Save plot
            if self.config['analysis']['save_plots']:
                for fmt in self.config['output']['plot_formats']:
                    filename = f"{self.output_dir}/feature_importance_{model_name.lower()}.{fmt}"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    logger.info(f"Feature importance plot saved: {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            raise
    
    def plot_data_distribution(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Create data distribution visualizations.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
        """
        logger.info("Creating data distribution plots")
        
        try:
            # Target distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            df[target_col].value_counts().plot(kind='bar')
            plt.title('Target Variable Distribution')
            plt.xlabel('Recovery Status')
            plt.ylabel('Count')
            
            # Numeric features distribution
            plt.subplot(1, 2, 2)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                # Plot distribution of a few key features
                key_features = [col for col in ['depth_m', 'dhw', 'avg_sedimentation'] if col in numeric_cols]
                if key_features:
                    for i, col in enumerate(key_features[:3]):
                        plt.subplot(2, 2, i+2)
                        df[col].hist(bins=20, alpha=0.7)
                        plt.title(f'{col} Distribution')
                        plt.xlabel(col)
                        plt.ylabel('Frequency')
            
            plt.tight_layout()
            
            # Save plot
            if self.config['analysis']['save_plots']:
                for fmt in self.config['output']['plot_formats']:
                    filename = f"{self.output_dir}/data_distribution.{fmt}"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    logger.info(f"Data distribution plot saved: {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating data distribution plots: {e}")
            raise
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """
        Create correlation matrix heatmap.
        
        Args:
            df: Input DataFrame
        """
        logger.info("Creating correlation matrix plot")
        
        try:
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] > 1:
                # Calculate correlation matrix
                corr_matrix = numeric_df.corr()
                
                # Create heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                
                # Save plot
                if self.config['analysis']['save_plots']:
                    for fmt in self.config['output']['plot_formats']:
                        filename = f"{self.output_dir}/correlation_matrix.{fmt}"
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        logger.info(f"Correlation matrix plot saved: {filename}")
                
                plt.show()
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix plot: {e}")
            raise


def main():
    """
    Main function to demonstrate the data processing pipeline.
    """
    try:
        # Initialize processor
        processor = CoralDataProcessor()
        
        # Load data
        datasets = processor.load_coral_data()
        
        # Merge datasets
        merged_df = processor.merge_datasets(datasets)
        
        # Create target variable
        df_with_target = processor.create_target_variable(merged_df)
        
        # Feature engineering
        df_engineered = processor.perform_feature_engineering(df_with_target)
        
        # Preprocess features
        df_processed = processor.preprocess_features(df_engineered)
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(df_processed)
        
        # Feature selection
        X_train_selected, selected_features = processor.perform_feature_selection(X_train, y_train)
        X_test_selected = X_test[selected_features]
        
        # Initialize analyzer
        analyzer = FeatureAnalyzer(processor.config)
        
        # Create visualizations
        analyzer.plot_data_distribution(df_processed, processor.target_column)
        analyzer.plot_correlation_matrix(df_processed)
        
        logger.info("Data processing pipeline completed successfully!")
        
        return {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': selected_features,
            'processor': processor,
            'analyzer': analyzer
        }
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()