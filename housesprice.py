# House Price Prediction Pipeline
# Author: [Your Name]
# Date: January 15, 2025
# Description: Modular ML pipeline for predicting house prices using the Ames Housing dataset


"""
House Price Prediction Model
--------------------------------
This script performs exploratory data analysis and builds a prediction model
for the Ames Housing dataset. It includes data cleaning, feature engineering,
and a linear regression model to predict house price.

Author: Andrew Obwocha
Date: January 15, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import logging
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class Logger:
    """
    Custom logger class for the house price prediction pipeline
    """
    def __init__(self, name, log_dir='logs'):
        """
        Initialize logger with custom formatting and handlers
        
        Args:
            name (str): Logger name
            log_dir (str): Directory to store log files
        """
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            f'{log_dir}/house_price_prediction_{timestamp}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def get_logger(self):
        """Return logger instance"""
        return self.logger

class DataLoader:
    """
    Class to handle data loading and initial splitting
    """
    def __init__(self, data_path):
        """
        Initialize data loader with path and logger
        
        Args:
            data_path (str): Path to the training data CSV file
        """
        self.data_path = data_path
        self.logger = Logger('DataLoader').get_logger()
        self.df = None
        self.numerical_df = None
        self.categorical_df = None
        
    def load_data(self):
        """Load and perform initial data splitting"""
        self.logger.info("Loading dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            self.logger.info(f"Loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            
            # Split into numerical and categorical dataframes
            self.numerical_df = self.df.select_dtypes(include=['int64', 'float64'])
            self.categorical_df = self.df.select_dtypes(exclude=['int64', 'float64'])
            
            return self.df, self.numerical_df, self.categorical_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

class ExploratoryDataAnalysis:
    """
    Class to handle all exploratory data analysis tasks
    """
    def __init__(self, df, numerical_df, categorical_df):
        """
        Initialize EDA class with dataframes and logger
        
        Args:
            df (pd.DataFrame): Complete dataset
            numerical_df (pd.DataFrame): Numerical features
            categorical_df (pd.DataFrame): Categorical features
        """
        self.df = df
        self.numerical_df = numerical_df
        self.categorical_df = categorical_df
        self.logger = Logger('EDA').get_logger()
        
    def run_eda(self):
        """Execute complete EDA pipeline"""
        self.logger.info("Starting Exploratory Data Analysis...")
        
        try:
            # Basic statistics
            self._analyze_basic_stats()
            
            # Missing values
            self._analyze_missing_values()
            
            # Correlation analysis
            self._analyze_correlations()
            
            # Visualizations
            self._create_visualizations()
            
        except Exception as e:
            self.logger.error(f"Error in EDA: {str(e)}")
            raise
            
    def _analyze_basic_stats(self):
        """Analyze basic statistics of numerical features"""
        self.logger.info("Analyzing basic statistics...")
        stats = self.numerical_df.describe()
        self.logger.debug(f"\nNumerical Data Summary:\n{stats}")
        
    def _analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        self.logger.info(f"\nFeatures with missing values:\n{missing_values}")
        
    def _analyze_correlations(self):
        """Analyze feature correlations with target variable"""
        correlations = self.numerical_df.corr()['SalePrice'].sort_values(ascending=False)
        self.logger.info(f"\nTop 10 correlated features with Sale Price:\n{correlations[:10]}")
        
    def _create_visualizations(self):
        """Create and save important visualizations"""
        self.logger.info("Creating visualizations...")
        
        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        top_corr_features = self.numerical_df.corr()['SalePrice'].sort_values(ascending=False)[:10].index
        correlation_matrix = self.numerical_df[top_corr_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap - Top Features')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        # Price distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['SalePrice'], kde=True)
        plt.title('Sale Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.savefig('price_distribution.png')
        plt.close()

class DataPreprocessor:
    """
    Class to handle all data preprocessing tasks
    """
    def __init__(self, df):
        """
        Initialize preprocessor with dataframe and logger
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.df = df.copy()
        self.logger = Logger('Preprocessor').get_logger()
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self):
        """Execute complete preprocessing pipeline"""
        self.logger.info("Starting data preprocessing...")
        
        try:
            # Remove unnecessary columns
            self._remove_unnecessary_columns()
            
            # Feature engineering
            self._engineer_features()
            
            # Handle missing values
            self._handle_missing_values()
            
            # Select features
            self._select_features()
            
            # Scale features
            self._scale_features()
            
            return self.X, self.y
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
            
    def _remove_unnecessary_columns(self):
        """Remove ID and other unnecessary columns"""
        if 'Id' in self.df.columns:
            self.df = self.df.drop('Id', axis=1)
            self.logger.debug("Removed Id column")
            
    def _engineer_features(self):
        """Create new features"""
        self.logger.info("Engineering new features...")
        
        # Age-related features
        self.df['HouseAge'] = self.df['YrSold'] - self.df['YearBuilt']
        self.df['YearsSinceReno'] = self.df['YrSold'] - self.df['YearRemodAdd']
        
        # Quality encodings
        quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
        quality_cols = ['ExterQual', 'KitchenQual', 'BsmtQual']
        
        for col in quality_cols:
            self.df[f'{col}_encoded'] = self.df[col].map(quality_mapping)
            
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        self.logger.info("Handling missing values...")
        self.df = self.df.fillna(0)
        
    def _select_features(self):
        """Select final feature set"""
        selected_features = [
            'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
            'YearBuilt', 'HouseAge', 'YearsSinceReno', 'ExterQual_encoded',
            'KitchenQual_encoded', 'BsmtQual_encoded'
        ]
        
        self.X = self.df[selected_features]
        self.y = self.df['SalePrice']
        self.logger.info(f"Selected {len(selected_features)} features")
        
    def _scale_features(self):
        """Scale features using StandardScaler"""
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns
        )
        self.logger.info("Features scaled using StandardScaler")

class ModelTrainer:
    """
    Class to handle model training and evaluation
    """
    def __init__(self, X, y):
        """
        Initialize trainer with features, target, and logger
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.X = X
        self.y = y
        self.logger = Logger('ModelTrainer').get_logger()
        self.models = {}
        self.results = {}
        
    def train_models(self):
        """Train and evaluate multiple models"""
        self.logger.info("Starting model training...")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            # Initialize models
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            # Train and evaluate each model
            for name, model in models.items():
                self.logger.info(f"\nTraining {name}...")
                self._train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
                
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
            
    def _train_and_evaluate(self, name, model, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model"""
        # Train model
        model.fit(X_train, y_train)
        self.models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='r2'
        )
        
        # Store results
        self.results[name] = {
            'metrics': metrics,
            'cv_scores': {
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
        }
        
        # Log results
        self._log_results(name, metrics, cv_scores)
        
        # Plot feature importance for Random Forest
        if name == 'Random Forest':
            self._plot_feature_importance(model)
            
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate model performance metrics"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
    def _log_results(self, name, metrics, cv_scores):
        """Log model results"""
        self.logger.info(f"\nPerformance Metrics for {name}:")
        self.logger.info(f"RMSE: ${metrics['rmse']:,.2f}")
        self.logger.info(f"MAE: ${metrics['mae']:,.2f}")
        self.logger.info(f"R² Score: {metrics['r2']:.3f}")
        self.logger.info(f"Cross-validation R² scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
    def _plot_feature_importance(self, model):
        """Plot feature importance for tree-based models"""
        importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Initialize logger
    main_logger = Logger('Main').get_logger()
    main_logger.info("Starting House Price Prediction Pipeline...")
    
    try:
        # Load data
        data_loader = DataLoader('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
        df, numerical_df, categorical_df = data_loader.load_data()
        
        # Perform EDA
        eda = ExploratoryDataAnalysis(df, numerical_df, categorical_df)
        eda.run_eda()
        
        # Preprocess data
        preprocessor = DataPreprocessor(df)
        X, y = preprocessor.preprocess_data()
        
        # Train models
        trainer = ModelTrainer(X, y)
        results = trainer.train_models()
        
        main_logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        main_logger.error(f"Pipeline failed: {str(e)}")
        raise