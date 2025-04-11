# Ames Housing Price Prediction Pipeline

## Overview

This Python script implements a complete machine learning pipeline to predict house prices based on the Ames Housing dataset. It handles various stages, including data loading, exploratory data analysis (EDA), feature engineering, data preprocessing, model training, and evaluation. The script is structured using Python classes for modularity and includes logging for tracking the process.

## Features

*   **Modular Design:** Organized into classes for different pipeline stages (`DataLoader`, `ExploratoryDataAnalysis`, `DataPreprocessor`, `ModelTrainer`, `Logger`).
*   **Data Loading:** Loads the dataset from a CSV file.
*   **Exploratory Data Analysis (EDA):**
    *   Calculates basic descriptive statistics.
    *   Identifies and reports missing values.
    *   Analyzes correlations between numerical features and the target variable (`SalePrice`).
    *   Generates and saves visualizations:
        *   Correlation heatmap for top features.
        *   Sale price distribution histogram.
*   **Data Preprocessing:**
    *   Removes the `Id` column.
    *   **Feature Engineering:** Creates new features like `HouseAge` and `YearsSinceReno`. Encodes ordinal quality features (e.g., `ExterQual`, `KitchenQual`).
    *   **Missing Value Handling:** Imputes missing values using a simple `fillna(0)` strategy (see **Important Considerations** below).
    *   **Feature Selection:** Selects a predefined subset of features based on initial analysis or domain knowledge.
    *   **Scaling:** Scales numerical features using `StandardScaler`.
*   **Model Training & Evaluation:**
    *   Splits data into training and testing sets.
    *   Trains multiple regression models:
        *   Linear Regression
        *   Ridge Regression
        *   Random Forest Regressor
    *   Evaluates models using standard metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² Score.
    *   Performs 5-fold cross-validation on the training set to assess model generalization (using R² score).
    *   Generates and saves a feature importance plot for the Random Forest model.
*   **Logging:** Implements detailed logging to track pipeline execution, saving logs to a file and printing info/errors to the console. Log files are timestamped and stored in a `logs/` directory.

## Requirements

*   Python 3.x
*   Required Python libraries:
    *   `numpy`
    *   `pandas`
    *   `matplotlib`
    *   `seaborn`
    *   `scikit-learn`

## Installation

1.  **Clone or download the script:** Ensure you have the `houseprice.py` file.
2.  **Install dependencies:** Open your terminal or command prompt and run:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    # or pip3 install numpy pandas matplotlib seaborn scikit-learn
    ```

## Input Data

*   The expected input is a CSV file conforming to the structure of the Ames Housing dataset provided in the Kaggle competition "House Prices - Advanced Regression Techniques".

## Usage

1.  **Modify the data path:** Open `houseprice.py` and change the path passed to the `DataLoader` class to your local path for `train.csv`.
2.  **Navigate to the directory** containing the script in your terminal.
3.  **Run the script:**
    ```bash
    python houseprice.py
    # or python3 houseprice.py
    ```
4.  The script will execute the entire pipeline, printing logs to the console and saving detailed logs and output plots to the current directory (or a `logs/` subdirectory for log files).

## Outputs

*   **Console Output:** Logs indicating the progress of each stage (Loading, EDA, Preprocessing, Training) and final model performance metrics.
*   **Log Files:** Timestamped log files stored in a `logs/` directory (created if it doesn't exist) containing detailed execution information.
*   **Image Files:**
    *   `correlation_heatmap.png`: Heatmap of top correlated features.
    *   `price_distribution.png`: Histogram of the `SalePrice`.
    *   `feature_importance.png`: Bar plot showing feature importances from the Random Forest model.

## License

MIT License

## Author

Andrew Obwocha
