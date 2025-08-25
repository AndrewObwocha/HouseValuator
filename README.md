# House Valuator ML Application

## Welcome!
HouseValuator is an ML prediction application designed to forecast house prices in the central United States of America. Leveraging a linear regression, this ML pipeline provides an R^2 accuracy of 0.8. I aim to provide a realistic trend analysis of housing price evolution affecting that section of the economy.


## HouseValuator's Vision
My vision is to explore the trend associated with rising housing prices in North America to make informed investment decisions. The trend is proposed to be linear and positive; this analysis is to evaluate the validity of that claim.

## Features
- **Data Loading** — Loads the dataset from a CSV file.
- **Exploratory Data Analysis (EDA)**
    *   Calculates descriptive statistics.
    *   Identifies and reports missing values.
    *   Analyzes correlations between numerical features and the target variable (`SalePrice`).
    *   Generates and saves visualizations:
        *   Correlation heatmap for top features.
        *   Sale price distribution histogram.
- **Data Preprocessing**
    *   Removes the `Id` column.
    *   **Feature Engineering:** Creates new features like `HouseAge` and `YearsSinceReno`. Encodes ordinal quality features (e.g., `ExterQual`, `KitchenQual`).
    *   **Missing Value Handling:** Imputes missing values using a simple `fillna(0)` strategy (see **Important Considerations** below).
    *   **Feature Selection:** Selects a predefined subset of features based on initial analysis or domain knowledge.
    *   **Scaling:** Scales numerical features using `StandardScaler`.
- **Model Training & Evaluation**
    *   Splits data into training and testing sets.
    *   Trains multiple regression models:
        *   Linear Regression
        *   Ridge Regression
        *   Random Forest Regressor
    *   Evaluates models using standard metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² Score.
    *   Performs 5-fold cross-validation on the training set to assess model generalization (using R² score).
    *   Generates and saves a feature importance plot for the Random Forest model.
- **Logging** — Implements detailed logging to track pipeline execution, saving logs to a file and printing info/errors to the console. Log files are timestamped and stored in a `logs/` directory.

## Technologies Used
- **Python** — Primary programming language used to handle computations
- **Pandas** — Parsing data from csv file for easier data manipulation and basic mathematical operations
- **matplotlib** - Data visualization offering a more intuitive description of the data's shape
- **seaborn** — Improved visuals offering depth in mathematical diagrams and information available
- **scikit-learn** — Conduct training of multiple regression models
 
## Setup & Running

**Pre-Requisites**
- Download and install Python 3.8 from the official website — https://www.python.org/downloads/

**Setup**
1. **Clone the repository**
   ```sh
   git clone https://github.com/{yourUsername}/HouseValuator.git
   cd HouseValuator
   ```
2. **Create and activate virtual environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install the dependencies**
	 ```sh
	 pip3 install -r requirements.txt
	 ```
4.  **Modify the data path:** Open `houseprice.py` and change the path passed to the `DataLoader` class to your local path for `train.csv`.
5.  **Navigate to the directory** containing the script in your terminal.
6.  **Run the script:**
    ```bash
    python houseprice.py
    # or python3 houseprice.py
    ```
7.  The script will execute the entire pipeline, printing logs to the console and saving detailed logs and output plots to the current directory (or a `logs/` subdirectory for log files).

## Contributing

Contribution is not only welcome, but encouraged! Here are some ways you can contribute:

- **Idea requests** — You can send  ideas by opening an issue with the tag idea-request.
- **Bug reports** — You can report a bug by opening an issue with the tag bug
- **Pull requests** — You can contribute directly by forking, coding, and submitting PRs!

## License

This project is licensed under the MIT License.

For further information, feel free to initiate contact:

- **Email** — obwochandrew@gmail.com 
- **Project Link** — https://github.com/AndrewObwocha/HouseValuator
