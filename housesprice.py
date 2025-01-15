# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_df = df[numerical_cols]
categorical_df = df[[column for column in df.columns if column not in numerical_cols]]

numerical_df.head()

# %%
categorical_df.head()

# %%
numerical_df.describe()

# %%
numerical_df = numerical_df.drop('Id', axis=1)
df = df.drop('Id', axis=1)
numerical_df.columns

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='MSSubClass', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against MSSubClass')
plt.xlabel('MSSubClass')
plt.ylabel('SalePrice')
plt.show()

# %%
numerical_df = numerical_df[numerical_df['LotFrontage'] <= 200]
df = df[df['LotFrontage'] <= 200]

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='LotFrontage', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against LotFrontage')
plt.xlabel('LotFrontage')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['LotFrontage'], bins=20, kde=True)
plt.title('Histogram of LotFrontage')
plt.xlabel('LotFrontage')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='LotFrontage', data=numerical_df)
plt.title('Boxplot of LotFrontage')
plt.xlabel('LotFrontage')
plt.ylabel('Count')
plt.show()

# %%
numerical_df = numerical_df[numerical_df['LotArea'] <= 30000]
df = df[df['LotArea'] <= 30000]

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='LotArea', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against LotArea')
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['LotArea'], bins=20, kde=True)
plt.title('Histogram of LotArea')
plt.xlabel('LotArea')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='LotArea', data=numerical_df)
plt.title('Boxplot of LotArea')
plt.xlabel('LotArea')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against YearBuilt')
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='YearRemodAdd', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against YearRemodAdd')
plt.xlabel('YearRemodAdd')
plt.ylabel('SalePrice')
plt.show()

# %%
numerical_df['MasVnrArea'] = np.sqrt(numerical_df['MasVnrArea'])
df['MasVnrArea'] = np.sqrt(df['MasVnrArea'])
skewed_cols = ['MasVnrArea']
transformed_cols = ['MasVnrArea']

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against MasVnrArea')
plt.xlabel('MasVnrArea')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['MasVnrArea'], bins=10, kde=True)
plt.title('Histogram of MasVnrArea')
plt.xlabel('MasVnrArea')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='MasVnrArea', data=numerical_df)
plt.title('Boxplot of MasVnrArea')
plt.xlabel('MasVnrArea')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['BsmtFinSF1'] = np.sqrt(numerical_df['BsmtFinSF1'])
df['BsmtFinSF1'] = np.sqrt(df['BsmtFinSF1'])
transformed_cols.append('BsmtFinSF1')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='BsmtFinSF1', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against BsmtFinSF1')
plt.xlabel('BsmtFinSF1')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['BsmtFinSF1'], bins=10, kde=True)
plt.title('Histogram of BsmtFinSF1')
plt.xlabel('BsmtFinSF1')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='BsmtFinSF1', data=numerical_df)
plt.title('Boxplot of BsmtFinSF1')
plt.xlabel('BsmtFinSF1')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['BsmtFinSF2'] = np.sqrt(numerical_df['BsmtFinSF2'])
df['BsmtFinSF2'] = np.sqrt(df['BsmtFinSF2'])
skewed_cols.append('BsmtFinSF2')
transformed_cols.append('BsmtFinSF2')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='BsmtFinSF2', y='SalePrice', data=numerical_df)
plt.title('Scatterplot of SalePrice against BsmtFinSF2')
plt.xlabel('BsmtFinSF2')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['BsmtFinSF2'], bins=20, kde=True)
plt.title('Histogram for BsmtFinSF2')
plt.xlabel('BsmtFinSF2')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='BsmtFinSF2', data=numerical_df)
plt.title('Boxplot of BsmtFinSF2')
plt.xlabel('BsmtFinSF2')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['BsmtUnfSF'] = np.sqrt(numerical_df['BsmtUnfSF'])
df['BsmtUnfSF'] = np.sqrt(df['BsmtUnfSF'])

transformed_cols.append('BsmtUnfSF')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='BsmtUnfSF', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against BsmtUnfSF')
plt.xlabel('BsmtUnfSF')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['BsmtUnfSF'], bins=10, kde=True)
plt.title('Histogram of BsmtUnfSF')
plt.xlabel('BsmtUnfSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='BsmtUnfSF', data=numerical_df)
plt.title('Boxplot of BsmtUnfSF')
plt.xlabel('BsmtUnfSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against TotalBsmtSF')
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['TotalBsmtSF'], bins=10, kde=True)
plt.title('Histogram of TotalBsmtSF')
plt.xlabel('TotalBsmtSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='TotalBsmtSF', data=numerical_df)
plt.title('Boxplot of TotalBsmtSF')
plt.xlabel('TotalBsmtSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against 1stFlrSF')
plt.xlabel('1stFlrSF')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['1stFlrSF'], bins=10, kde=True)
plt.title('Histogram of 1stFlrSF')
plt.xlabel('1stFlrSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='1stFlrSF', data=numerical_df)
plt.title('Boxplot of 1stFlrSF')
plt.xlabel('1stFlrSF')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['2ndFlrSF'] = np.sqrt(numerical_df['2ndFlrSF'])
df['2ndFlrSF'] = np.sqrt(df['2ndFlrSF'])

skewed_cols.append('2ndFlrSF')
transformed_cols.append('2ndFlrSF')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='2ndFlrSF', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against 2ndFlrSF')
plt.xlabel('2ndFlrSF')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['2ndFlrSF'], bins=10, kde=True)
plt.title('Histogram of 2ndFlrSF')
plt.xlabel('2ndFlrSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='2ndFlrSF', data=numerical_df)
plt.title('Boxplot of 2ndFlrSF')
plt.xlabel('2ndFlrSF')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['LowQualFinSF'] = np.sqrt(numerical_df['LowQualFinSF'])
df['LowQualFinSF'] = np.sqrt(df['LowQualFinSF'])

skewed_cols.append('LowQualFinSF')
transformed_cols.append('LowQualFinSF')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='LowQualFinSF', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against LowQualFinSF')
plt.xlabel('LowQualFinSF')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['LowQualFinSF'], bins=10, kde=True)
plt.title('Histogram of LowQualFinSF')
plt.xlabel('LowQualFinSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='LowQualFinSF', data=numerical_df)
plt.title('Boxplot of LowQualFinSF')
plt.xlabel('LowQualFinSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against GrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['GrLivArea'], bins=10, kde=True)
plt.title('Histogram of GrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='GrLivArea', data=numerical_df)
plt.title('Boxplot of GrLivArea')
plt.xlabel('GrLivArea')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='BsmtFullBath', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against BsmtFullBath')
plt.xlabel('BsmtFullBath')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GarageCars', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against GarageCars')
plt.xlabel('GarageCars')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='GarageArea', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against GarageArea')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['GarageArea'], bins=10, kde=True)
plt.title('Histogram of GarageArea')
plt.xlabel('GarageArea')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='GarageArea', data=numerical_df)
plt.title('Boxplot of GarageArea')
plt.xlabel('GarageArea')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['WoodDeckSF'] = np.sqrt(numerical_df['WoodDeckSF'])
df['WoodDeckSF'] = np.sqrt(df['WoodDeckSF'])
transformed_cols.append('WoodDeckSF')
skewed_cols.append('WoodDeckSF')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='WoodDeckSF', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against WoodDeckSF')
plt.xlabel('WoodDeckSF')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['WoodDeckSF'], bins=10, kde=True)
plt.title('Histogram of WoodDeckSF')
plt.xlabel('WoodDeckSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='WoodDeckSF', data=numerical_df)
plt.title('Boxplot of WoodDeckSF')
plt.xlabel('WoodDeckSF')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['OpenPorchSF'] = np.sqrt(numerical_df['OpenPorchSF'])
df['OpenPorchSF'] = np.sqrt(df['OpenPorchSF'])

transformed_cols.append('OpenPorchSF')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='OpenPorchSF', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against OpenPorchSF')
plt.xlabel('OpenPorchSF')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['OpenPorchSF'], bins=10, kde=True)
plt.title('Histogram of OpenPorchSF')
plt.xlabel('OpenPorchSF')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='OpenPorchSF', data=numerical_df)
plt.title('Boxplot of OpenPorchSF')
plt.xlabel('OpenPorchSF')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['EnclosedPorch'] = np.sqrt(numerical_df['EnclosedPorch'])
df['EnclosedPorch'] = np.sqrt(df['EnclosedPorch'])

transformed_cols.append('EnclosedPorch')
skewed_cols.append('EnclosedPorch')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='EnclosedPorch', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against EnclosedPorch')
plt.xlabel('EnclosedPorch')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['EnclosedPorch'], bins=10, kde=True)
plt.title('Histogram of EnclosedPorch')
plt.xlabel('EnclosedPorch')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='EnclosedPorch', data=numerical_df)
plt.title('Boxplot of EnclosedPorch')
plt.xlabel('EnclosedPorch')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['3SsnPorch'] = np.sqrt(numerical_df['3SsnPorch'])
df['3SsnPorch'] = np.sqrt(df['3SsnPorch'])

transformed_cols.append('3SsnPorch')
skewed_cols.append('3SsnPorch')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='3SsnPorch', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against 3SsnPorch')
plt.xlabel('3SsnPorch')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['3SsnPorch'], bins=10, kde=True)
plt.title('Histogram of 3SsnPorch')
plt.xlabel('3SsnPorch')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='3SsnPorch', data=numerical_df)
plt.title('Boxplot of 3SsnPorch')
plt.xlabel('3SsnPorch')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['ScreenPorch'] = np.sqrt(numerical_df['ScreenPorch'])
transformed_cols.append('ScreenPorch')
skewed_cols.append('ScreenPorch')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='ScreenPorch', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against ScreenPorch')
plt.xlabel('ScreenPorch')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['ScreenPorch'], bins=10, kde=True)
plt.title('Histogram of ScreenPorch')
plt.xlabel('ScreenPorch')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='ScreenPorch', data=numerical_df)
plt.title('Boxplot of ScreenPorch')
plt.xlabel('ScreenPorch')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['PoolArea'] = np.sqrt(numerical_df['PoolArea'])
transformed_cols.append('PoolArea')
skewed_cols.append('PoolArea')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PoolArea', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against PoolArea')
plt.xlabel('PoolArea')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['PoolArea'], bins=10, kde=True)
plt.title('Histogram of PoolArea')
plt.xlabel('PoolArea')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='PoolArea', data=numerical_df)
plt.title('Boxplot of PoolArea')
plt.xlabel('PoolArea')
plt.ylabel('Count')
plt.show()

# %%
numerical_df['MiscVal'] = np.sqrt(numerical_df['MiscVal'])
transformed_cols.append('MiscVal')
skewed_cols.append('MiscVal')

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='MiscVal', y='SalePrice', data=numerical_df)
plt.title('Scatterpolot of SalePrice against MiscVal')
plt.xlabel('MiscVal')
plt.ylabel('SalePrice')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(numerical_df['MiscVal'], bins=10, kde=True)
plt.title('Histogram of MiscVal')
plt.xlabel('MiscVal')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='MiscVal', data=numerical_df)
plt.title('Boxplot of MiscVal')
plt.xlabel('MiscVal')
plt.ylabel('Count')
plt.show()

# %%
transformed_cols

# %%
skewed_cols

# %%
numerical_df = numerical_df[[column for column in numerical_df if column not in skewed_cols]]

# %%
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation of features')
plt.show()

# %%
high_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            high_correlations.append({
                'Feature 1' : correlation_matrix.columns[i],
                'Feature 2' : correlation_matrix.columns[j],
                'Correlation' : correlation_matrix.iloc[i, j]
            }) 

high_corr_df = pd.DataFrame(high_correlations).sort_values(
    by='Correlation',
    key=abs,
    ascending=False
)

high_corr_df

# %%
high_corr_df[high_corr_df['Feature 1'] == 'SalePrice']

# %%
current_year = df['YrSold']
df['HouseAge'] = current_year - df['YearBuilt']
df['YearsSinceReno'] = current_year - df['YearRemodAdd']

# %%
numerical_cols = [
    'OverallQual', 
    'GrLivArea', 
    'TotalBsmtSF', 
    'GarageCars', 
    'YearBuilt',
    'HouseAge',
    'YearsSinceReno',
    'SalePrice'
]

# %%
df.columns

# %%
df_columns = []
for col in df.columns:
    if col not in df.select_dtypes(include=['int64', 'float64']) or col in numerical_cols:
        df_columns.append(col)

df = df[df_columns]
        

# %%
df.columns

# %%
quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1,
    'NA': 0
}

quality_cols = ['ExterQual', 'KitchenQual', 'BsmtQual']
for col in quality_cols:
    df[f'{col}_encoded'] = df[col].map(quality_mapping)

categorical_cols = [
    'ExterQual_encoded', 
    'KitchenQual_encoded',
    'BsmtQual_encoded',    
]

df_columns = []
for col in df.columns:
    if col in df.select_dtypes(include=['int64', 'float64']) or col in categorical_cols:
        df_columns.append(col)

df = df[df_columns]


# %%
df.columns

# %%
df['BsmtQual_encoded'] = df['BsmtQual_encoded'].fillna(0)
df.isnull().sum()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.3f}")
