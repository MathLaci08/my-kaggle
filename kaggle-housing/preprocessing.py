import pandas as pd

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

import numpy as np

import pathlib
from math import sqrt, ceil

train_csv = pathlib.Path(__file__, "..\\data\\train.csv").resolve()
test_csv = pathlib.Path(__file__, "..\\data\\test.csv").resolve()
random_state = 237

# read data from the provided csv files
X = pd.read_csv(train_csv)
X_test = pd.read_csv(test_csv)

# drop rows without labels
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

# set labels to variable y
y = pd.DataFrame(X.SalePrice, columns=['SalePrice'])
X.drop(['Id', 'Utilities', 'SalePrice'], axis=1, inplace=True)
X_test.drop(['Id', 'Utilities'], axis=1, inplace=True)

# make MSSubClass categorical
X['MSSubClass'] = X['MSSubClass'].apply(str)
X_test['MSSubClass'] = X_test['MSSubClass'].apply(str)

# get numerical features
numerical_vars = X.select_dtypes(exclude="object").columns
nv_for_detection = pd.Index([f for f in numerical_vars if X[f].nunique() > 500])
cols = ceil(sqrt(nv_for_detection.size))
rows = ceil(nv_for_detection.size // cols)

# detect outliers
fig, axes = plt.subplots(rows, cols)
fig.tight_layout(pad=2.0)

for feature in nv_for_detection:
    loc = nv_for_detection.get_loc(feature)
    axes[loc // cols][loc % cols].scatter(x=X[feature], y=y, s=5)
    axes[loc // cols][loc % cols].set(xlabel=feature, ylabel='SalePrice')
plt.show()

# deleting outliers
outliers = X[(X[nv_for_detection[5]] > 4000) & (y.SalePrice < 500000) | (X[nv_for_detection[0]] > 100000)].index
X = X.drop(outliers)
y = y.drop(outliers)

fig, axes = plt.subplots(rows, cols)
fig.tight_layout(pad=2.0)

for feature in nv_for_detection:
    loc = nv_for_detection.get_loc(feature)
    axes[loc // cols][loc % cols].scatter(x=X[feature], y=y, s=5)
    axes[loc // cols][loc % cols].set(xlabel=feature, ylabel='SalePrice')
plt.show()

# check the distribution of SalePrice
sns.distplot(y, fit=norm)

(mu, sigma) = norm.fit(y)

plt.legend([f'Normal distribution $\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# Get also the QQ-plot
plt.figure()
stats.probplot(y.SalePrice, plot=plt)
plt.show()

# distribution is right skewed, so logarithmic transformation will be made
y = np.log(y)

# check the distribution of the transformed SalePrice
sns.distplot(y, fit=norm)

(mu, sigma) = norm.fit(y)

plt.legend([f'Normal distribution $\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# Get also the QQ-plot
plt.figure()
stats.probplot(y.SalePrice, plot=plt)
plt.show()

# deal with missing values
print(f'Number of missing values before imputing: {X.isnull().sum().sum() + X_test.isnull().sum().sum()}')

fill_with_none = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType',
                  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

fill_with_zero = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                  '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces',
                  'GarageCars', 'GarageArea', 'GarageYrBlt', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                  'ScreenPorch', 'PoolArea', 'MiscVal']

fill_with_most_frequent = ['Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'MSZoning', 'SaleType']

for col in fill_with_none:
    X[col] = X[col].fillna('None')
    X_test[col] = X_test[col].fillna('None')

for col in fill_with_zero:
    X[col] = X[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)

for col in fill_with_most_frequent:
    X[col] = X[col].fillna(X[col].mode()[0])
    X_test[col] = X_test[col].fillna(X_test[col].mode()[0])

X['Functional'] = X['Functional'].fillna('Typ')
X_test['Functional'] = X_test['Functional'].fillna('Typ')

print(f'Number of missing values after imputing: {X.isnull().sum().sum() + X_test.isnull().sum().sum()}')

# correlation map between the remaining features
corr_map = X.join(y).corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corr_map, vmax=0.9, square=True)
plt.show()

# get column names for categorical and numerical features
categorical_vars = X.select_dtypes(include='object').columns
numerical_vars = X.columns.difference(categorical_vars)

X_num = X[numerical_vars]
X_test_num = X_test[numerical_vars]

# one hot encode categorical columns
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

X_cat = pd.DataFrame(
    one_hot_encoder.fit_transform(X[categorical_vars]),
    columns=one_hot_encoder.get_feature_names(categorical_vars)
)
X_test_cat = pd.DataFrame(
    one_hot_encoder.transform(X_test[categorical_vars]),
    columns=X_cat.columns
)

X = X_num.join(X_cat)
X_test = X_test_num.join(X_test_cat)

# check the skew of all numerical features
skewed_features = X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("Skew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_features})
print(skewness)

# transform skewed features
skewed_features = skewness[abs(skewness.Skew) > 0.75].index
print(f"There are {skewed_features.size} skewed features")

for feature in skewed_features:
    X[feature] = np.log1p(X[feature])
    X_test[feature] = np.log1p(X_test[feature])     # box-cox transformation instead?

# check the skew of all numerical features again
# skewed_features = X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("Skew in numerical features: \n")
# skewness = pd.DataFrame({'Skew': skewed_features})
# print(skewness)

# standardize data
std_scaler = StandardScaler()

X = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(std_scaler.transform(X_test), columns=X.columns)

# splitting the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=random_state)
