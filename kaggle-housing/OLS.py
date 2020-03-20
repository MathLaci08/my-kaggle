import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

import pathlib

train_csv = pathlib.Path(__file__, "..\\data\\train.csv").resolve()
test_csv = pathlib.Path(__file__, "..\\data\\test.csv").resolve()
random_state = 237

# read data from the provided csv files
X = pd.read_csv(train_csv)
X_test = pd.read_csv(test_csv)

# drop rows without labels
X.dropna(axis=0, subset=['SalePrice'], inplace=True)

# set labels to variable y
y = X.SalePrice
X.drop(['Id', 'SalePrice', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

# splitting the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=random_state)

# get column names for categorical and numerical features
categorical_vars = X.select_dtypes(include='object').columns.append(pd.Index(['MSSubClass']))
numerical_vars = X.columns.difference(categorical_vars)

# design transformers for categorical and numerical data and then transform
numerical_imputer = SimpleImputer(strategy='mean')

X_train_num = pd.DataFrame(numerical_imputer.fit_transform(X_train[numerical_vars]), columns=numerical_vars)
X_valid_num = pd.DataFrame(numerical_imputer.transform(X_valid[numerical_vars]), columns=numerical_vars)

categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

X_train_cat = pd.DataFrame(categorical_imputer.fit_transform(X_train[categorical_vars]), columns=categorical_vars)
X_train_cat = pd.DataFrame(categorical_encoder.fit_transform(X_train_cat), columns=categorical_encoder.get_feature_names(categorical_vars))
X_valid_cat = pd.DataFrame(categorical_imputer.transform(X_valid[categorical_vars]), columns=categorical_vars)
X_valid_cat = pd.DataFrame(categorical_encoder.transform(X_valid_cat), columns=X_train_cat.columns)

X_train_encoded = X_train_num.join(X_train_cat)
X_valid_encoded = X_valid_num.join(X_valid_cat)

std_scaler = StandardScaler()

X_train = pd.DataFrame(std_scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns)
X_valid = pd.DataFrame(std_scaler.transform(X_valid_encoded), columns=X_train_encoded.columns)

X_train = sm.add_constant(X_train, has_constant='add')
X_valid = sm.add_constant(X_valid, has_constant='add')

# create OLS model, then fit the data
model = sm.OLS(list(y_train), X_train)
results = model.fit()
print(results.summary())

# exclude non-significant features
pval = results.pvalues
significant_cols = pval[pval < 0.05].axes[0]

model = sm.OLS(list(y_train), X_train[significant_cols])
results = model.fit()
print(results.summary())

pval = results.pvalues
significant_cols = pval[pval < 0.05].axes[0]

model = sm.OLS(list(y_train), X_train[significant_cols])
results = model.fit()
print(results.summary())

# make predicitons for validation data
predictions = results.predict(X_valid[significant_cols])
print(predictions)
print(y_valid)

# plot the difference between the ground truth and the prediction and then print the MAE for the validation set
plt.plot(predictions.subtract(y_valid.values))

plt.show()

print("Mean abs error: ", mean_absolute_error(y_valid.values, predictions))
print("Significant features: ", list(significant_cols))
