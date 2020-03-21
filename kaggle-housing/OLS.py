import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd

# this should go to a config file with all the other (future) hyper-parameters
random_state = 237


def predict(x, x_test):
    # cut off target variable and id from the data
    y = pd.DataFrame(x['SalePrice'], columns=['SalePrice'])
    x.drop(['SalePrice'], axis=1, inplace=True)

    test_ids = x_test.Id
    x_test.drop(['Id'], axis=1, inplace=True)

    # splitting the data into training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, train_size=0.9, test_size=0.1, random_state=random_state
    )

    x_train = sm.add_constant(x_train)
    x_valid = sm.add_constant(x_valid)
    x_test = sm.add_constant(x_test)

    # OLS
    model = sm.OLS(list(y_train.SalePrice), x_train)
    results = model.fit()

    # print(results.summary())

    # make predictions for validation data
    predictions = results.predict(x_valid)

    print("Mean abs error: ", mean_absolute_error(np.exp(y_valid.SalePrice.values), np.exp(predictions)))

    test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': np.exp(results.predict(x_test))})
    test_pred.to_csv('submission.csv', index=False)
