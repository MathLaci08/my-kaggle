import pandas as pd
import numpy as np
from i_model import IModel

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

random_state = 237


class DecisionTree(IModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test):
        # cut off target variable and id from the data
        y = pd.DataFrame(x['SalePrice'], columns=['SalePrice'])
        x.drop(['SalePrice'], axis=1, inplace=True)

        test_ids = x_test.Id
        x_test.drop(['Id'], axis=1, inplace=True)

        # standardize target variable
        mu = y.mean()
        sigma = y.std()

        y = (y - mu) / sigma

        # splitting the data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, train_size=0.8, test_size=0.2, random_state=random_state
        )

        model = DecisionTreeRegressor(criterion='mae', random_state=random_state)
        model.fit(x_train, y_train)

        # make predictions for validation data, then transform
        prediction = pd.DataFrame(model.predict(x_valid), columns=['SalePrice'])
        prediction = prediction * sigma + mu
        y_valid = y_valid * sigma + mu

        print("Decision tree MAE: ", mean_absolute_error(np.exp(y_valid.SalePrice.values), np.exp(prediction)))

        test_pred = pd.DataFrame(model.predict(x_test), columns=['SalePrice']) * sigma + mu
        test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': np.exp(test_pred.SalePrice)})
        test_pred.to_csv('submission_decision.csv', index=False)
