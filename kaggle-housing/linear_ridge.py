import logging
from i_model import IModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV

import pandas as pd

# this should go to a config file with all the other (future) hyper-parameters
random_state = 237


class LinearRidge(IModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test):
        """
        IModel's predict functions Linear Ridge Regression implementation.

        :param x: training data set
        :param x_test: test data set
        """

        logging.info("Performing Linear Ridge Regression prediction...")
        # cut off target variable and id from the data
        y = pd.DataFrame(x['SalePrice'], columns=['SalePrice'])
        x.drop(['SalePrice'], axis=1, inplace=True)

        test_ids = x_test.Id
        x_test.drop(['Id'], axis=1, inplace=True)

        # standardize target variable
        mu = y.mean()
        sigma = y.std()

        y = (y - mu) / sigma

        # Linear Ridge Regression
        model = RidgeCV(fit_intercept=False)
        model.fit(x, y)

        # make predictions for validation data and transform
        y_valid = self.inverse_transform(y, mu, sigma)
        prediction = self.inverse_transform(pd.DataFrame(model.predict(x), columns=['SalePrice']), mu, sigma)

        logging.info(f"Linear Ridge Regression training MAE: {mean_absolute_error(y_valid, prediction)}")

        test_pred = self.inverse_transform(pd.DataFrame(model.predict(x_test), columns=['SalePrice']), mu, sigma)
        test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})
        test_pred.to_csv('submissions\\submission_ridge.csv', index=False)
