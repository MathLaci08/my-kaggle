import statsmodels.api as sm
import logging
from housing_model import HousingModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import pandas as pd

# this should go to a config file with all the other (future) hyper-parameters
random_state = 237


class OLS(HousingModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test, cv=False):
        """
        IModel's predict functions OLS implementation.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """

        logging.info("Performing OLS prediction...")
        if cv:
            logging.warning("Cross validation not implemented for OLS.")

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
            x, y, train_size=0.9, test_size=0.1, random_state=random_state
        )

        x_train = sm.add_constant(x_train)
        x_valid = sm.add_constant(x_valid)
        x_test = sm.add_constant(x_test)

        # OLS
        model = sm.OLS(list(y_train.SalePrice), x_train)
        results = model.fit()

        # make predictions for validation data and transform
        y_valid = self.inverse_transform(y_valid, mu, sigma)
        prediction = self.inverse_transform(pd.DataFrame(results.predict(x_valid), columns=['SalePrice']), mu, sigma)

        logging.info(f"OLS validation MAE: {mean_absolute_error(y_valid, prediction)}")

        test_pred = self.inverse_transform(pd.DataFrame(results.predict(x_test), columns=['SalePrice']), mu, sigma)
        test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})
        test_pred.to_csv('submissions\\submission_OLS.csv', index=False)
