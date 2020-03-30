import logging
from housing_model import HousingModel

from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV

import pandas as pd

# this should go to a config file with all the other (future) hyper-parameters
random_state = 237
n_split = 10


class LinearRidge(HousingModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test, cv=False):
        """
        IModel's predict functions Linear Ridge Regression implementation.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
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

        # splitting the data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, train_size=0.9, test_size=0.1, random_state=random_state
        )

        def dse_mae(y_true, y_pred):
            return mean_absolute_error(
                self.inverse_transform(y_true, mu, sigma), self.inverse_transform(y_pred, mu, sigma)
            )

        scoring = make_scorer(dse_mae, greater_is_better=False)

        # Linear Ridge Regression
        if cv:
            model = RidgeCV(alphas=(0.1, 0.3, 1.0, 3.0, 10.0), cv=n_split, scoring=scoring)
            model.fit(x_train, y_train)
        else:
            model = Ridge(random_state=random_state)
            model.fit(x_train, y_train)

        # make predictions for training data and transform
        y_valid = self.inverse_transform(y_valid, mu, sigma)
        prediction = self.inverse_transform(pd.DataFrame(model.predict(x_valid), columns=['SalePrice']), mu, sigma)

        logging.info(
            f"{'RidgeCV' if cv else 'Ridge'} Regression validation MAE: {mean_absolute_error(y_valid, prediction)}"
        )

        test_pred = self.inverse_transform(pd.DataFrame(model.predict(x_test), columns=['SalePrice']), mu, sigma)

        test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})
        test_pred.to_csv(f'submissions\\submission_{"ridge_cv" if cv else "ridge"}.csv', index=False)

        logging.info("DONE!")
