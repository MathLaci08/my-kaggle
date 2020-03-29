import pandas as pd
import numpy as np
import logging
from typing import Union
from i_model import IModel

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, make_scorer

random_state = 237
n_split = 5


class XGB(IModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test, cv=False):
        """
        IModel's predict functions XGBoost implementation.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """

        logging.info("Performing XGBoost prediction...")
        # cut off target variable and id from the data
        y = pd.DataFrame(x['SalePrice'], columns=['SalePrice'])
        x.drop(['SalePrice'], axis=1, inplace=True)

        test_ids = x_test.Id
        x_test.drop(['Id'], axis=1, inplace=True)

        # standardize target variable
        mu = y.mean()
        sigma = y.std()

        y = (y - mu) / sigma

        def dse_mae(y_true, y_pred):
            return mean_absolute_error(
                self.inverse_transform(y_true, mu, sigma), self.inverse_transform(y_pred, mu, sigma)
            )

        scoring = make_scorer(dse_mae, greater_is_better=False)

        # XGB Regression
        if cv:
            model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=random_state)

            fit_params = {
                'eval_set': [(x, y)],
                'eval_metric': 'mae',
                'early_stopping_rounds': 5,
                'verbose': False
            }
            scores = cross_validate(
                model, x, y, cv=n_split, scoring=scoring, return_estimator=True, fit_params=fit_params
            )
            estimators = scores["estimator"]
            logs = [f'Model evaluation {i + 1} of {n_split}: {-1 * scores["test_score"][i]}' for i in range(n_split)]
            logging.info('\nXGBoost cross validation MAE:\n' + '\n'.join(logs))

            test_pred = None

            for e in estimators:
                new_pred = self.inverse_transform(pd.DataFrame(e.predict(x_test), columns=['SalePrice']), mu, sigma)
                if test_pred is None:
                    test_pred = new_pred
                else:
                    test_pred += new_pred

            test_pred = (test_pred / len(estimators)).astype(int)
        else:
            model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=random_state)

            # splitting the data into training and validation sets
            x_train, x_valid, y_train, y_valid = train_test_split(
                x, y, train_size=0.9, test_size=0.1, random_state=random_state
            )

            model.fit(
                x_train, y_train, eval_set=[(x_train, y_train)],
                eval_metric='mae', early_stopping_rounds=5, verbose=False
            )

            # make predictions for validation data, then transform
            y_valid = self.inverse_transform(y_valid, mu, sigma)
            prediction = self.inverse_transform(pd.DataFrame(model.predict(x_valid), columns=['SalePrice']), mu, sigma)

            logging.info(f"XGBoost validation MAE: {mean_absolute_error(y_valid, prediction)}")

            test_pred = self.inverse_transform(pd.DataFrame(model.predict(x_test), columns=['SalePrice']), mu, sigma)

        test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': test_pred})
        test_pred.to_csv(f'submissions\\submission_{"XGB_cv" if cv else "XGB"}.csv', index=False)

        logging.info("DONE!")
