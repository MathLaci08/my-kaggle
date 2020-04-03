import pandas as pd
import numpy as np
import logging
from titanic_model import TitanicModel

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score

random_state = 237
n_split = 5


class XGB(TitanicModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test, cv=False):
        """
        IModel's predict functions XGBoost implementation.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """

        logging.info("Performing XGBClassifier prediction...")
        # cut off target variable and id from the data
        y = pd.DataFrame(x['Survived'], columns=['Survived'])
        x.drop(['Survived'], axis=1, inplace=True)

        test_ids = x_test.PassengerId
        x_test.drop(['PassengerId'], axis=1, inplace=True)

        model = XGBClassifier(n_estimators=1000, learning_rate=0.01, random_state=random_state)

        # splitting the data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, train_size=0.9, test_size=0.1, random_state=random_state
        )
        y_train = y_train.Survived.values
        y_valid = y_valid.Survived.values

        fit_params = {
            'eval_set': [(x_valid, y_valid)],
            'eval_metric': 'mae',
            'early_stopping_rounds': 5,
            'verbose': False
        }

        # XGB Regression
        if cv:
            scores = cross_validate(
                model, x_train, y_train, cv=n_split, scoring='accuracy', return_estimator=True, fit_params=fit_params
            )
            estimators = scores["estimator"]
            logs = [
                f'Model evaluation {i + 1} of {n_split}: {scores["test_score"][i] * 100:.2f}%' for i in range(n_split)
            ]
            logging.info('\nXGBClassifier cross validation accuracy:\n' + '\n'.join(logs))

            test_pred = None

            for e in estimators:
                new_pred = e.predict(x_test)
                if test_pred is None:
                    test_pred = new_pred
                else:
                    test_pred += new_pred

            test_pred = np.around(test_pred / len(estimators)).astype(int)
        else:
            model.fit(x_train, y_train, **fit_params)

            # make predictions for validation data, then transform
            accuracy_percent = accuracy_score(y_valid, model.predict(x_valid)) * 100
            logging.info(f"XGBClassifier validation accuracy: {accuracy_percent}%")

            test_pred = model.predict(x_test)

        test_pred = pd.DataFrame({'PassengerId': test_ids, 'Survived': test_pred})
        test_pred.to_csv(f'submissions\\submission_{"XGB_cv" if cv else "XGB"}.csv', index=False)

        logging.info("DONE!")
