import logging
from i_model import IModel

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import pandas as pd

# this should go to a config file with all the other (future) hyper-parameters
random_state = 237
n_split = 10


class LogisticReg(IModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test, cv=False):
        """
        IModel's predict functions Logistic Regression implementation.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """

        logging.info("Performing Logistic Regression prediction...")
        # cut off target variable and id from the data
        y = pd.DataFrame(x['Survived'], columns=['Survived'])
        x.drop(['Survived'], axis=1, inplace=True)

        test_ids = x_test.PassengerId
        x_test.drop(['PassengerId'], axis=1, inplace=True)

        # splitting the data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, train_size=0.9, test_size=0.1, random_state=random_state
        )
        y_train = y_train.Survived.values

        # Linear Ridge Regression
        if cv:
            model = LogisticRegressionCV(
                scoring='accuracy', cv=n_split, random_state=random_state, penalty='l2', max_iter=1000
            )
            model.fit(x_train, y_train)
        else:
            model = LogisticRegression(random_state=random_state, penalty='l2', max_iter=1000)
            model.fit(x_train, y_train)

        # make predictions for training data and transform
        prediction = pd.DataFrame(model.predict(x_valid), columns=['Survived'])

        accuracy_percent = accuracy_score(y_valid, prediction, normalize=True) * 100
        logging.info(
            f"{'LogisticCV' if cv else 'Logistic'} Regression validation accuracy: {accuracy_percent:.2f}%"
        )

        test_pred = pd.DataFrame({'PassengerId': test_ids, 'Survived': model.predict(x_test)})
        test_pred.astype(int).to_csv(f'submissions\\submission_{"logistic_cv" if cv else "logistic"}.csv', index=False)

        logging.info("DONE!")
