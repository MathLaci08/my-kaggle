import logging
from titanic_model import TitanicModel

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC

import pandas as pd

# this should go to a config file with all the other (future) hyper-parameters
random_state = 237
n_split = 10


class SVClassifier(TitanicModel):
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
        y = y.Survived.values
        y_train = y_train.Survived.values

        # Linear Ridge Regression
        if cv:
            model = SVC(kernel='rbf', random_state=random_state)

            scores = cross_validate(model, x, y, cv=n_split, scoring='accuracy', return_estimator=True)

            estimators = scores["estimator"]
            logs = [f'Model accuracy {i + 1} of {n_split}: {scores["test_score"][i] * 100:.2f}%' for i in range(n_split)]
            logging.info('\nSVC cross validation MAE:\n' + '\n'.join(logs))

            test_pred = None

            for e in estimators:
                new_pred = pd.DataFrame(e.predict(x_test), columns=['Survived'])
                if test_pred is None:
                    test_pred = new_pred
                else:
                    test_pred += new_pred

            test_pred = (test_pred / len(estimators)).astype(int)
            test_pred = pd.DataFrame({'PassengerId': test_ids, 'Survived': test_pred.Survived})
        else:
            model = SVC(kernel='rbf', random_state=random_state)
            model.fit(x_train, y_train)

            prediction = pd.DataFrame(model.predict(x_valid), columns=['Survived'])

            accuracy_percent = accuracy_score(y_valid, prediction, normalize=True) * 100
            logging.info(
                f"{'SVC_CV' if cv else 'SVC'} validation accuracy: {accuracy_percent:.2f}%"
            )

            test_pred = pd.DataFrame({'PassengerId': test_ids, 'Survived': model.predict(x_test)})

        test_pred.astype(int).to_csv(f'submissions\\submission_{"svc_cv" if cv else "svc"}.csv', index=False)

        logging.info("DONE!")
