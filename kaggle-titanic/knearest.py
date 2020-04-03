import logging
from titanic_model import TitanicModel

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

# this should go to a config file with all the other (future) hyper-parameters
random_state = 237
n_split = 10


class KNearest(TitanicModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test, cv=False):
        """
        IModel's predict functions Logistic Regression implementation.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """

        logging.info("Performing KNearestNeighbor prediction...")
        if cv:
            logging.warning("Cross validation not implemented for KNearestNeighbor.")

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
        y_valid = y_valid.Survived.values
        best_score = 0

        # KNearestNeighbor prediction
        for k in range(1, 10):
            model = KNeighborsClassifier(n_neighbors=k, p=2)
            model.fit(x_train, y_train)

            score = model.score(x_valid, y_valid) * 100
            if score > best_score:
                best_score = max(best_score, score)
                best_model = model
                logging.info(f'Best_model: {k}')

            logging.info(
                f"KNearestNeighbor classification accuracy for k={k}: {score:.2f}%"
            )

        test_pred = pd.DataFrame({'PassengerId': test_ids, 'Survived': best_model.predict(x_test)})
        test_pred.astype(int).to_csv(f'submissions\\submission_knearest.csv', index=False)

        logging.info("DONE!")
