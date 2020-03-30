from i_preprocessing import IPreProcessing
import logging
from math import sqrt, ceil

import pandas as pd

from category_encoders import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class TitanicPreProcessing(IPreProcessing):
    """
    Class for data pre-processing related methods.
    """

    def process_data(self):
        """
        Method for determining the preprocessed data. If the data set haven't been preprocessed before, or forced to be
        ignored, the method calls all the necessary functions for the pre-processing.
        """

        logging.info('Processing data...')
        test_ids = self._separate_target()
        numerical_vars = self._detect_outliers()
        self._imputer()
        self._correlation_map()
        self._encode_categories()
        self._transform_skewed_features(numerical_vars)
        self._standardize_data()

        x_pca = self.X
        x_test_pca = self.X_test

        self.X = self.X.join(self.y)
        self.X_test = test_ids.to_frame().join(self.X_test)

        self._save_data()

        self.X = x_pca
        self.X_test = x_test_pca

        self._pca()

        self.X = self.X.join(self.y)
        self.X_test = test_ids.to_frame().join(self.X_test)

        self._save_data(prefix='pca')

    def _separate_target(self):
        """
        Private function for some preliminary steps. Drops non-labelled data, separates y from X and the test
        identifiers from the test set. Also converts the 'MSSubClass' feature to 'object' type (because it is
        categorical).

        :return: The test identifiers for future usage.
        """

        logging.info(f'#{self._index()} - Dropping unnecessary features...')
        # drop rows without labels
        self.X.dropna(axis=0, subset=['Survived'], inplace=True)

        # set labels to variable y
        self.y = pd.DataFrame(self.X.Survived, columns=['Survived'])
        self.X.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
        test_ids = self.X_test.PassengerId
        self.X_test.drop(['PassengerId'], axis=1, inplace=True)

        logging.info(f'#{self._step_index} - DONE!')

        return test_ids

    def _detect_outliers(self):
        """
        Private function for detecting the outliers in the data set. First determines those numerical variables which
        have much unique values, and then plots the target variable as the function of these features. Base on the
        graphs it drops the outliers from the data, resets indices for X and y and finally plots the functions again.

        :return: Set of all numerical feature names.
        """

        logging.info(f'#{self._index()} - Detecting outliers...')
        # get numerical features
        numerical_vars = self.X.select_dtypes(exclude="object").columns
        nv_for_detection = pd.Index(numerical_vars)
        cols = ceil(sqrt(nv_for_detection.size))
        rows = ceil(nv_for_detection.size / cols)

        # detect outliers
        fig, axes = plt.subplots(rows, cols)
        fig.tight_layout(pad=2.0)

        for feature in nv_for_detection:
            loc = nv_for_detection.get_loc(feature)
            axes[loc // cols][loc % cols].scatter(x=self.X[feature], y=self.y, s=5)
            axes[loc // cols][loc % cols].set(xlabel=feature, ylabel='Survived')

        plt.show()
        logging.info(f'#{self._step_index} - DONE!')

        # deleting outliers
        logging.info('There are no outliers!')

        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)

        return numerical_vars

    def _imputer(self):
        """
        Private function for dealing with missing values in the data sets. The method first creates lists of the
        feature names based on how to impute data in them, then fills the columns with appropriate values.
        """

        logging.info(f'#{self._index()} - Imputing appropriate values in empty cells...')
        # deal with missing values
        missing_values = self.X.isnull().sum() + self.X_test.isnull().sum()
        logging.info(f'Number of missing values before imputing: {missing_values.sum()}')

        plt.figure(figsize=(10, 5))
        sns.heatmap(self.X.isnull(), yticklabels=0, cbar=False, cmap='viridis')
        plt.show()

        missing_data = pd.DataFrame({'Missing Values': missing_values})
        print(missing_data.head(20))

        self.X['Age'] = self.X['Age'].fillna(self.X['Age'].mean())
        self.X_test['Age'] = self.X_test['Age'].fillna(self.X['Age'].mean())

        self.X['Cabin'] = self.X['Cabin'].fillna('None')
        self.X_test['Cabin'] = self.X_test['Cabin'].fillna('None')

        self.X['Embarked'] = self.X['Embarked'].fillna('S')
        self.X_test['Embarked'] = self.X_test['Embarked'].fillna('S')

        self.X['Fare'] = self.X['Fare'].fillna(self.X['Fare'].mean())
        self.X_test['Fare'] = self.X_test['Fare'].fillna(self.X['Fare'].mean())

        missing_values = self.X.isnull().sum().sum() + self.X_test.isnull().sum().sum()

        logging.info(f'Number of missing values after imputing: {missing_values}')
        logging.info(f'#{self._step_index} - DONE!')

    def _encode_categories(self):
        """
        This private method stands for encoding categorical variables. Label encoding used for ordinal categories and
        one-hot encoding used for nominal categories.
        """

        logging.info(f'#{self._index()} - Encoding categorical columns...')

        def encode(data):
            # encode Sex column
            data['Sex'] = data['Sex'] == 'male'
            data = data.rename(columns={'Sex': 'Male'})

            # encode Cabin column
            cabins = data['Cabin'].apply(
                lambda x: pd.Series(
                    [str(x)[0], len(str(x).split(" "))] if str(x) != 'None' else [chr(ord('A') - 1), 0],
                    index=['Cabin class', '#Cabins']
                )
            )
            cabins['Cabin class'] = cabins['Cabin class'].apply(lambda x: ord(x) - ord('A') + 1)
            data = data.join(cabins)
            data.drop(['Cabin'], axis=1, inplace=True)

            # encode Embarked column
            one_hot_embarked = data['Embarked'].apply(
                lambda x: pd.Series([x == 'S', x == 'C', x == 'Q'], index=['Southampton', 'Cherbourg', 'Queenstown'])
            )
            data = data.join(one_hot_embarked)
            data.drop(['Embarked'], axis=1, inplace=True)

            # encode Name column
            data['Name'] = data['Name'].apply(lambda x: str(x).split(",")[0])
            data = data.rename(columns={'Name': 'Surname'})

            # encode Ticket column
            special_types = ['Normal', 'LINE', 'S.O./P.P. 3']
            replaceable = str.maketrans(dict.fromkeys('./ '))
            data['Ticket'] = data['Ticket'].apply(
                lambda x: 'normal' if str(x).isnumeric() else x.lower()
            ).apply(
                lambda x: x if x in special_types else str(x).rsplit(' ', 1)[0].translate(replaceable)
            ).apply(
                lambda x: x.replace('soton', 's-').replace('ston', 's-').replace('sca', 'a')
            )

            return data

        self.X = encode(self.X)
        self.X_test = encode(self.X_test)

        # one_hot_encoder = OneHotEncoder(use_cat_names=True)
        # one_hot_columns = one_hot_encoder.fit_transform(self.X[['Surname', 'Ticket']])
        # one_hot_columns_test = one_hot_encoder.transform(self.X_test[['Surname', 'Ticket']])
        # self.X = self.X.join(one_hot_columns)
        # self.X_test = self.X_test.join(one_hot_columns_test)

        self.X.drop(['Surname', 'Ticket'], axis=1, inplace=True)
        self.X_test.drop(['Surname', 'Ticket'], axis=1, inplace=True)

        logging.info(f'#{self._step_index} - DONE!')
