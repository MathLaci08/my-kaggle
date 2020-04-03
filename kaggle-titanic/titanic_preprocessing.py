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
    Many ideas below came from: https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
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

        # replace SibSp and Parch with Family size
        numerical_vars = numerical_vars.drop(['SibSp', 'Parch', 'Fare', 'Age'])
        numerical_vars = numerical_vars.insert(0, 'Family size').insert(0, 'FareBins').insert(0, 'AgeBins')

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
        self.X.drop(['PassengerId', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)
        test_ids = self.X_test.PassengerId
        self.X_test.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

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

        self.X['Age'] = self.X['Age'].fillna(self.X['Age'].median())
        self.X_test['Age'] = self.X_test['Age'].fillna(self.X['Age'].median())

        self.X['Embarked'] = self.X['Embarked'].fillna(self.X['Embarked'].mode()[0])
        self.X_test['Embarked'] = self.X_test['Embarked'].fillna(self.X['Embarked'].mode()[0])

        self.X['Fare'] = self.X['Fare'].fillna(self.X['Fare'].median())
        self.X_test['Fare'] = self.X_test['Fare'].fillna(self.X['Fare'].median())

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

            # encode Name column
            name_cols = data['Name'].apply(
                lambda x: pd.Series(
                    [str(x).split(",")[0], str(x).split(", ")[1].split(".")[0]], index=['Family name', 'Title']
                )
            )
            data = data.join(name_cols)

            # identify Titles with same meaning
            data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}, inplace=True)

            # group rare Titles
            title_names = (data['Title'].value_counts() < 10)
            data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] else x)

            # create Family size and Alone column from SibSp, Parch cols
            data['Family size'] = data['SibSp'] + data['Parch'] + 1
            data['Alone'] = data['Family size'] == 1

            # make 5 equal size groups from Fares
            data['Fare'] = pd.qcut(data['Fare'], 5, labels=False)

            # make 5 groups from Ages
            data['Age'] = pd.cut(data['Age'], 5, labels=False)

            # rename columns and delete unnecessary features
            data = data.rename(columns={'Sex': 'Male', 'Fare': 'FareBins', 'Age': 'AgeBins'})
            data.drop(['Name', 'SibSp', 'Parch'], axis=1, inplace=True)

            return data

        self.X = encode(self.X)
        self.X_test = encode(self.X_test)

        for col in self.X.columns:
            if self.X[col].dtype != 'float64':
                table = self.X.join(self.y)[[col, 'Survived']].groupby(col, as_index=False).mean()
                table['Survived'] = (table['Survived'] * 100).map('{:.2f} %'.format)
                logging.info(f'Survival ratio by: {col}\n{table}\n{"-" * 10}\n')

        one_hot_encoder = OneHotEncoder(use_cat_names=True)
        one_hot_columns = one_hot_encoder.fit_transform(self.X[['Title', 'Embarked']])
        one_hot_columns_test = one_hot_encoder.transform(self.X_test[['Title', 'Embarked']])
        self.X = self.X.join(one_hot_columns)
        self.X_test = self.X_test.join(one_hot_columns_test)

        self.X.drop(['Family name', 'Title', 'Embarked'], axis=1, inplace=True)
        self.X_test.drop(['Family name', 'Title', 'Embarked'], axis=1, inplace=True)

        logging.info(f'#{self._step_index} - DONE!')
