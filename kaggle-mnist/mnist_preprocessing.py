import pandas as pd
from i_preprocessing import IPreProcessing

import logging


class MnistPreProcessing(IPreProcessing):
    """
    Class for data pre-processing related methods.
    Many ideas below came from: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook
    """

    def process_data(self):
        """
        Method for determining the preprocessed data. If the data set haven't been preprocessed before, or forced to be
        ignored, the method calls all the necessary functions for the pre-processing.
        """

        logging.info('Processing data...')
        self._separate_target()
        self._imputer()
        self._standardize_data()
        self._pca()

        self.X = self.X.join(self.y)
        self.X_test = pd.DataFrame(
            range(1, self.X_test.shape[0] + 1), columns=['ImageId']
        ).astype(int).join(self.X_test)

        self._save_data()

    def _separate_target(self):
        """
        Private function for some preliminary steps. Drops non-labelled data, separates y from X and the test
        identifiers from the test set. Also converts the 'MSSubClass' feature to 'object' type (because it is
        categorical).
        :return: The test identifiers for future usage.
        """

        logging.info(f'#{self._index()} - Dropping unnecessary features...')
        # drop rows without labels
        self.X.dropna(axis=0, subset=['label'], inplace=True)

        # set labels to variable y
        self.y = pd.DataFrame(self.X.label.values, columns=['Label'])
        self.X.drop(['label'], axis=1, inplace=True)

        logging.info(f'#{self._step_index} - DONE!')

    def _detect_outliers(self):
        pass

    def _normalize_target(self):
        pass

    def _imputer(self):
        """
        Private function for dealing with missing values in the data sets. The method first creates lists of the
        feature names based on how to impute data in them, then fills the columns with appropriate values.
        """

        logging.info(f'#{self._index()} - Imputing appropriate values in empty cells...')
        # deal with missing values
        missing_values = self.X.isnull().sum() + self.X_test.isnull().sum()
        logging.info(f'Number of missing values: {missing_values.sum()}')

        logging.info(f'#{self._step_index} - DONE!')

    def _encode_categories(self):
        pass
