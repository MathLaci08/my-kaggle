import abc
import logging
import pathlib
import importlib
from typing import Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.stats import skew
from scipy.special import boxcox1p


n_components = 9
box_cox_lambda = 0.15


class IPreProcessing(abc.ABC):
    """
    Class for data pre-processing related methods.
    """

    def __init__(self):
        """
        Class instance initializer method.
        """

        try:
            self.pp_X = None
            self.pp_X_test = None

            self.path = importlib.import_module(self.__module__).__file__
            train_csv = pathlib.Path(self.path, "..\\data\\train.csv").resolve()
            test_csv = pathlib.Path(self.path, "..\\data\\test.csv").resolve()

            # read data from the provided csv files
            self.X = pd.read_csv(train_csv)
            self.y = None
            self.X_test = pd.read_csv(test_csv)
            self._step_index = -1
        except FileNotFoundError as e:
            logging.error("Please download the data, before creating instance!")
            raise e

    def _index(self):
        self._step_index += 1
        return self._step_index

    @abc.abstractmethod
    def process_data(self) -> None:
        """
        Method for determining the preprocessed data. If the data set haven't been preprocessed before, or forced to be
        ignored, the method calls all the necessary functions for the pre-processing.
        """

        pass

    def load_data(self, with_pca: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the previously processed data from the saved csv files.

        :param with_pca: if True, function will return a data set on which pca was performed before
        :return: train and test set if data is preprocessed, else None.
        """

        try:
            logging.info('Trying to load data...')
            prefix = 'pp_pca' if with_pca else 'pp'
            pp_train_csv = pathlib.Path(self.path, f"..\\data\\{prefix}_train.csv").resolve()
            pp_test_csv = pathlib.Path(self.path, f"..\\data\\{prefix}_test.csv").resolve()

            self.pp_X = pd.read_csv(pp_train_csv)
            self.pp_X_test = pd.read_csv(pp_test_csv)
            logging.info('DONE!')

            return self.pp_X, self.pp_X_test
        except FileNotFoundError:
            logging.warning("Data is not preprocessed. Calling process_data() function...")
            self.process_data()
            return self.load_data(with_pca=with_pca)

    @abc.abstractmethod
    def _separate_target(self) -> np.ndarray:
        """
        Private function for some preliminary steps. Drops non-labelled data, separates y from X and the test
        identifiers from the test set. Also converts the numeric type categorical features to 'object' type.

        :return: The test identifiers for future usage.
        """

        pass

    @abc.abstractmethod
    def _detect_outliers(self) -> np.ndarray:
        """
        Private function for detecting the outliers in the data set. First determines those numerical variables which
        have much unique values, and then plots the target variable as the function of these features. Base on the
        graphs it drops the outliers from the data, resets indices for X and y and finally plots the functions again.

        :return: Set of all numerical feature names.
        """

        pass

    def _normalize_target(self) -> None:
        """
        This private function checks the distribution of the target variable and then (if necessary) transforms it with
        an appropriate transformation. Finally plots the distribution again.
        """

        pass

    @abc.abstractmethod
    def _imputer(self) -> None:
        """
        Private function for dealing with missing values in the data sets. The method first creates lists of the
        feature names based on how to impute data in them, then fills the columns with appropriate values.
        """

        pass

    def _correlation_map(self):
        """
        Private function for plotting the correlation map between the features.
        """

        logging.info(f'#{self._index()} - Checking correlation between features...')
        # correlation map between the remaining features
        corr_map = self.X.join(self.y).corr()
        plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_map, vmax=0.9, square=True)
        plt.show()
        logging.info(f'#{self._step_index} - DONE!')

    @abc.abstractmethod
    def _encode_categories(self) -> None:
        """
        This private method stands for encoding categorical variables. Label encoding used for ordinal categories and
        one-hot encoding used for nominal categories.
        """

        pass

    def _transform_skewed_features(self, numerical_vars: np.ndarray) -> None:
        """
        Private method for transforming features with high skew.

        :param numerical_vars: Set of all originally numerical variables.
        """

        logging.info(f'#{self._index()} - Determine and transform skewed features...')
        # check the skew of all numerical features
        skewed_features = self.X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        logging.info("Skew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_features})
        logging.info(skewness)

        # transform skewed features
        skewed_features = skewness[abs(skewness.Skew) > 0.75].index
        logging.info(f"There are {skewed_features.size} skewed features")

        for feature in skewed_features:
            self.X[feature] = boxcox1p(self.X[feature], box_cox_lambda)
            self.X_test[feature] = boxcox1p(self.X_test[feature], box_cox_lambda)

        # check the skew of all numerical features again
        skewed_features = self.X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        logging.info("Skew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_features})
        logging.info(skewness)
        logging.info(f'#{self._step_index} - DONE!')

    def _standardize_data(self) -> None:
        """
        This private function's job is the standardization of all the variables.
        """

        logging.info(f'#{self._index()} - Standardizing variables...')
        # standardize data
        std_scaler = StandardScaler(copy=False)

        self.X = pd.DataFrame(std_scaler.fit_transform(self.X), columns=self.X.columns)
        self.X_test = pd.DataFrame(std_scaler.transform(self.X_test), columns=self.X.columns)
        logging.info(f'#{self._step_index} - DONE!')

    def _pca(self) -> None:
        """
        This private function do the principal component analysis on our data, and as a result, dimension reduction
        will be made.
        """

        logging.info(f'#{self._index()} - Performing principal component analysis...')
        # dimension reduction
        logging.info(f"Number of features before PCA: {self.X.shape[1]}")

        pca = PCA(n_components=n_components)

        self.X = pd.DataFrame(
            pca.fit_transform(self.X),
            columns=["PCA" + str(n) for n in range(1, n_components + 1)]
        )
        self.X_test = pd.DataFrame(
            pca.transform(self.X_test),
            columns=["PCA" + str(n) for n in range(1, n_components + 1)]
        )

        logging.info(f"Number of features after PCA: {self.X.shape[1]}")
        logging.info(f'#{self._step_index} - DONE!')

    def _save_data(self, prefix: str = None) -> None:
        """
        Private method for saving the preprocessed data to csv files.

        :param prefix: prefix for the file name is necessary
        """

        logging.info(f'#{self._index()} - Saving processed data...')
        prefix = 'pp_' + prefix if prefix else 'pp'
        self.X.to_csv(f'data\\{prefix}_train.csv', index=False)
        self.X_test.to_csv(f'data\\{prefix}_test.csv', index=False)

        self.already_preprocessed = True
        logging.info(f'#{self._step_index} - DONE!')
