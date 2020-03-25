import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from category_encoders import OneHotEncoder, OrdinalEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

import pathlib
from math import sqrt, ceil
import logging


n_components = 100


class PreProcessing:
    """
    Class for data pre-processing related methods.
    Many ideas below came from: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook
    """

    def __init__(self):
        """
        Class instance initializer method.
        """

        try:
            self.pp_X = None
            self.pp_X_test = None
            train_csv = pathlib.Path(__file__, "..\\data\\train.csv").resolve()
            test_csv = pathlib.Path(__file__, "..\\data\\test.csv").resolve()

            # read data from the provided csv files
            self.X = pd.read_csv(train_csv)
            self.y = None
            self.X_test = pd.read_csv(test_csv)
        except FileNotFoundError as e:
            logging.error("Please download the data, before creating instance!")
            raise e

    def process_data(self):
        """
        Method for determining the preprocessed data. If the data set haven't been preprocessed before, or forced to be
        ignored, the method calls all the necessary functions for the pre-processing.
        """

        logging.info('Processing data...')
        test_ids = self.__separate_target()
        numerical_vars = self.__detect_outliers()
        self.__normalize_target()
        self.__imputer()
        self.__correlation_map()
        self.__encode_categories()
        self.__transform_skewed_features(numerical_vars)
        self.__standardize_data()

        x_pca = self.X
        x_test_pca = self.X_test

        self.X = self.X.join(self.y)
        self.X_test = test_ids.to_frame().join(self.X_test)

        self.__save_data()

        self.X = x_pca
        self.X_test = x_test_pca

        self.__pca()

        self.X = self.X.join(self.y)
        self.X_test = test_ids.to_frame().join(self.X_test)

        self.__save_data(prefix='pca')

    def load_data(self, with_pca=False):
        """
        Loads the previously processed data from the saved csv files.

        :param with_pca: if True, function will return a data set on which pca was performed before
        :return: train and test set if data is preprocessed, else None.
        """

        try:
            logging.info('Trying to load data...')
            prefix = 'pp_pca' if with_pca else 'pp'
            pp_train_csv = pathlib.Path(__file__, f"..\\data\\{prefix}_train.csv").resolve()
            pp_test_csv = pathlib.Path(__file__, f"..\\data\\{prefix}_test.csv").resolve()

            self.pp_X = pd.read_csv(pp_train_csv)
            self.pp_X_test = pd.read_csv(pp_test_csv)
            logging.info('DONE!')

            return self.pp_X, self.pp_X_test
        except FileNotFoundError:
            logging.warning("Data is not preprocessed. Calling process_data() function...")
            self.process_data()
            return self.load_data(with_pca=with_pca)

    def __separate_target(self):
        """
        Private function for some preliminary steps. Drops non-labelled data, separates y from X and the test
        identifiers from the test set. Also converts the 'MSSubClass' feature to 'object' type (because it is
        categorical).

        :return: The test identifiers for future usage.
        """


        logging.info('#0 - Dropping unnecessary features...')
        # drop rows without labels
        self.X.dropna(axis=0, subset=['SalePrice'], inplace=True)

        # set labels to variable y
        self.y = pd.DataFrame(self.X.SalePrice, columns=['SalePrice'])
        self.X.drop(['Id', 'Utilities', 'SalePrice'], axis=1, inplace=True)
        test_ids = self.X_test.Id
        self.X_test.drop(['Id', 'Utilities'], axis=1, inplace=True)

        # make MSSubClass categorical
        self.X['MSSubClass'] = self.X['MSSubClass'].apply(str)
        self.X_test['MSSubClass'] = self.X_test['MSSubClass'].apply(str)
        logging.info('#0 - DONE!')

        return test_ids

    def __detect_outliers(self):
        """
        Private function for detecting the outliers in the data set. First determines those numerical variables which
        have much unique values, and then plots the target variable as the function of these features. Base on the
        graphs it drops the outliers from the data, resets indices for X and y and finally plots the functions again.

        :return: Set of all numerical feature names.
        """

        logging.info('#1 - Detecting outliers...')
        # get numerical features
        numerical_vars = self.X.select_dtypes(exclude="object").columns
        nv_for_detection = pd.Index([f for f in numerical_vars if self.X[f].nunique() > 500])
        cols = ceil(sqrt(nv_for_detection.size))
        rows = ceil(nv_for_detection.size // cols)

        # detect outliers
        fig, axes = plt.subplots(rows, cols)
        fig.tight_layout(pad=2.0)

        for feature in nv_for_detection:
            loc = nv_for_detection.get_loc(feature)
            axes[loc // cols][loc % cols].scatter(x=self.X[feature], y=self.y, s=5)
            axes[loc // cols][loc % cols].set(xlabel=feature, ylabel='SalePrice')
        plt.show()
        logging.info('#1 - DONE!')

        # deleting outliers
        logging.info('#2 - Deleting outliers...')
        condition1 = (self.X[nv_for_detection[5]] > 4000) & (self.y.SalePrice < 500000)
        condition2 = (self.X[nv_for_detection[0]] > 150000)
        outliers = self.X[condition1 | condition2].index
        self.X = self.X.drop(outliers).reset_index(drop=True)
        self.y = self.y.drop(outliers).reset_index(drop=True)

        fig, axes = plt.subplots(rows, cols)
        fig.tight_layout(pad=2.0)

        for feature in nv_for_detection:
            loc = nv_for_detection.get_loc(feature)
            axes[loc // cols][loc % cols].scatter(x=self.X[feature], y=self.y, s=5)
            axes[loc // cols][loc % cols].set(xlabel=feature, ylabel='SalePrice')
        plt.show()
        logging.info('#2 - DONE!')

        return numerical_vars

    def __normalize_target(self):
        """
        This private function checks the distribution of the target variable and then transforms it with a logarithmic
        transformation (as the variable is right-skewed). Finally plots the distribution again.
        """

        logging.info('#3 - Normalizing target variable...')
        # check the distribution of SalePrice
        sns.distplot(self.y, fit=norm)

        (mu, sigma) = norm.fit(self.y)

        plt.legend([f'Normal distribution $\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')

        # Get also the QQ-plot
        plt.figure()
        stats.probplot(self.y.SalePrice, plot=plt)
        plt.show()

        # distribution is right skewed, so logarithmic transformation will be made
        self.y = np.log(self.y)

        # check the distribution of the transformed SalePrice
        sns.distplot(self.y, fit=norm)

        (mu, sigma) = norm.fit(self.y)

        plt.legend([f'Normal distribution $\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )'], loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')

        # Get also the QQ-plot
        plt.figure()
        stats.probplot(self.y.SalePrice, plot=plt)
        plt.show()
        logging.info('#3 - DONE!')

    def __imputer(self):
        """
        Private function for dealing with missing values in the data sets. The method first creates lists of the
        feature names based on how to impute data in them, then fills the columns with appropriate values.
        """

        logging.info('#4 - Imputing appropriate values in empty cells...')
        # deal with missing values
        missing_values = self.X.isnull().sum().sum() + self.X_test.isnull().sum().sum()
        logging.info(f'Number of missing values before imputing: {missing_values}')

        fill_with_none = ['Alley', 'Fence', 'MiscFeature', 'MasVnrType', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                          'GarageType', 'GarageFinish']

        fill_with_na = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                        'GarageQual', 'GarageCond', 'PoolQC']

        fill_with_zero = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                          'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                          'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'GarageArea', 'GarageYrBlt', 'WoodDeckSF',
                          'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

        fill_with_most_frequent = ['Electrical', 'Exterior1st', 'Exterior2nd', 'MSZoning', 'SaleType']

        for col in fill_with_none:
            self.X[col] = self.X[col].fillna('None')
            self.X_test[col] = self.X_test[col].fillna('None')

        for col in fill_with_na:
            self.X[col] = self.X[col].fillna('NA')
            self.X_test[col] = self.X_test[col].fillna('NA')

        for col in fill_with_zero:
            self.X[col] = self.X[col].fillna(0)
            self.X_test[col] = self.X_test[col].fillna(0)

        for col in fill_with_most_frequent:
            self.X[col] = self.X[col].fillna(self.X[col].mode()[0])
            self.X_test[col] = self.X_test[col].fillna(self.X_test[col].mode()[0])

        self.X['Functional'] = self.X['Functional'].fillna('Typ')
        self.X_test['Functional'] = self.X_test['Functional'].fillna('Typ')

        missing_values = self.X.isnull().sum().sum() + self.X_test.isnull().sum().sum()

        logging.info(f'Number of missing values after imputing: {missing_values}')
        logging.info('#4 - DONE!')

    def __correlation_map(self):
        """
        Private function for plotting the correlation map between the features.
        """

        logging.info('#5 - Checking correlation between features...')
        # correlation map between the remaining features
        corr_map = self.X.join(self.y).corr()
        plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_map, vmax=0.9, square=True)
        plt.show()
        logging.info('#5 - DONE!')

    def __encode_categories(self):
        """
        This private method stands for encoding categorical variables. Label encoding used for ordinal categories and
        one-hot encoding used for nominal categories.
        """

        logging.info('#6 - Encoding categorical columns...')
        # get column names for categorical and numerical features
        categorical_vars = self.X.select_dtypes(include='object').columns
        numerical_vars = self.X.columns.difference(categorical_vars)

        ordinal = pd.Index(['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                            'GarageQual', 'GarageCond', 'PoolQC'])
        nominal = categorical_vars.difference(ordinal)

        standard_mapping = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        mapping_for_ordinals = [{'col': column, 'mapping': standard_mapping} for column in ordinal]

        x_num = self.X[numerical_vars]
        x_test_num = self.X_test[numerical_vars]

        # one hot encode categorical columns
        one_hot_encoder = OneHotEncoder(use_cat_names=True)
        label_encoder = OrdinalEncoder(drop_invariant=True, mapping=mapping_for_ordinals, handle_unknown='error')

        x_cat_nom = one_hot_encoder.fit_transform(self.X[nominal])
        x_cat_ord = label_encoder.fit_transform(self.X[ordinal])
        x_test_cat_nom = one_hot_encoder.transform(self.X_test[nominal])
        x_test_cat_ord = label_encoder.transform(self.X_test[ordinal])

        self.X = x_num.join(x_cat_ord).join(x_cat_nom)
        self.X_test = x_test_num.join(x_test_cat_ord).join(x_test_cat_nom)
        logging.info('#6 - DONE!')

    def __transform_skewed_features(self, numerical_vars):
        """
        Private method for transforming features with high skew.

        :param numerical_vars: Set of all original numerical variables.
        """

        logging.info('#7 - Determine and transform skewed features...')
        # check the skew of all numerical features
        skewed_features = self.X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        logging.info("Skew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_features})
        logging.info(skewness)

        # transform skewed features
        skewed_features = skewness[abs(skewness.Skew) > 1].index
        logging.info(f"There are {skewed_features.size} skewed features")

        for feature in skewed_features:
            self.X[feature] = np.log1p(self.X[feature])
            self.X_test[feature] = np.log1p(self.X_test[feature])  # box-cox transformation instead?

        # check the skew of all numerical features again
        skewed_features = self.X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        logging.info("Skew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_features})
        logging.info(skewness)
        logging.info('#7 - DONE!')

    def __standardize_data(self):
        """
        This private function's job is the standardization of all the variables.
        """

        logging.info('#8 - Standardizing variables...')
        # standardize data
        std_scaler = StandardScaler(copy=False)

        self.X = pd.DataFrame(std_scaler.fit_transform(self.X), columns=self.X.columns)
        self.X_test = pd.DataFrame(std_scaler.transform(self.X_test), columns=self.X.columns)
        logging.info('#8 - DONE!')

    def __pca(self):
        """
        This private function do the principal component analysis on our data, and as a result, dimension reduction
        will be made.
        """

        logging.info('#9 - Performing principal component analysis...')
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
        logging.info('#9 - DONE!')

    def __save_data(self, prefix=None):
        """
        Private method for saving the preprocessed data to csv files.
        """

        logging.info('#10 - Saving processed data...')
        prefix = 'pp_' + prefix if prefix else 'pp'
        self.X.to_csv(f'data\\{prefix}_train.csv', index=False)
        self.X_test.to_csv(f'data\\{prefix}_test.csv', index=False)

        self.already_preprocessed = True
        logging.info('#10 - DONE!')
