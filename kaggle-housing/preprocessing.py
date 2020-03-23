import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from category_encoders import OneHotEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew

import pathlib
from math import sqrt, ceil


class PreProcessing:
    """
    Class for data preprocessing related methods.
    Many ideas below came from: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook
    """
    # this should go to a config file with all the other (future) hyper-parameters
    n_components = 150

    def __init__(self, force=False):
        """
        Class instance initializer method.

        :param force: Boolean parameter. If True, previously created preprocessed data ignored, and steps will be
        re-computed. Defaults to False.
        """
        self.already_preprocessed = False
        try:
            if force:
                raise RuntimeWarning
            pp_train_csv = pathlib.Path(__file__, "..\\data\\pp_train.csv").resolve()
            pp_test_csv = pathlib.Path(__file__, "..\\data\\pp_test.csv").resolve()

            self.pp_X = pd.read_csv(pp_train_csv)
            self.pp_X_test = pd.read_csv(pp_test_csv)

            self.already_preprocessed = True
        except (FileNotFoundError, RuntimeWarning):
            self.pp_X = None
            self.pp_X_test = None
            train_csv = pathlib.Path(__file__, "..\\data\\train.csv").resolve()
            test_csv = pathlib.Path(__file__, "..\\data\\test.csv").resolve()

            # read data from the provided csv files
            self.X = pd.read_csv(train_csv)
            self.y = None
            self.X_test = pd.read_csv(test_csv)

    def get_preprocessed_data(self):
        """
        Method for determining the preprocessed data. If the data set haven't been preprocessed before, or forced to be
        ignored, the method calls all the necessary functions for the pre-processing.

        :return: The preprocessed data.
        """
        if self.already_preprocessed:
            return self.pp_X, self.pp_X_test
        else:
            test_ids = self.__separate_target()
            numerical_vars = self.__detect_outliers()
            self.__normalize_target()
            self.__imputer()
            self.__correlation_map()
            self.__one_hot_encode()
            self.__transform_skewed_features(numerical_vars)
            self.__standardize_data()
            self.__pca()

            self.X = self.X.join(self.y)
            self.X_test = test_ids.to_frame().join(self.X_test)

            self.__save_data()

            return self.X, self.X_test

    def __separate_target(self):
        """
        Private function for some preliminary steps. Drops non-labelled data, separates y from X and the test
        identifiers from the test set. Also converts the 'MSSubClass' feature to 'object' type (because it is
        categorical).

        :return: The test identifiers for future usage.
        """
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

        return test_ids

    def __detect_outliers(self):
        """
        Private function for detecting the outliers in the data set. First determines those numerical variables which
        have much unique values, and then plots the target variable as the function of these features. Base on the
        graphs it drops the outliers from the data, resets indices for X and y and finally plots the functions again.

        :return: Set of all numerical feature names.
        """
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

        # deleting outliers
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

        return numerical_vars

    def __normalize_target(self):
        """
        This private function checks the distribution of the target variable and then transforms it with a logarithmic
        transformation (as the variable is right-skewed). Finally plots the distribution again.
        """
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

    def __imputer(self):
        """
        Private function for dealing with missing values in the data sets. The method first creates lists of the
        feature names based on how to impute data in them, then fills the columns with appropriate values.
        """
        # deal with missing values
        missing_values = self.X.isnull().sum().sum() + self.X_test.isnull().sum().sum()
        print(f'Number of missing values before imputing: {missing_values}')

        fill_with_none = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType', 'BsmtQual',
                          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish',
                          'GarageQual', 'GarageCond']

        fill_with_zero = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                          'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                          'BsmtHalfBath', 'Fireplaces', 'GarageCars', 'GarageArea', 'GarageYrBlt', 'WoodDeckSF',
                          'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

        fill_with_most_frequent = ['Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'MSZoning', 'SaleType']

        for col in fill_with_none:
            self.X[col] = self.X[col].fillna('None')
            self.X_test[col] = self.X_test[col].fillna('None')

        for col in fill_with_zero:
            self.X[col] = self.X[col].fillna(0)
            self.X_test[col] = self.X_test[col].fillna(0)

        for col in fill_with_most_frequent:
            self.X[col] = self.X[col].fillna(self.X[col].mode()[0])
            self.X_test[col] = self.X_test[col].fillna(self.X_test[col].mode()[0])

        self.X['Functional'] = self.X['Functional'].fillna('Typ')
        self.X_test['Functional'] = self.X_test['Functional'].fillna('Typ')

        missing_values = self.X.isnull().sum().sum() + self.X_test.isnull().sum().sum()

        print(f'Number of missing values after imputing: {missing_values}')

    def __correlation_map(self):
        """
        Private function for plotting the correlation map between the features.
        """
        # correlation map between the remaining features
        corr_map = self.X.join(self.y).corr()
        plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_map, vmax=0.9, square=True)
        plt.show()

    def __one_hot_encode(self):
        """
        This private method stands for one-hot encoding categorical variables.
        """
        # get column names for categorical and numerical features
        categorical_vars = self.X.select_dtypes(include='object').columns
        numerical_vars = self.X.columns.difference(categorical_vars)

        x_num = self.X[numerical_vars]
        x_test_num = self.X_test[numerical_vars]

        # one hot encode categorical columns
        one_hot_encoder = OneHotEncoder(use_cat_names=True)

        x_cat = one_hot_encoder.fit_transform(self.X[categorical_vars])
        x_test_cat = one_hot_encoder.transform(self.X_test[categorical_vars])

        self.X = x_num.join(x_cat)
        self.X_test = x_test_num.join(x_test_cat)

    def __transform_skewed_features(self, numerical_vars):
        """
        Private method for transforming features with high skew.

        :param numerical_vars: Set of all original numerical variables.
        """
        # check the skew of all numerical features
        skewed_features = self.X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("Skew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_features})
        print(skewness)

        # transform skewed features
        skewed_features = skewness[abs(skewness.Skew) > 1].index
        print(f"There are {skewed_features.size} skewed features")

        for feature in skewed_features:
            self.X[feature] = np.log1p(self.X[feature])
            self.X_test[feature] = np.log1p(self.X_test[feature])  # box-cox transformation instead?

        # check the skew of all numerical features again
        skewed_features = self.X[numerical_vars].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("Skew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_features})
        print(skewness)

    def __standardize_data(self):
        """
        This private function's job is the standardization of all the variables.
        """
        # standardize data
        std_scaler = StandardScaler(copy=False)

        self.X = pd.DataFrame(std_scaler.fit_transform(self.X), columns=self.X.columns)
        self.X_test = pd.DataFrame(std_scaler.transform(self.X_test), columns=self.X.columns)

    def __pca(self):
        """
        This private function do the principal component analysis on our data, and as a result, dimension reduction
        will be made.
        """
        # dimension reduction
        print(f"Number of features before PCA: {self.X.shape[1]}")

        pca = PCA(n_components=self.n_components)

        self.X = pd.DataFrame(
            pca.fit_transform(self.X),
            columns=["PCA" + str(n) for n in range(1, self.n_components + 1)]
        )
        self.X_test = pd.DataFrame(
            pca.transform(self.X_test),
            columns=["PCA" + str(n) for n in range(1, self.n_components + 1)]
        )

        print(f"Number of features after PCA: {self.X.shape[1]}")

    def __save_data(self):
        """
        Private method for saving the preprocessed data to csv files.
        """
        self.X.to_csv('data\\pp_train.csv', index=False)
        self.X_test.to_csv('data\\pp_test.csv', index=False)
