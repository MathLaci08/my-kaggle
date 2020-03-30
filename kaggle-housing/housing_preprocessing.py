import pandas as pd
import numpy as np
from i_preprocessing import IPreProcessing

from category_encoders import OneHotEncoder, OrdinalEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from scipy.stats import norm

from math import sqrt, ceil
import logging


class HousingPreProcessing(IPreProcessing):
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
        test_ids = self._separate_target()
        numerical_vars = self._detect_outliers()
        self._normalize_target()
        self._imputer()
        self._correlation_map()
        self._encode_categories()
        self._transform_skewed_features(numerical_vars)
        self._standardize_data()

        self.X = self.X.join(self.y)
        self.X_test = test_ids.to_frame().join(self.X_test)

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
        self.X.dropna(axis=0, subset=['SalePrice'], inplace=True)

        # set labels to variable y
        self.y = pd.DataFrame(self.X.SalePrice, columns=['SalePrice'])
        self.X.drop(['Id', 'Utilities', 'SalePrice'], axis=1, inplace=True)
        test_ids = self.X_test.Id
        self.X_test.drop(['Id', 'Utilities'], axis=1, inplace=True)

        # make MSSubClass categorical
        self.X['MSSubClass'] = self.X['MSSubClass'].apply(str)
        self.X_test['MSSubClass'] = self.X_test['MSSubClass'].apply(str)
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
        logging.info(f'#{self._step_index} - DONE!')

        # deleting outliers
        logging.info(f'#{self._index()} - Deleting outliers...')
        condition1 = (self.X[nv_for_detection[5]] > 4000) & (self.y.SalePrice < 500000)
        condition2 = (self.X[nv_for_detection[0]] > 150000)
        outliers = self.X[condition1].index
        self.X = self.X.drop(outliers).reset_index(drop=True)
        self.y = self.y.drop(outliers).reset_index(drop=True)

        fig, axes = plt.subplots(rows, cols)
        fig.tight_layout(pad=2.0)

        for feature in nv_for_detection:
            loc = nv_for_detection.get_loc(feature)
            axes[loc // cols][loc % cols].scatter(x=self.X[feature], y=self.y, s=5)
            axes[loc // cols][loc % cols].set(xlabel=feature, ylabel='SalePrice')
        plt.show()
        logging.info(f'#{self._step_index} - DONE!')

        return numerical_vars

    def _normalize_target(self):
        """
        This private function checks the distribution of the target variable and then transforms it with a logarithmic
        transformation (as the variable is right-skewed). Finally plots the distribution again.
        """

        logging.info(f'#{self._index()} - Normalizing target variable...')
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
        logging.info(f'#{self._step_index} - DONE!')

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
            self.X_test[col] = self.X_test[col].fillna(self.X[col].mode()[0])

        self.X['Functional'] = self.X['Functional'].fillna('Typ')
        self.X_test['Functional'] = self.X_test['Functional'].fillna('Typ')

        missing_values = self.X.isnull().sum().sum() + self.X_test.isnull().sum().sum()

        logging.info(f'Number of missing values after imputing: {missing_values}')
        logging.info(f'#{self._step_index} - DONE!')

    def _encode_categories(self):
        """
        This private method stands for encoding categorical variables. Label encoding used for ordinal categories and
        one-hot encoding used for nominal categories.
        """

        logging.info(f'#{self._index()} - Encoding categorical columns...')
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
        logging.info(f'#{self._step_index} - DONE!')
