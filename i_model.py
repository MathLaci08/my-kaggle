import abc
import pandas as pd
import numpy as np


class IModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        """
        Constructor of the model.
        """
        pass

    @abc.abstractmethod
    def predict(self, x: pd.DataFrame, x_test: pd.DataFrame, cv: bool) -> None:
        """
        Class method for creating predictions.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """
        pass

    @staticmethod
    def inverse_transform(data_frame: pd.DataFrame, mu: float, sigma: float) -> np.array:
        """
        This method transforms the data back to its original scale: de-standardize, get its exponential and finally
        rounds to the nearest 100 decimal.

        :param data_frame: the prediction of the model
        :param mu: mean of the original target variable
        :param sigma: standard deviation of the original variable
        :return: the transformed data frame
        """

        data_frame = data_frame * sigma + mu
        data_frame = np.exp(data_frame['SalePrice'].values)
        data_frame = np.around(data_frame).astype(int)

        return data_frame
