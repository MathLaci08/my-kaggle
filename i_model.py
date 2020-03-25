import abc
import pandas as pd


class IModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        """
        Constructor of the model.
        """
        pass

    @abc.abstractmethod
    def predict(self, x: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """
        Class method for creating predictions.

        :param x: training data set
        :param x_test: test data set
        """
        pass
