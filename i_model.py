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
    def predict(self, x: pd.DataFrame, x_test: pd.DataFrame, cv: bool) -> None:
        """
        Class method for creating predictions.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """
        pass
