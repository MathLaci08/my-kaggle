import abc
import pandas as pd
import numpy as np
import tensorflow as tf

from typing import Union
from i_model import IModel


class TitanicModel(IModel, abc.ABC):
    @abc.abstractmethod
    def predict(self, x: pd.DataFrame, x_test: pd.DataFrame, cv: bool) -> None:
        """
        Class method for creating predictions.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """
        pass