import abc
import pandas as pd


class IModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def predict(self, x: pd.DataFrame, x_test: pd.DataFrame) -> None:
        pass
