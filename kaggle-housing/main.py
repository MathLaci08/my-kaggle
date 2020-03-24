import pathlib
import pandas as pd
import preprocessing

from OLS import OLS
from decision_tree import DecisionTree
from random_forest import RandomForest
from extreme_grad_boost import XGB
from neural_network import NeuralNetwork


models = ['XGB', 'random', 'decision', 'OLS', 'NN']

pp = preprocessing.PreProcessing(force=True)
pp.process_data(perform_pca=True, n_components=100)

OLS().predict(*pp.load_data())
DecisionTree().predict(*pp.load_data())
RandomForest().predict(*pp.load_data())
XGB().predict(*pp.load_data())
NeuralNetwork().predict(*pp.load_data())

all_in_one = None

for model in models:
    submission_path = pathlib.Path(__file__, f"..\\submission_{model}.csv").resolve()
    if all_in_one is None:
        all_in_one = pd.read_csv(submission_path)
    else:
        all_in_one.SalePrice = all_in_one.SalePrice.add(pd.read_csv(submission_path).SalePrice)


all_in_one.SalePrice = all_in_one.SalePrice / len(models)
all_in_one.to_csv('submission.csv', index=False)
