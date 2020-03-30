import pathlib
import logging
# import pandas as pd
# import numpy as np
import titanic_preprocessing

# from OLS import OLS
# from decision_tree import DecisionTree
# from random_forest import RandomForest
# from extreme_grad_boost import XGB
# from neural_network import NeuralNetwork
# from linear_ridge import LinearRidge
#
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


models = []
submissions_dir = pathlib.Path(__file__, "..\\submissions").resolve()
logging.basicConfig(level=logging.INFO, format='*** %(levelname)s *** %(message)s')

if not submissions_dir.exists():
    submissions_dir.mkdir()

pp = titanic_preprocessing.TitanicPreProcessing()
pp.load_data()

# NeuralNetwork().predict(*pp.load_data(), cv=True)
# NeuralNetwork().predict(*pp.load_data())
# XGB().predict(*pp.load_data(), cv=True)
# XGB().predict(*pp.load_data())
# LinearRidge().predict(*pp.load_data(), cv=True)
# LinearRidge().predict(*pp.load_data())
# RandomForest().predict(*pp.load_data(), cv=True)
# RandomForest().predict(*pp.load_data())
# DecisionTree().predict(*pp.load_data(), cv=True)
# DecisionTree().predict(*pp.load_data())
# OLS().predict(*pp.load_data(with_pca=True))
#
# all_in_one = None
#
# for model in models:
#     submission_path = pathlib.Path(__file__, f"..\\submissions\\submission_{model}.csv").resolve()
#     if all_in_one is None:
#         all_in_one = pd.read_csv(submission_path)
#     else:
#         all_in_one.SalePrice = all_in_one.SalePrice.add(pd.read_csv(submission_path).SalePrice)
#
#
# all_in_one.SalePrice = all_in_one.SalePrice / len(models)
# np.around(all_in_one).astype(int).to_csv('submissions\\submission.csv', index=False)
