import pathlib
import logging
import pandas as pd
import numpy as np
import mnist_preprocessing

from neural_network import NeuralNetwork
from knearest import KNearest
from logistic import LogisticReg

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


models = ['NN_cv', 'NN', 'knearest']
submissions_dir = pathlib.Path(__file__, "..\\submissions").resolve()
logging.basicConfig(level=logging.INFO, format='*** %(levelname)s *** %(message)s')

if not submissions_dir.exists():
    submissions_dir.mkdir()

pp = mnist_preprocessing.MnistPreProcessing()

NeuralNetwork().predict(*pp.load_data(), cv=True)
NeuralNetwork().predict(*pp.load_data())
KNearest().predict(*pp.load_data())
LogisticReg().predict(*pp.load_data())

all_in_one = None

for model in models:
    submission_path = pathlib.Path(__file__, f"..\\submissions\\submission_{model}.csv").resolve()
    if all_in_one is None:
        all_in_one = pd.read_csv(submission_path)
    else:
        all_in_one = all_in_one.join(pd.read_csv(submission_path).Label, rsuffix=f'_{model}')

all_in_one.Label = all_in_one[['Label', 'Label_NN', 'Label_knearest']].apply(
    lambda row: pd.Series([row.mode().iloc[0]], index=['Label']), axis=1
)
all_in_one.drop(['Label_NN', 'Label_knearest'], axis=1, inplace=True)
all_in_one.to_csv('submissions\\submission.csv', index=False)
