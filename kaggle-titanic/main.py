import pathlib
import logging
import pandas as pd
import numpy as np
import titanic_preprocessing

from logistic import LogisticReg
from svclassifier import SVClassifier
from neural_network import NeuralNetwork

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


models = ['logistic', 'logistic_cv', 'svc', 'svc_cv', 'NN', 'NN_cv']
submissions_dir = pathlib.Path(__file__, "..\\submissions").resolve()
logging.basicConfig(level=logging.INFO, format='*** %(levelname)s *** %(message)s')

if not submissions_dir.exists():
    submissions_dir.mkdir()

pp = titanic_preprocessing.TitanicPreProcessing()
pp.load_data()

LogisticReg().predict(*pp.load_data(), cv=True)
LogisticReg().predict(*pp.load_data())
SVClassifier().predict(*pp.load_data(), cv=True)
SVClassifier().predict(*pp.load_data())
NeuralNetwork().predict(*pp.load_data(), cv=True)
NeuralNetwork().predict(*pp.load_data())

all_in_one = None

for model in models:
    submission_path = pathlib.Path(__file__, f"..\\submissions\\submission_{model}.csv").resolve()
    if all_in_one is None:
        all_in_one = pd.read_csv(submission_path)
    else:
        all_in_one.Survived = all_in_one.Survived.add(pd.read_csv(submission_path).Survived)

all_in_one.Survived = all_in_one.Survived / len(models)
np.around(all_in_one).astype(int).to_csv('submissions\\submission.csv', index=False)
