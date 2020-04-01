import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import pathlib

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from titanic_model import TitanicModel

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense

random_state = 237
n_split = 5


class NeuralNetwork(TitanicModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test, cv=False):
        """
        IModel's predict functions NN implementation.

        :param x: training data set
        :param x_test: test data set
        :param cv: True if prediction should be made with cross-validation
        """

        logging.info("Performing NN prediction...")
        # cut off target variable and PassengerId from the data
        y = x['Survived'].apply(lambda t: pd.Series([1-t, t], index=['0', '1']))
        x.drop(['Survived'], axis=1, inplace=True)

        test_ids = x_test.PassengerId
        x_test.drop(['PassengerId'], axis=1, inplace=True)

        model = self.create_model(x.shape[1])

        # fit the model
        if cv:
            checkpoint_name = 'checkpoints\\NN_model_cv.ckpt'
            try:
                # load saved weight
                model.load_weights(checkpoint_name)
            except OSError:
                pathlib.Path.mkdir(pathlib.Path('checkpoints').resolve(), exist_ok=True)
                index = 0
                log_messages = ['Improvement in accuracy:\n']

                # create checkpoints
                checkpoint = ModelCheckpoint(
                    checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'
                )
                for train_index, valid_index in KFold(n_split).split(x):
                    x_train, x_valid = x.iloc[train_index, :], x.iloc[valid_index, :]
                    y_train, y_valid = y.iloc[train_index, :], y.iloc[valid_index, :]
                    model.fit(
                        x_train, y_train, validation_data=(x_valid, y_valid),
                        epochs=100, batch_size=32, callbacks=[checkpoint]
                    )

                    index += 1
                    evalu = model.evaluate(x_valid, y_valid)
                    log_messages.append(f'Model accuracy {index} of {n_split}: {evalu[1] * 100:.2f}%')
                logging.info('\n'.join(log_messages))
        else:
            checkpoint_name = 'checkpoints\\NN_model.ckpt'
            try:
                # load saved weight
                model.load_weights(checkpoint_name)
            except OSError:
                pathlib.Path.mkdir(pathlib.Path('checkpoints').resolve(), exist_ok=True)
                # create checkpoints
                checkpoint = ModelCheckpoint(
                    checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'
                )
                model.fit(x, y, validation_split=0.1, epochs=500, batch_size=32, callbacks=[checkpoint])

        y_valid = pd.DataFrame(y['1'].values, columns=['Survived'])
        pv_0 = model.predict(x.to_numpy())[:, 0]
        pred_valid = pd.DataFrame(np.where(pv_0 > 0.5, 0, 1), columns=['Survived'])

        nn_acc = "NN full data accuracy with cross validation" if cv else "NN full data accuracy"
        logging.info(f"{nn_acc}: {accuracy_score(y_valid, pred_valid) * 100:.2f}%")

        pt_0 = model.predict(x_test.to_numpy())[:, 0]
        test_pred = pd.DataFrame({'PassengerId': test_ids, 'Survived': np.where(pt_0 > 0.5, 0, 1)})
        test_pred.to_csv(f'submissions\\submission_{"NN_cv" if cv else "NN"}.csv', index=False)

        logging.info("DONE!")

    @staticmethod
    def create_model(input_dim):
        model = Sequential()

        # the input layer
        model.add(Dense(16, kernel_initializer='normal', input_dim=input_dim, activation='sigmoid'))

        # the hidden layers
        model.add(Dense(32, kernel_initializer='normal', activation='sigmoid'))
        model.add(Dense(32, kernel_initializer='normal', activation='sigmoid'))
        model.add(Dense(32, kernel_initializer='normal', activation='sigmoid'))

        # the output layer
        model.add(Dense(2, kernel_initializer='normal', activation='softmax'))

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

        return model
