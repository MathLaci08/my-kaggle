import pandas as pd
import tensorflow as tf
import logging
import pathlib

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from i_model import IModel

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense

random_state = 237
n_split = 5


class NeuralNetwork(IModel):
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
        # cut off target variable and id from the data
        y = pd.DataFrame(x['SalePrice'], columns=['SalePrice'])
        x.drop(['SalePrice'], axis=1, inplace=True)

        test_ids = x_test.Id
        x_test.drop(['Id'], axis=1, inplace=True)

        # standardize target variable
        mu = y.mean()
        sigma = y.std()

        y = (y - mu) / sigma

        # create own scoring function
        def dse_mae(y_true, y_pred):
            return tf.keras.losses.mae(
                self.inverse_transform(y_true, mu, sigma), self.inverse_transform(y_pred, mu, sigma)
            )

        model = self.create_model(x.shape[1], scoring=dse_mae)

        # fit the model
        if cv:
            checkpoint_name = 'checkpoints\\NN_model_cv.ckpt'
            try:
                # load saved weight
                model.load_weights(checkpoint_name)
            except OSError:
                pathlib.Path.mkdir(pathlib.Path('checkpoints').resolve(), exist_ok=True)
                index = 0
                log_messages = ['Improvement in validation loss:\n']

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
                    log_messages.append(f'Model evaluation {index} of {n_split}: {model.evaluate(x_valid, y_valid)[1]}')
                logging.info('\n'.join(log_messages))
        else:
            checkpoint_name = 'checkpoints\\NN_model.ckpt'
            try:
                # load saved weight
                model.load_weights(checkpoint_name)
            except OSError:
                # create checkpoints
                checkpoint = ModelCheckpoint(
                    checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'
                )
                model.fit(x, y, validation_split=0.1, epochs=500, batch_size=32, callbacks=[checkpoint])

        y_valid = self.inverse_transform(pd.DataFrame(y.SalePrice.values, columns=['SalePrice']), mu, sigma)
        pred_valid = self.inverse_transform(
            pd.DataFrame(model.predict(x.to_numpy()), columns=['SalePrice']), mu, sigma
        )

        nn_mae = "NN full data MAE with cross validation" if cv else "NN full data MAE"
        logging.info(f"{nn_mae}: {mean_absolute_error(y_valid, pred_valid)}")

        predictions = self.inverse_transform(
            pd.DataFrame(model.predict(x_test.to_numpy()), columns=['SalePrice']), mu, sigma
        )

        test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
        test_pred.to_csv(f'submissions\\submission_{"NN_cv" if cv else "NN"}.csv', index=False)

        logging.info("DONE!")

    @staticmethod
    def create_model(input_dim, scoring):
        model = Sequential()

        # the input layer
        model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))

        # the hidden layers
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))

        # the output layer
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[scoring])

        return model
