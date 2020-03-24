import pandas as pd
import numpy as np
from i_model import IModel

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense

random_state = 237


class NeuralNetwork(IModel):
    def __init__(self):
        super().__init__()

    def predict(self, x, x_test):
        # cut off target variable and id from the data
        y = pd.DataFrame(x['SalePrice'], columns=['SalePrice'])
        x.drop(['SalePrice'], axis=1, inplace=True)

        test_ids = x_test.Id
        x_test.drop(['Id'], axis=1, inplace=True)

        # standardize target variable
        mu = y.mean()
        sigma = y.std()

        y = (y - mu) / sigma

        model = Sequential()

        # the input layer
        model.add(Dense(128, kernel_initializer='normal', input_dim=x.shape[1], activation='relu'))

        # the hidden layers
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))

        # the output layer
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        # compile the network
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        print(model.summary())

        # create checkpoints
        checkpoint_name = 'Best-model.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]

        # fit the model
        model.fit(x, y, epochs=500, batch_size=32, validation_split=0.2, callbacks=callbacks_list)

        # load the best weight
        model.load_weights(checkpoint_name)
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

        predictions = pd.DataFrame(model.predict(x_test.to_numpy()), columns=['SalePrice'])
        predictions = predictions * sigma + mu

        test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': np.exp(predictions.SalePrice)})
        test_pred.to_csv('submission_NN.csv', index=False)
