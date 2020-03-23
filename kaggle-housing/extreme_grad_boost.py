import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

random_state = 237


def predict(x, x_test):
    # cut off target variable and id from the data
    y = pd.DataFrame(x['SalePrice'], columns=['SalePrice'])
    x.drop(['SalePrice'], axis=1, inplace=True)

    test_ids = x_test.Id
    x_test.drop(['Id'], axis=1, inplace=True)

    # splitting the data into training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=random_state
    )

    model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=random_state)

    model.fit(
        x_train, y_train, eval_metric='mae', early_stopping_rounds=5, eval_set=[(x_train, y_train)], verbose=False
    )

    print("Mean abs error: ", mean_absolute_error(np.exp(y_valid.SalePrice.values), np.exp(model.predict(x_valid))))

    test_pred = pd.DataFrame({'Id': test_ids, 'SalePrice': np.exp(model.predict(x_test))})
    test_pred.to_csv('submission.csv', index=False)