import OLS
import preprocessing

pp = preprocessing.PreProcessing()
X, X_test = pp.get_preprocessed_data()

OLS.predict(X, X_test)
