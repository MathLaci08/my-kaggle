import OLS
import preprocessing
import decision_tree
import random_tree
import extreme_grad_boost

pp = preprocessing.PreProcessing()
X, X_test = pp.get_preprocessed_data()

OLS.predict(X, X_test)
decision_tree.predict(X, X_test)
random_tree.predict(X, X_test)
extreme_grad_boost.predict(X, X_test)
