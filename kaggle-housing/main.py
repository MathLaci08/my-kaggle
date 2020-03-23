import OLS
import preprocessing
import decision_tree
import random_tree
import extreme_grad_boost
import neural_network

pp = preprocessing.PreProcessing(force=True)
X, X_test = pp.get_preprocessed_data()

# OLS.predict(X, X_test)
# decision_tree.predict(X, X_test)
# random_tree.predict(X, X_test)
# extreme_grad_boost.predict(X, X_test)
neural_network.predict(X, X_test)
