import numpy as np
import matplotlib.pyplot as plt
from Classifier_package.classifier import Classifier
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics



xls_file = "default_credit_card_data.xls"
clf = Classifier()
X, y = clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = clf.train_test_split(X, y,
                                   test_size=0.3, random_state=4)

reg = sklearn.neural_network.MLPRegressor(
hidden_layer_sizes = (10,10),
learning_rate = "adaptive",
learning_rate_init = 0.001,
max_iter = 100000,
tol = 1e-10,
verbose = True,
)

reg = reg.fit(X_train, y_train)

predict = reg.fit(X_train,y_train)

print(f" \n accuracy ={reg.score(X_test,y_test)}")
