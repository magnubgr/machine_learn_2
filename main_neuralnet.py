#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet_package.neuralnet import NeuralNet


"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
clf = NeuralNet()
X, y = clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = clf.train_test_split(X, y, test_size=0.3, random_state=4)

# coef, loss_arr = clf.fit_data(X_train, y_train, learning_rate=0.1, n_iter=400)
# pred = clf.predict(X_test)
# accuracy = clf.accuracy(pred, y_test.flatten())
# print(f"accuracy_score: {accuracy}")

# plt.style.use("seaborn-talk")
# plt.plot(loss_arr)
# plt.xlabel("Number of iterations", size=15)
# plt.ylabel("Cost function", size=15)
# plt.legend([r"Cost function $C(\beta)$"])
# plt.show()

