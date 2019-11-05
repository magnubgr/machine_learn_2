#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet_package.neuralnet import NeuralNet


"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
nn_clf = NeuralNet()
X, y = nn_clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = nn_clf.train_test_split(X, y, test_size=0.3, random_state=4)

nn_clf.initialize_weights(X_train, y_train, n_hidden_neurons=50)

# a_h, probs = nn_clf.feed_forward(X_train)
# print(probs)
# print(probs>0.5)

# accuracy = nn_clf.accuracy(y_train, (probs>0.5))
# print(f"accuracy: {accuracy}")


# output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient = nn_clf.backpropagation(X_train, y_train)

n_iterations = 1000
learning_rate = 0.00001
nn_clf.fit(X_train, y_train, n_iterations, learning_rate )

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
