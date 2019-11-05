#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet_package.neuralnet import NeuralNetClassifier


"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
nn_clf = NeuralNetClassifier(
                        n_hidden_neurons = 50,
                        learning_rate = 0.00001,
                        max_iter = 1000,
                        tol = 1e-4,
                        verbose = True
                        )

X, y = nn_clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = nn_clf.train_test_split(X, y, test_size=0.3, random_state=4)
nn_clf.initialize_weights(X_train, y_train)
loss_array = nn_clf.fit(X_train, y_train)
y_predict = nn_clf.predict(X_train)

nn_clf.accuracy(y_train)

plt.style.use("seaborn-talk")
plt.plot(loss_array)
plt.xlabel("Iterations", size=15)
plt.ylabel("Loss from Cost function", size=15)
plt.show()

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
