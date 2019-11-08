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
                        L2_penalty = 0.0001,
                        learning_rate = 0.00001,
                        max_iter = 500,
                        tol = 1e-4,
                        verbose = True
                        )


X, y = nn_clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = nn_clf.train_test_split(X, y, test_size=0.3, random_state=4)
nn_clf.initialize_weights(X_train, y_train)
train_loss, test_loss, train_score, test_score = nn_clf.fit(X_train, y_train, X_test, y_test)


plt.style.use("seaborn-talk")
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(["Training Loss","Testing Loss"])
plt.xlabel("Iterations", size=15)
plt.ylabel("Loss from Cost function", size=15)
plt.savefig("plots/NN_classifier/NN_clf_training_loss.png")
plt.show()

plt.style.use("seaborn-talk")
plt.plot(train_score)
plt.plot(test_score)
plt.legend(["Training Score","Testing Score"])
plt.xlabel("Iterations", size=15)
plt.ylabel("Accuracy Score", size=15)
plt.savefig("plots/NN_classifier/NN_clf_training_score.png")
plt.show()


train_probs = nn_clf.predict(X_train)
train_score = nn_clf.accuracy(y_train, train_probs)
test_probs = nn_clf.predict(X_test)
test_score = nn_clf.accuracy(y_test, test_probs)
print(f"Accuracy Score for training set: {train_score}")
print(f"Accuracy Score for testing set:  {test_score}")