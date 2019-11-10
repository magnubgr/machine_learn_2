#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet_package.neuralnet import NeuralNetClassifier
import scikitplot as skplt

"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
nn_clf = NeuralNetClassifier(
                        n_hidden_neurons = 80,
                        L2_penalty = 0.0001,
                        learning_rate = 0.00001,
                        max_iter = 500,
                        tol = 1e-5,
                        verbose = True
                        )


X, y = nn_clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = nn_clf.train_test_split(X, y, test_size=0.33, random_state=4)
nn_clf.initialize_weights(X_train, y_train)
train_loss, test_loss, train_score, test_score = nn_clf.fit(X_train, y_train, X_test, y_test)

def plot_accuracy_loss(train_score, test_score):
    plt.style.use("seaborn-talk")
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(["Training Loss","Testing Loss"])
    plt.xlabel("Iterations", size=15)
    plt.ylabel("Loss from Cost function", size=15)
    plt.savefig("plots/NN_classifier/NN_clf_training_loss.png")
    plt.show()

    plt.style.use("seaborn-talk")
    plt.plot(test_score)
    plt.plot(train_score)
    plt.legend(["Training Score","Testing Score"])
    plt.xlabel("Iterations", size=15)
    plt.ylabel("Accuracy Score", size=15)
    plt.savefig("plots/NN_classifier/NN_clf_training_score.png")
    plt.show()

    train_probs = nn_clf.predict(X_train)
    train_score = nn_clf.accuracy(y_train, train_probs)
    test_probs = nn_clf.predict(X_test)
    test_score = nn_clf.accuracy(y_test, test_probs)
    #print(f"Accuracy Score for training set: {train_score}")
    print(f"Accuracy Score for testing set:  {test_score}")
    #print(f"cost for training set: {test_loss[-1]}")

plot_accuracy_loss(train_score, test_score)

def cumulative_gain():
    def bestCurve(y):
        defaults = sum(y == 1)
        total = len(y)
        x = np.linspace(0, 1, total)
        y1 = np.linspace(0, 1, defaults)
        y2 = np.ones(total-defaults)
        y3 = np.concatenate([y1,y2])
        return x, y3

    x, y3 = bestCurve(y_test)

    default = nn_clf.feed_forward(X_test)[1]
    not_default = 1-default
    x = np.linspace(0, 1, len(default))
    y_probas = np.zeros((len(default), 2))
    y_probas[:,0] = not_default.ravel()
    y_probas[:,1] = default.ravel()

    y_actual_probas = np.zeros((len(y_test), 2))
    y_actual_probas[:,0] = y_test.ravel()==0
    y_actual_probas[:,1] = y_test.ravel()==1

    fig, ax = plt.subplots()
    skplt.metrics.plot_cumulative_gain(y_test,y_probas, ax=ax)
    ax.plot(x, y3)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.xlabel("Gain", size=13)
    plt.ylabel("Percentage of sample", size=13)
    ax.legend(["Not Default","Default","Baseline","Best"])
    plt.savefig("plots/NN_classifier/NN_cumulative_gain.png")
    plt.show()

# cumulative_gain()
