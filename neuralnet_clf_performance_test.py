#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet_package.neuralnet import NeuralNetClassifier
import scikitplot as skplt

import time
import sklearn.neural_network
import sklearn.metrics


"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
nn_clf = NeuralNetClassifier(
                    n_hidden_neurons = 80,
                    L2_penalty = 0.0001,
                    learning_rate = 0.00001,
                    max_iter = 1000,
                    tol = 1e-5,
                    verbose = True
                    )


X, y = nn_clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = nn_clf.train_test_split(X, y, test_size=0.33, random_state=4)
nn_clf.initialize_weights(X_train, n_output_neurons=1)
train_loss, test_loss, train_score, test_score = nn_clf.fit(X_train, y_train, X_test, y_test)

def plot_accuracy_loss(train_loss, test_loss, train_score, test_score):
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
    print(f"Accuracy Score for testing set: {test_score}")
    #print(f"cost for training set: {test_loss[-1]}")





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

    not_default = ax.lines[0]
    default = ax.lines[1]
    baseline = ax.lines[2]
    best = ax.lines[3]
    from scipy.integrate import simps   
    default_area     = simps(default.get_ydata(), default.get_xdata())
    best_area = simps(best.get_ydata(), best.get_xdata())
    baseline_area = simps(baseline.get_ydata(), baseline.get_xdata())
    print("LogReg Area ratio",(default_area-baseline_area)/(best_area-baseline_area) )





def nn_clf_sklearn():

    learning_rate = np.array([0.01, 0.001, 0.0001, 0.00001])
    n = len(learning_rate)
    timer = np.zeros(n)
    accuracy_score = np.zeros(n)


    for i in range(len(learning_rate)):
        time1 = time.time()
        print(int(100*i/len(learning_rate)), "%")
        reg = sklearn.neural_network.MLPClassifier(
                                hidden_layer_sizes = (40, 40, 40, 40),
                                learning_rate = "adaptive",
                                learning_rate_init = learning_rate[i],
                                max_iter = 1000,
                                tol = 1e-10,
                                verbose = False,
                                )
        reg = reg.fit(X_train, y_train.ravel())
        predict = reg.predict(X_test)
        accuracy_score[i] = reg.score(X_test,y_test.ravel())
        time2 = time.time()
        timer[i] =time2 -time1
        print("time = ",timer[i]," s")
    # print(np.shape(predict.reshape(1,-1)))
    # print(np.shape(y_test.reshape(1,-1)))
    # print(f" \n accuracy ={reg.score(X_test, y_test.ravel())}")
    plt.semilogx(learning_rate,accuracy_score, "*")
    plt.semilogx(learning_rate,accuracy_score)
    plt.xlabel(r"Learning rate $\eta$")
    plt.ylabel("Accuracy score")
    plt.title("Scikit-Learn NeuralNet score for different learning rates")
    plt.show()

    # print(timer)
    print(accuracy_score)


plot_accuracy_loss(train_loss, test_loss, train_score, test_score)
cumulative_gain()
nn_clf_sklearn()