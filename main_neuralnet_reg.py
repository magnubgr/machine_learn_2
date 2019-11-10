import numpy as np
import matplotlib.pyplot as plt
from Regression_package.LogisticRegressor import LogisticRegression
from Regression_package.Franke_function import FrankeFunction
from NeuralNet_package.neural_v2 import NeuralNetRegression
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics
import sys

def regression():
    # np.random.seed(12)
    n = 40
    x = np.linspace(0,1,n); np.random.shuffle(x)
    y = np.linspace(0,1,n); np.random.shuffle(y)
    X,Y = np.meshgrid(x,y)
    Z = FrankeFunction(X,Y)
    X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
    y_d = Z.ravel()[:, np.newaxis]
    y_d = (y_d-np.min(y_d))/(np.max(y_d)-np.min(y_d))
    print(np.max(y_d), np.min(y_d))


    nn_reg = NeuralNetRegression(
                        n_hidden_neurons=80,
                        L2_penalty=0.0001,
                        learning_rate=0.0001,
                        max_iter=2000,
                        tol=1e-6,
                        verbose=False)

    X_train, X_test, y_train, y_test = nn_reg.train_test_split(X_d, y_d, test_size=0.33, random_state=4)
    nn_reg.initialize_weights(X_train, y_train)
    train_loss, test_loss, train_score, test_score = nn_reg.fit(X_train, y_train, X_test, y_test)
    pred = nn_reg.predict(X_test)

    print(f"MSE = {nn_reg.MSE(y_test,pred)}")
    print(f"R2 ={nn_reg.R2(y_test,pred)}")

    plt.style.use("seaborn-talk")
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(["Training Loss","Testing Loss"])
    plt.xlabel("Iterations", size=15)
    plt.ylabel("Loss from MSE function", size=15)
    plt.show()

    plt.style.use("seaborn-talk")
    plt.plot(train_score[1:])
    plt.plot(test_score[1:])
    plt.legend(["Training R2 Score","Testing R2 Score"])
    plt.xlabel("Iterations", size=15)
    plt.ylabel("R2 Score", size=15)
    plt.show()

regression()



def regression_sklearn():
    """
    plots the test vs train dataset
    """
    n = 30
    x = np.linspace(0,1,n); np.random.shuffle(x)
    y = np.linspace(0,1,n); np.random.shuffle(y)
    X,Y = np.meshgrid(x,y)
    Z = FrankeFunction(X,Y)
    X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
    y_d = Z.ravel()[:, np.newaxis]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                                        X_d, y_d, test_size=0.33)
    m = 40
    n_iterations = np.linspace(330,360,m,dtype=int)
    R2_score_train = np.zeros(m)
    R2_score_test = np.zeros(m)
    for i in range(len(n_iterations)):
        reg = sklearn.neural_network.MLPRegressor(
                                hidden_layer_sizes = (30,30,30,30,30),
                                learning_rate = "adaptive",
                                learning_rate_init=0.01,
                                max_iter= n_iterations[i],
                                tol = 1e-11,
                                verbose = False,
                                )
        reg = reg.fit(X_train,y_train)
        pred = reg.predict(X_test)
        R2_score_train[i] = reg.score(X_train,y_train)
        R2_score_test[i] = reg.score(X_test,y_test)
        print(f"MSE = {sklearn.metrics.mean_squared_error(y_test,pred)}")
        print(f"R2 = {reg.score(X_test,y_test)}")
    plt.plot(n_iterations,R2_score_test, "-*")
    plt.plot(n_iterations,R2_score_train, "-*")
    plt.legend(["test", "train"])
    plt.xlabel("n_iterations")
    plt.ylabel("R2 score")
    plt.title("scikitlearn neural net for multiple n_iterations and for test and train")
    plt.show()

# regression_sklearn()
