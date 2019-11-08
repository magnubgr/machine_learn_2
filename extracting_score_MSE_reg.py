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

np.random.seed(2019)
def regression_package():
    # np.random.seed(12)
    n = 50
    x = np.linspace(0,1,n); np.random.shuffle(x)
    y = np.linspace(0,1,n); np.random.shuffle(y)
    X,Y = np.meshgrid(x,y)
    Z = FrankeFunction(X,Y)
    X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
    y_d = Z.ravel()[:, np.newaxis]
    y_d = (y_d-np.min(y_d))/(np.max(y_d)-np.min(y_d))
    print(np.max(y_d), np.min(y_d))


    nn_reg = NeuralNetRegression(
                        n_hidden_neurons=120,
                        L2_penalty=0.0001,
                        learning_rate=0.00000001,
                        max_iter=1000000,
                        tol=1e-20,
                        verbose=True)

    X_train, X_test, y_train, y_test = nn_reg.train_test_split(X_d, y_d, test_size=0.33, random_state=4)
    nn_reg.initialize_weights(X_train, y_train)
    train_loss, test_loss, train_score, test_score = nn_reg.fit(X_train, y_train, X_test, y_test)
    pred = nn_reg.predict(X_test)

    print(f"MSE = {nn_reg.MSE(y_test,pred)}")
    print(f"R2 ={nn_reg.R2(y_test,pred)}")

    plt.style.use("seaborn-talk")
    plt.plot(train_loss[1:-1:60])
    plt.plot(test_loss[1:-1:60])
    plt.legend(["Training Loss","Testing Loss"])
    plt.xlabel("Iterations", size=15)
    plt.ylabel("Loss from MSE function", size=15)
    plt.show()

    plt.style.use("seaborn-talk")
    plt.plot(train_score[1:-1:60])
    plt.plot(test_score[1:-1:60])
    plt.legend(["Training R2 Score","Testing R2 Score"])
    plt.xlabel("Iterations", size=15)
    plt.ylabel("R2 Score", size=15)
    plt.show()

# regression_package()


def regression_sklearn():
    # np.random.seed(12)
    n = 50
    x = np.linspace(0,1,n); np.random.shuffle(x)
    y = np.linspace(0,1,n); np.random.shuffle(y)
    X,Y = np.meshgrid(x,y)
    Z = FrankeFunction(X,Y)
    X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
    y_d = Z.ravel()[:, np.newaxis]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_d, y_d, test_size=0.33)

    m = 1
    features = np.linspace(120,120,m,dtype=float)
    R2_score = np.zeros(m)
    for i in range(len(features)):
        reg = sklearn.neural_network.MLPRegressor(
                                hidden_layer_sizes = int(features[i]),
                                learning_rate = "adaptive",
                                learning_rate_init=0.00001,
                                max_iter= 10000,
                                tol = 1e-11,
                                verbose = False,
                                )
        reg = reg.fit(X_train,y_train)
        pred = reg.predict(X_test)
        R2_score[i] = reg.score(X_test,y_test)

        print(f"MSE = {sklearn.metrics.mean_squared_error(y_test,pred)}")
        print(f"R2 = {reg.score(X_test,y_test)}")
    plt.plot(features,R2_score, "*")
    plt.plot(features,R2_score)
    plt.xlabel("features")
    plt.ylabel("R2 score")
    plt.title("scikitlearn neural net for multiple features")
    plt.show()


regression_sklearn()
