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
    # print(np.max(y_d), np.min(y_d))


    nn_reg = NeuralNetRegression(
                        n_hidden_neurons=50,
                        L2_penalty=0.0001,
                        learning_rate=0.001,
                        max_iter=1000,
                        tol=1e-5,
                        verbose=False)

    X_train, X_test, y_train, y_test = nn_reg.train_test_split(X_d, y_d, test_size=0.3, random_state=4)
    nn_reg.initialize_weights(X_train, y_train)
    train_loss, test_loss, train_score, test_score = nn_reg.fit(X_train, y_train, X_test, y_test)
    # pred = nn_reg.predict(X_test)

    # print(f"MSE = {nn_reg.MSE(y_test,pred)}")
    # print(f"R2 ={nn_reg.R2(y_test,pred)}")

regression()