import numpy as np
import matplotlib.pyplot as plt
from Classifier_package.classifier import Classifier
from NeuralNet_package.neural_v2 import NeuralNetRegression
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics
import sys


sys.path.insert(1, './data/')
import data/FrankeFunction


def classifier():
    xls_file = "default_credit_card_data.xls"
    clf = Classifier()
    X, y = clf.read_credit_card_file(xls_file)
    X_train, X_test, y_train, y_test = clf.train_test_split(X, y,
                                   test_size=0.3, random_state=4)
    n = 7
    learning_rate = 10**(-np.linspace(0,n-1,n))
    timer = np.zeros(n)
    accuracy_score = np.zeros(n)


    for i in range(len(learning_rate)):
        time1 = time.time()
        print(int(100*i/len(learning_rate)), "%")
        reg = sklearn.neural_network.MLPClassifier(
                                hidden_layer_sizes = (50,50,50),
                                learning_rate = "adaptive",
                                learning_rate_init = learning_rate[i],
                                max_iter = 1000,
                                tol = 1e-20,
                                verbose = False,
                                )
        reg = reg.fit(X_train, y_train.ravel())
        predict= reg.predict(X_test)
        accuracy_score[i] = reg.score(X_test,y_test.ravel())
        time2 = time.time()
        timer[i] =time2 -time1
        print("time = ",timer[i]," s")
    # print(np.shape(predict.reshape(1,-1)))
    # print(np.shape(y_test.reshape(1,-1)))
    # print(f" \n accuracy ={reg.score(X_test, y_test.ravel())}")
    plt.semilogx(learning_rate,accuracy_score, "*")
    plt.semilogx(learning_rate,accuracy_score)
    plt.xlabel("learning_rate")
    plt.ylabel("accuracy score")
    plt.title("scikitlearn neural net for multiple learning rates")
    plt.show()
    print(timer)
    print(accuracy_score)



def regression():
    np.random.seed(2019)
    tif_file = "veggli_terrain"
    n = 200
    x = np.linspace(0,1,n); np.random.shuffle(x)
    y = np.linspace(0,1,n); np.random.shuffle(y)
    X,Y = np.meshgrid(x,y)
    dat = NeuralNetRegression(x,y)
    Z = dat.Z()
    X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
    y_d = Z.ravel()[:, np.newaxis]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X_d, y_d, test_size=0.4)

    reg = sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes = (100, 20),
    learning_rate = "adaptive",
    learning_rate_init=0.01,
    max_iter= 10000,
    tol = 1e-10,
    verbose = True, )
    reg = reg.fit(X_train,y_train)
    pred = reg.predict(X_test)
    print(f"MS = {sklearn.metrics.mean_squared_error(y_test,pred)}")
    print(f"R2 ={reg.score(X_test,y_test)}")

regression()
