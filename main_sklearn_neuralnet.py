import numpy as np
import matplotlib.pyplot as plt
from Classifier_package.classifier import Classifier
from NeuralNet_package.neural_v2 import NeuralNetRegression
from additional_files.Franke_function import FrankeFunction
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics


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
    tif_file = "veggli_terrain"
    n = 500
    x = np.linspace(0,1,n) ; np.random.shuffle(x)
    y = np.linspace(0,1,n) ; np.random.shuffle(y)
    reg = NeuralNetRegression(x,y)
    



regression()
