#!/usr/bin/env python
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from Classifier_package.classifier import Classifier

"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
clf = Classifier()
X, y = clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = clf.train_test_split(X, y, test_size=0.3, random_state=4)
#clf.display_data()
def heat_map():
    # learning_rate = np.arange(0,1,0.5)
    learning_rate = np.linspace(0,1,10)
    n_iterations = np.arange(20,350,5)
    accuracy_score = np.zeros((len(learning_rate),len(n_iterations)))
    for i in range(len(learning_rate)):
        if 100*i%len(learning_rate) == 0:
            print(int(100*i/len(learning_rate)), "%")
        for j in range(len(n_iterations)):
            clf.fit_data(X_train, y_train,
            learning_rate=learning_rate[i], n_iter=n_iterations[j])
            pred = clf.predict(X_test)
            accuracy = clf.accuracy(pred, y_test.flatten())
            accuracy_score[i,j]=accuracy

    plt.plot(learning_rate,accuracy_score[:,10])#int(len(n_iterations)/2)])
    plt.title('accuracy score as a function of learning rate. # iterations = '+str(n_iterations[10]))
    plt.xlabel('learning rate')
    plt.ylabel('accuracy score')
    plt.show()

    heat_map = sb.heatmap(accuracy_score, xticklabels=learning_rate, yticklabels=n_iterations,  cmap="viridis")
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    plt.title(r"Heatmap of accuracy_score for different learning_rate and iterations", size=20)
    plt.xlabel(r"learning_rate $\gamma $ ", size=18)
    plt.ylabel(r"n_iterations ", size=18)
    plt.show()
heat_map()

def n_iterations():
    n_iterations = np.arange(100,400,2)
    learning_rate = 1.5
    accuracy_score = np.zeros(len(n_iterations))
    for j in range(len(n_iterations)):
        print(100*j/len(n_iterations),'%')
        clf.fit_data(X_train, y_train,
        learning_rate=learning_rate, n_iter=n_iterations[j])
        pred = clf.predict(X_test)
        accuracy = clf.accuracy(pred, y_test.flatten())
        accuracy_score[j]=accuracy



    plt.plot(n_iterations, accuracy_score)
    plt.title(r"The accuracy for different n_iterations at learning rate ="+str(learning_rate), size=20)
    plt.xlabel(r"n_iterations ", size=18)
    plt.ylabel(r"accuracy ", size=18)
    plt.show()


def learning_rate():
    learning_rate = np.linspace(0,2,100)
    n_iterations = 300
    accuracy_score = np.zeros(len(learning_rate))
    for i in range(len(n_iterations)):
        print(100*j/len(n_iterations),'%')
        clf.fit_data(X_train, y_train,
        learning_rate=learning_rate[i], n_iter=n_iterations)
        pred = clf.predict(X_test)
        accuracy_score[j]= clf.accuracy(pred, y_test.flatten())

    plt.plot(learning_rate, accuracy_score)
    plt.title(r"The accuracy for different learning rate at n_iterations ="+str(n_iterations), size=20)
    plt.xlabel(r"learning_rate $\gamma $", size=18)
    plt.ylabel(r"accuracy ", size=18)
    plt.show()

def printing_accuracy(learning_rate, n_iterations):
    clf.fit_data(X_train, y_train,
    learning_rate=learning_rate, n_iter=n_iterations)
    pred = clf.predict(X_test)
    accuracy = clf.accuracy(pred, y_test.flatten())
    print(f"accuracy_score: {accuracy} with learning rate"+ \
        "at {learning_rate} and # iterations {n_iterations}")

printing_accuracy(1.6,300)
# n_iterations()
