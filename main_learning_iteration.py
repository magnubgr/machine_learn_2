#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler #, OneHotEncoder
# from sklearn.model_selection import  train_test_split
# from sklearn.linear_model import LogisticRegression
# from functools import partial
# import sys
from Classifier_package.classifier import Classifier
# import pandas as pd
# import sklearn

"""
runs the classifier class with data from the data package
"""

xls_file = "default_credit_card_data.xls"
clf = Classifier()
X, y = clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = clf.train_test_split(X, y, test_size=0.3, random_state=4)
#clf.display_data()
n = 30
learning_rate = np.linspace(0,30,n)
n_iterations = np.arange(0,100)
# print ( len(n_iterations), len(learning_rate))
# exit()
accuracy_score = np.zeros((len(learning_rate),len(n_iterations)))
for i in range(n):
    if 100*i%n == 0:
        print(int(100*i/n), "%")
    for j in range(len(n_iterations)):
        clf.fit_data(X_train, y_train,
        learning_rate=learning_rate[i], n_iter=n_iterations[j])
        pred = clf.predict(X_test)
        accuracy = clf.accuracy(pred, y_test.flatten())
        accuracy_score[i,j] =accuracy

import seaborn as sb
plt.plot(learning_rate,accuracy_score[:,int(len(n_iterations)/2)])
print(int(len(n_iterations)/2))
plt.title('accuracy score as a function of learning rate')
plt.xlabel('learning rate')
plt.ylabel('accuracy score')
plt.show()

heat_map = sb.heatmap(accuracy_score, xticklabels=learning_rate, yticklabels=n_iterations,  cmap="viridis")
heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
plt.title(r"Heatmap of accuracy_score for different learning_rate and iterations", size=20)
plt.xlabel(r"learning_rate $\gamma $ ", size=18)
plt.ylabel(r"n_iterations ", size=18)
plt.show()
