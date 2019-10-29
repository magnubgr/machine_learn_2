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

coef, loss_arr = clf.fit_data(X_train, y_train, learning_rate=0.1, n_iter=400)
pred = clf.predict(X_test)
accuracy = clf.accuracy(pred, y_test.flatten())
print(f"accuracy_score: {accuracy}")

plt.style.use("seaborn-talk")
plt.plot(loss_arr)
plt.xlabel("Number of iterations", size=15)
plt.ylabel("Cost function", size=15)
plt.legend([r"Cost function $C(\beta)$"])
plt.show()

# #clf.display_data()
# n = 40
# learning_rate = np.linspace(0,5,n)
# accuracy_score = np.zeros_like(learning_rate)
# for i in range(len(learning_rate)):
#     if 100*i%n == 0:
#         print(int(100*i/n), "%")
#     clf.fit_data(X_train, y_train, learning_rate=learning_rate[i], n_iter=300)
#     pred = clf.predict(X_test)
#     accuracy = clf.accuracy(pred, y_test.flatten())
#     print(f"accuracy_score: {accuracy}")
#     accuracy_score[i] =accuracy

# plt.plot(learning_rate,accuracy_score)
# plt.title('accuracy score as a function of learning rate')
# plt.xlabel('learning rate')
# plt.ylabel('accuracy score')
# plt.show()