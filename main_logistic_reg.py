#!/usr/bin/env python
import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
from Regression_package.LogisticRegressor import LogisticRegression
"""
runs the classifier class with data from the data package
"""
xls_file = "default_credit_card_data.xls"
clf = LogisticRegression()
X, y = clf.read_credit_card_file(xls_file)
X_train, X_test, y_train, y_test = clf.train_test_split(X, y, test_size=0.3, random_state=4)

coef, loss_arr = clf.fit_data(X_train, y_train, learning_rate=1.6, n_iter=300)
pred = clf.predict(X_test)
accuracy = clf.accuracy(pred, y_test.flatten())
print(f"accuracy_score: {accuracy}")

plt.plot(loss_arr)
plt.xlabel("Number of iterations", size=15)
plt.ylabel("Cost function", size=15)
plt.legend([r"Cost function $C(\beta)$"])
plt.show()

print(y.shape)

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

    default = clf.prob(X_test, coef)
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
    plt.savefig("plots/Logistic_Regression/LogReg_cumulative_gain.png")
    plt.show()
    # fig, ax = plt.subplots()
    # skplt.metrics.plot_roc(y_test,y_probas, ax=ax, plot_micro=False, plot_macro=False)
    # skplt.metrics.plot_roc(y_test,y_actual_probas, ax=ax, plot_micro=False, plot_macro=False)
    # plt.axis([-0.05, 1.05, -0.05, 1.05])
    # plt.xlabel("False Positive Rate", size=13)
    # plt.ylabel("True Positive Rate", size=13)
    # ax.legend(["Not Default","Default","Best Not Default","Best Default"])
    # plt.savefig("plots/Logistic_Regression/LogReg_roc_curve.png")
    # plt.show()

# cumulative_gain()
