#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import  train_test_split
from functools import partial
import sys

"""
Describe the model and input argument
"""


class RegSolver:
    def __init__(self,x,y,y_data, degree = 5):
        self.degree = degree
        self.x = x
        self.y = y
        self.y_data = y_data
        self.n = len(x)
        self.getX()


    def getX(self):
        """
        Creates the design matrix X based on x and y data
        The degree can be adjusted from the init
        """
        xy_zip = np.array(list(zip(self.x,self.y)))
        poly = PolynomialFeatures(self.degree)      # using sklearn.preprocessing
        self.X_ = poly.fit_transform(xy_zip)
        return self.X_

    def X(self):
        return self.X_

    def var(self,l):
        """
        returns the variance of an 1-dimensional array
        """
        return np.sum((l-np.mean(l))**2)/len(l)

    def MSE(self, y_data=[], y_model=[]):
        """
        takes the MSE of two arrays, y_data being the data and y_model being
        the model fitted on the data
        """
        y_data = self.y_data if len(y_data) == 0 else y_data
        y_model = self.y_model if len(y_model)==0 else y_model
        # return np.mean( (self.y_model-self.y_data)**2 )
        return np.mean( (y_model-y_data)**2 )

    def R2(self, y_data=[], y_model=[]):
        """
        Calculates the R2 score of the function of the y_model
        comaped to the y_data. 1 is the best score and 0 is the worst.
        """
        y_data = self.y_data if len(y_data)==0 else y_data
        y_model = self.y_model if len(y_model)==0 else y_model
        return 1 - np.sum((y_data - y_model) ** 2) /\
         np.sum((y_data - np.mean(y_data)) ** 2)

    def confidence_interval(self,beta = [], filename=""):
        """
        Calculates the confidence intervals for the coefficients beta,
        and plots them in a figure.
        """
        beta_min = np.zeros_like(beta)
        beta_max = np.zeros_like(beta)
        xtx_inv = np.linalg.inv(self.X_.T.dot(self.X_))
        std_dev = np.sqrt( self.var(self.y_data) )
        for j in range(len(beta)):
            vj = xtx_inv[j,j]
            beta_min[j] = beta[j] - 1.96 * np.sqrt(vj) * std_dev
            beta_max[j] = beta[j] + 1.96 * np.sqrt(vj) * std_dev

        (_, caps, _) = plt.errorbar(np.arange(0, len(beta)) ,beta, \
        yerr=(beta_max-beta), fmt='o', color='black',ecolor='lightgray',\
        elinewidth=3, capsize=6)
        for cap in caps:
            cap.set_markeredgewidth(1)
        plt.plot(beta, "k--")
        plt.title(r"Plot of the Confidence intervals at 95% for the \
        $\beta$ values", size=18)
        plt.xlabel(r"$j$", size=16)
        plt.ylabel(r"$\beta_j$ with its confidance intervals", size=16)
        plt.savefig("plots/{}".format(filename)) if filename!="" else ""
        plt.show()

        beta = self.beta if len(beta)==0 else beta
        # return np.sqrt(self.var(beta))



    def printstats(self, y_data=[], y_model=[]):
        mse = self.MSE(y_data, y_model)
        r2 = self.R2(y_data, y_model)
        print("The Mean Squared Error (MSE) is: {:.3e}".format(mse))
        print("The R2 Score Function is: {:.3f}\n".format(r2))



    def kfold_divide_data(self, n_splits=5, shuffle=True):
        """
        takes the y_data and performs a k_fold and divide it into
        n_splits parts.
        """
        n = len(self.y_data)

        indices = np.linspace(0, n-1, n, dtype=int)

        np.random.shuffle(indices)

        k_indices = np.array_split(indices, n_splits)

        temp_indices = indices
        test_indices_list = []
        train_indices_list = []
        self.X_train_list, self.X_test_list, self.y_train_list, self.y_test_list = [], [], [], []
        for i in range(n_splits):
            test_indices = k_indices[i]
            train_indices = indices[~np.in1d(indices,test_indices)]

            test_indices_list.append( list(test_indices) )
            train_indices_list.append( list(train_indices) )

            X_train, X_test = self.X_[train_indices], self.X_[test_indices]
            y_data_train, y_data_test = self.y_data[train_indices], self.y_data[test_indices]

            self.X_train_list.append(X_train)
            self.X_test_list.append(X_test)
            self.y_train_list.append(y_data_train)
            self.y_test_list.append(y_data_test)



    def kfold_predict(self):
        mse_train = []
        mse_test = []

        for i in range(5): ## Not 5
            X_train, X_test = self.X_train_list[i], self.X_test_list[i]
            y_train, y_test = self.y_train_list[i], self.y_test_list[i]


            y_train_model, beta = self.predict(X_train, y_train)
            mse_train.append( self.MSE( y_train, y_train_model ) )

            y_test_model = X_test @ beta
            mse_test.append(  self.MSE( y_test, y_test_model ) )
            # print("The Mean Squared Error (MSE) for fold {:d} is {:.6f}".format(i+1, mse))

        mse_avg_train = np.mean(mse_train)
        mse_avg_test = np.mean(mse_test)
        return mse_avg_train, mse_avg_test
