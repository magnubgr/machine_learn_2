#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler #, OneHotEncoder
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from functools import partial
import sys
import time

import pandas as pd
import sklearn
import os


"""
Describe the model and input argument
"""


class Classifier:
    def __init__(self):
    # def __init__(self,x,y,y_data):

        # self.x = x
        # self.y = y
        # self.y_data = y_data
        # self.X = self.get_x()
        self.read_data = False

    def sigmoid(self, t):
        return 1./(1 + np.exp(-t))

    ################# Make the probability function which uses the sigmoid to make the "activation" function #################
    def prob(self, X, beta):
        return self.sigmoid( np.dot(X, beta) )

    def cost_function(self, beta, X, y):
        ##### With basic for-loop
        # n = len(self.y_data)
        # total_loss = 0
        # for i in range(n):
        #     total_loss -= y[i]*np.log(prob(X[i], beta)) + (1-y[i])*np.log(1-prob(X[i],beta))
        ##### With numpy sum
        total_loss = -np.sum( y*np.log(self.prob(X, beta)) + (1-y)*np.log(1-self.prob(X, beta)) )

        return total_loss

    def newt_it(self, X, n, gamma, tol=1e-2):
        #not working
        old_beta = 1
        #newtons iterative method
        for i in range(n):
            new_beta = old_beta - gamma*(X.T *(prob(X,old_beta)-y_data))
            if abs(new_beta-old_beta)<tol:
                break

            old_beta = new_beta
        self.beta = new_beta

    def gradient_descent(self, X, y, learning_rate=0.2, n_iter=100):
        beta_new = np.zeros((X.shape[1],1))
        beta_new = np.random.rand(X.shape[1],1)
        tol = 1e-2
        m = len(y)
        for i in range(n_iter):
            beta_old= beta_new
            print(100*i/n_iter)
            time1 = time.time()
            #print('X', np.shape(X),'Beta', np.shape(beta_old),'y',np.shape(y.ravel()))
            gradients =  np.dot(X.T, (self.prob(X, beta_old) - y.ravel()))
            time2 = time.time()
            beta_new = beta_old - learning_rate*gradients
            time3 = time.time()
            print("time2-1  =",time2-time1)
            print("time3-2  =",time3-time2)
            print("time3-1  =",time3-time1)
            if abs(np.sum(beta_new-beta_old))<tol:
                break

        return beta_new


    def read_credit_card_file(self, xls_file):
        #extract the data
        self.read_data = True
        path = os.path.dirname(os.path.realpath(__file__))
        file = path + "/../data/"+ xls_file # fix this pointer

        # df = pd.read_excel(file, header=[0, 1], sheetname="Data")
        self.df = pd.read_excel(file, header=1, skiprows=0, index_col=0)

        self.df.rename(
            index=str,
            columns={"default payment next month": "DEFAULT"},
            inplace=True
        )

        ## Drop rows that are outside of features given
        self.df = self.df[(self.df.EDUCATION != 0) &
                          (self.df.EDUCATION != 5) &
                          (self.df.EDUCATION != 6)]
        self.df = self.df[ (self.df.MARRIAGE != 0) ]

        for dfpay in [self.df.PAY_0, self.df.PAY_2, self.df.PAY_3, self.df.PAY_4, self.df.PAY_5, self.df.PAY_6]:
            self.df = self.df[(dfpay != -2) ]
                        # &(dfpay != 0)]

        # One-hot encoding the gender
        self.df["MALE"] = (self.df["SEX"]==1).astype("int")
        self.df.drop("SEX", axis=1, inplace=True)

        # One-hot encoding for education
        self.df["GRADUATE_SCHOOL"] = (self.df["EDUCATION"]==1).astype("int")
        self.df["UNIVERSITY"] = (self.df["EDUCATION"]==2).astype("int")
        self.df["HIGH_SCHOOL"] = (self.df["EDUCATION"]==3).astype("int")
        self.df.drop("EDUCATION", axis=1, inplace=True)

        # One-hot encoding for marriage
        self.df["MARRIED"] = (self.df["MARRIAGE"]==1).astype("int")
        self.df["SINGLE"] = (self.df["MARRIAGE"]==2).astype("int")
        self.df.drop("MARRIAGE", axis=1, inplace=True)

        self.X = self.df.loc[:, self.df.columns != 'DEFAULT'].values
        self.y = self.df.loc[:, self.df.columns == 'DEFAULT'].values

        ## Scale the features. So that for example LIMIT_BAL isnt larger than AGE
        standard_scaler = StandardScaler()
        # robust_scaler = RobustScaler()        # RobustScaler ignores outliers
        self.X = standard_scaler.fit_transform(self.X)

        return self.X, self.y

    def display_data(self):
        """
        prints the df to display the data
        Checks that you have read the data
        """
        if (self.read_data==True):
            print ("Checking out the numbers in the dataset")
            print(self.df)
        else:
            raise SyntaxError("need to read the data first" )

    def train_test_split(self, X, y, test_size=0.3, random_state=4):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test

    def fit_data(self, X_train, y_train, learning_rate=0.1, n_iter=100):
        ##### Scikit-Learn Logistic regression #####
        # self.log_reg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        # self.log_reg.fit(self.X, self.y.flatten())

        ##### Our implementation of Logistic regression #####

        # X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Adding intercept
        print(np.shape(y_train))
        print(y_train)
        y_train = y_train[:, np.newaxis]
        print(np.shape(y_train))
        print(y_train)
        self.beta = self.gradient_descent(X_train, y_train, learning_rate=learning_rate, n_iter=n_iter)


    def predict(self, X_test):
        ##### Scikit-Learn predict #####
        # prediction = self.log_reg.predict(X_test)

        ##### Our implementation of predict #####
        # X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test] # Add ones to first column
        prediction = self.prob(X_test, self.beta)

        return prediction

    def accuracy(self,y_actual,y_model): #if decice to change
        """
        A function that checks how often the arrays match by checking if
        every element in each element matches and divide by the number of elements
        """
        if len(y_actual) != len(y_model):
            raise ValueError("the dimension of your two arrays doesn't match")
        return np.sum(y_actual==y_model)/len(y_actual)
