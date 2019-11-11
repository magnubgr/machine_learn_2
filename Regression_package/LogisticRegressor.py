#!/usr/bin/env python
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import os


"""
Describe the model and input argument
"""

class LogisticRegression:
    def __init__(self, verbose=False):
        
        self.read_data = False

    def sigmoid(self, t):
        """ Returns the sigmoid function of a given number 't' """
        return 1.0/(1 + np.exp(-t))

    def prob(self, X, beta):
        """ Calculates the probabilites with our selected beta-values

        Args:
            X: The design matrix X, here this is the matrix of datapoints
            beta: Our calculated coefficients, beta, for our solution

        Returns:
            An array with sigmoided elements from the dot product of X and beta

        """
        return self.sigmoid( np.dot(X, beta) )

    def cost_function(self, beta, X, y):
        """ Cross-entropy cost function """
        y = y.reshape(-1,1)
        total_loss = -np.mean( y*np.log(self.prob(X, beta)) + (1-y)*np.log(1-self.prob(X, beta)) )
        return total_loss

    def gradient_descent(self, X_train, X_test, y_train, y_test, learning_rate, n_iter, tol, verbose):
        """Standard gradient descent for optimizing the beta-coefficients.

        Args:
            X_train: Input/desigm matrix for training.
            X_test: Input/desigm matrix for testing.
            y_train: The output data or targets for training.
            y_test: The output data or targets for testing.
            learning_rate: The learning rate for our gradient descent.
            n_iter: Maximum iterations in the gradient descent.
            tol: Tolerance to stop learning. When changes are below tol.
            verbose: True or False to print the learning of the training set.

        Returns:
            beta_new: The coefficients for our solution.
            train_costs: A list of the cost through learning for the training set.
            test_costs: A list of the cost through learning for the testing set.
            train_scores: A list of the scores through learning for the training set.
            test_scores: A list of the scores through learning for the testing set

        """
        np.random.seed(12)
        beta_new = np.random.rand(X_train.shape[1],1)
        m = len(y_train)
        train_costs, test_costs = [],[]
        train_scores, test_scores = [],[]
        for i in range(n_iter):
            test_costs.append( self.cost_function(beta_new, X_test, y_test) )
            train_costs.append( self.cost_function(beta_new, X_train, y_train) )
            train_scores.append( self.accuracy(y_train, self.prob(X_train,beta_new)>0.5) )
            test_scores.append( self.accuracy(y_test,  self.prob(X_test,beta_new)>0.5) )

            if verbose:
                print("Loss:",train_costs[i])

            beta_old = beta_new
            gradients = (1/m) * np.dot(X_train.T, (self.prob(X_train, beta_old) - y_train.reshape(-1,1)))
            beta_new = beta_old - learning_rate * gradients

            if i>1 and abs(train_costs[i]-train_costs[i-1])<tol:
                print("Below tolerance: ", tol)
                break

        return beta_new, train_costs, test_costs, train_scores, test_scores


    def read_credit_card_file(self, xls_file):
        """ 
        Reads the credit card data
        Preprocessing the data.
        Returns the total design matrix X and the output data y.
        """
        #extract the data
        self.read_data = True
        path = os.path.dirname(os.path.realpath(__file__))
        file = path + "/../data/" + xls_file

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
        self.X = standard_scaler.fit_transform(self.X)
        # robust_scaler = RobustScaler()        # RobustScaler ignores outliers
        # self.X = robust_scaler.fit_transform(self.X)

        return self.X, self.y

    def display_data(self):
        """
        Prints the df to display the data
        Also checks that you have read the data
        """
        if (self.read_data==True):
            print ("Checking out the numbers in the dataset")
            print(self.df)
        else:
            raise SyntaxError("need to read the data first" )

    def train_test_split(self, X, y, test_size=0.33, random_state=4):
        """ 
        This method is just an implementation of ..
        Scikit-Learns train_test_split for use in our own class.         
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test

    def fit_data(self, X_train, X_test, y_train, y_test, learning_rate=1.4, tol=1e-3, n_iter=300,verbose=False):
        """
        This method calls the gradient descent, and could be improved ..
        to contain SGD, batches or mini-batches.
        """
        y_train = y_train[:, np.newaxis]
        y_test = y_test[:, np.newaxis]
        self.beta, train_costs,test_costs,train_scores,test_scores  = self.gradient_descent(X_train, X_test, y_train, y_test, learning_rate=learning_rate, tol=tol, n_iter=n_iter, verbose=verbose)
        return self.beta,train_costs,test_costs,train_scores,test_scores

    def predict(self, X_test):
        """
        This method returns the array of predicted outputs by setting..
        all values above 0.5 to be 1, and the rest to 0.
        """
        prediction = self.prob(X_test, self.beta)
        return (prediction>=0.5)

    def accuracy(self,y_actual,y_model): #if decide to change
        """
        A function that checks how often the arrays match by checking if
        every element in each element matches and divide by the number of elements
        """
        y_actual = y_actual.ravel()
        y_model = y_model.ravel()
        if len(y_actual) != len(y_model):
            raise ValueError("the dimension of your two arrays doesn't match")
        return np.sum(y_actual==y_model)/len(y_actual)
