#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class NeuralNet:
    def __init__(self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):
        self.read_data = False


        # building our neural network

        n_inputs, n_features = X_data.shape
        n_hidden_neurons = 50
        n_categories = 10

        # we make the weights normally distributed using numpy.random.randn

        # weights and bias in the hidden layer
        hidden_weights = np.random.randn(n_features, n_hidden_neurons)
        hidden_bias = np.zeros(n_hidden_neurons) + 0.01

        # weights and bias in the output layer
        output_weights = np.random.randn(n_hidden_neurons, n_categories)
        output_bias = np.zeros(n_categories) + 0.01

    

    def sigmoid(self,t):
        return 1./(1+ np.exp(-t))

    def cost(self, beta, X, y):
        ## Using the cross-entropy cost function
        # y = y.reshape(-1,1) # Test if needed
        total_loss = -np.mean( y*np.log(self.prob(X, beta)) + (1-y)*np.log(1-self.prob(X, beta)) )
        return total_loss

    def feed_forward(self):
        pass

    def backward_propagation(self):
        pass

    def accuracy(self,y_actual,y_model):
        """
        A function that checks how often the arrays match by checking if
        every element in each element matches and divide by the number of elements
        """
        y_actual = y_actual.ravel()
        y_model = y_model.ravel()
        if len(y_actual) != len(y_model):
            raise ValueError("the dimension of your two arrays doesn't match")
        return np.sum(y_actual==y_model)/len(y_actual)

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
        ################## THIS IS WRONG. DONT SCALE 0 and 1 #############################################
        standard_scaler = StandardScaler()
        self.X = standard_scaler.fit_transform(self.X)
        # robust_scaler = RobustScaler()        # RobustScaler ignores outliers
        # self.X = robust_scaler.fit_transform(self.X)

        return self.X, self.y

    def train_test_split(self, X, y, test_size=0.3, random_state=4):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test
