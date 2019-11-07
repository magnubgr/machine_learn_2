#!/usr/bin/env python
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import scipy.special as scs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class NeuralNetClassifier:
    def __init__(
            self,
            n_hidden_neurons,
            L2_penalty=0.0001,
            learning_rate=0.001,
            max_iter=1000,
            tol=1e-5,
            verbose=False):

        self.n_hidden_neurons = n_hidden_neurons
        self.L2_penalty = L2_penalty
        self.learning_rate = learning_rate
        self.max_iter = 1000
        self.tol = tol
        self.verbose = verbose
        self.read_data = False


    def initialize_weights(self, X, y):
        n_inputs, n_features = X.shape
        n_outputs = len(np.unique(y))-1

        self.hidden_weights = np.random.randn(n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, n_outputs)
        self.output_bias = np.zeros(n_outputs) + 0.01


    def sigmoid(self,t):
        return 1/(1+ np.exp(-t))
    # def d_sigmoid(self,t):
    #     return self.sigmoid(t) * ( 1 - self.sigmoid(t) )


    def cost(self,y_pred, y):
        ## Using the cross-entropy cost function
        # y = y.reshape(-1,1) # Test if needed
        total_loss = -np.mean(
                            scs.xlogy(y,y_pred)
                            + scs.xlogy(1-y, 1-y_pred)
                            )
        return total_loss

    def fit(self, X_train, y_train, X_test, y_test):
        train_loss = []
        test_loss = []
        train_score = []
        test_score = []
        eta = self.learning_rate
        lmbd = self.L2_penalty
        counter = 0
        self.probabilities = np.zeros(len(y_train))

        for i in range(self.max_iter):

            cost1 = self.cost(self.probabilities, y_train)
            train_loss.append(cost1)
            train_score.append(self.accuracy(y_train, self.probabilities>0.5))

            test_predict = self.predict(X_test)
            test_loss.append( self.cost( test_predict , y_test) )
            test_score.append( self.accuracy(y_test, test_predict>0.5) )
            if self.verbose:
                print ("cost_function", cost1)

            dWo, dBo, dWh, dBh = self.backpropagation(X_train, y_train)

            cost2 = self.cost(self.probabilities, y_train)

            if abs(cost1-cost2)<self.tol:
                if counter==1:
                    print("tolerance value reached")
                    break
                counter += 1
            else:
                counter =0
            # if ac1>ac2:
            #     print ("hail Cthulhu devourer of worlds")
            # regularization term gradients

            dWo += lmbd * self.output_weights
            dWh += lmbd * self.hidden_weights

            # update weights and biases
            self.output_weights -= eta * dWo
            self.output_bias -= eta * dBo
            self.hidden_weights -= eta * dWh
            self.hidden_bias -= eta * dBh
        return train_loss, test_loss, train_score, test_score

    def feed_forward(self, X):
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        ## Softmax activation:
            # exp_term = np.exp(z_o)
            # self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        ## Sigmoid activation
        self.probabilities = self.sigmoid(z_o)

        return a_h, self.probabilities

    def predict(self, X):
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        ## Softmax activation:
            # exp_term = np.exp(z_o)
            # self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        ## Sigmoid activation
        probabilities = self.sigmoid(z_o)

        return probabilities

    def backpropagation(self, X, y):
        a_h, self.probabilities = self.feed_forward(X)
        # print(f"accuracy: {self.accuracy(self.probabilities>0.5, y)}")

        # error in the output layer
        error_output = self.probabilities - y
        # error in the hidden layer
        d_sigmoid = a_h * (1 - a_h)
        error_hidden = np.matmul(error_output, self.output_weights.T) * d_sigmoid # This last oart is ofcourse the derivative of the sigmoid

        # gradients for the output layer
        output_weights_gradient = np.matmul(a_h.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)

        # gradient for the hidden layer
        hidden_weights_gradient = np.matmul(X.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)

        return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient



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
        prints the df to display the data.
        Checks that you have read the data.
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

        for dfpay in [self.df.PAY_0, self.df.PAY_2,
                    self.df.PAY_3, self.df.PAY_4,
                    self.df.PAY_5, self.df.PAY_6]:
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

        X = self.df.loc[:, self.df.columns != 'DEFAULT'].values
        y = self.df.loc[:, self.df.columns == 'DEFAULT'].values
        ## Scale the features. So that for example LIMIT_BAL isnt larger than AGE
        ################## THIS IS WRONG. DONT SCALE 0 and 1 #############################################
        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X)
        # robust_scaler = RobustScaler()        # RobustScaler ignores outliers
        # X = robust_scaler.fit_transform(X)

        return X, y

    def train_test_split(self, X, y, test_size=0.3, random_state=4):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test
