#!/usr/bin/env python
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import scipy.special as scs
import pandas as pd
import numpy as np
import os


"""
Super-class for the Neural Network classes
"""
class NeuralNet:

    def sigmoid(self,t):
        """ Returns the sigmoid function of a given number 't' """
        return 1.0/(1 + np.exp(-t))

    def train_test_split(self, X, y, test_size=0.3, random_state=4):
        """ 
        This method is just an implementation of ..
        Scikit-Learns train_test_split for use in our own class.         
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test


    def initialize_weights(self, X, n_output_neurons=1):
        """
        This method creates the architecture of the network.
        It initializes the weights and biases of the system.
        Only works for a one hidden layer neural network.  
        """
        n_inputs, n_features = X.shape
        n_outputs = n_output_neurons

        self.hidden_weights = np.random.randn(n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, n_outputs)
        self.output_bias = np.zeros(n_outputs) + 0.01

    def feed_forward(self, X):
        """
        The feed forward method takes the input and runs it through the network.
        That means adding the weighted sum, using activation function for both .. 
        the hidden layer and the output layer. 
        """
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        probabilities = self.sigmoid(z_o)
        return a_h, probabilities





"""
Neural Network for Classification.
"""
class NeuralNetClassifier(NeuralNet):
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
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.read_data = False

    def cost(self, y_pred, y):
        """
        This cost method uses the cross-entropy loss function.
        """
        total_loss = -np.mean(
                            scs.xlogy(y,y_pred)
                            + scs.xlogy(1-y, 1-y_pred)
                            )
        return total_loss

    def fit(self, X_train, y_train, X_test, y_test):
        """
        This fit method uses the regular gradient descent to train the network. 
        It also stores the loss and scores for each learning iteration for test and train sets.
        The method calls the backpropagation method.
        """
        train_loss, test_loss = [], []
        train_score, test_score = [], []
        eta = self.learning_rate
        lmbd = self.L2_penalty
        counter = 0
        a_h, self.probabilities = self.feed_forward(X_train)
        # self.probabilities = np.zeros(len(y_train))

        for i in range(self.max_iter):

            cost1 = self.cost(self.probabilities, y_train)
            train_loss.append(cost1)
            train_score.append(self.accuracy(y_train, self.probabilities>0.5))

            test_predict = self.predict(X_test)
            test_loss.append( self.cost( test_predict , y_test) )
            test_score.append( self.accuracy(y_test, test_predict) )
            if self.verbose:
                print ("Cost:", cost1)

            dWo, dBo, dWh, dBh = self.backpropagation(X_train, y_train)

            cost2 = self.cost(self.probabilities, y_train)

            if abs(cost1-cost2)<self.tol:
                if counter==1:
                    print("tolerance value reached")
                    break
                counter += 1
            else:
                counter = 0

            dWo += lmbd * self.output_weights
            dWh += lmbd * self.hidden_weights

            # update weights and biases
            self.output_weights -= eta * dWo
            self.output_bias -= eta * dBo
            self.hidden_weights -= eta * dWh
            self.hidden_bias -= eta * dBh
        return train_loss, test_loss, train_score, test_score


    def predict(self, X):
        """
        This method takes the probabilites from the feed forward method,
        and returns an output with 0's and 1's.
        """
        a_h, probabilities = self.feed_forward(X)
        return probabilities>0.5


    def backpropagation(self, X, y):
        """
        The backpropagation method is in charge of creating the gradients, 
        and uses matrix multiplication to make use of numpy's speed.
        """
        a_h, self.probabilities = self.feed_forward(X)

        error_output = self.probabilities - y
        d_sigmoid = a_h * (1 - a_h)
        error_hidden = np.matmul(error_output, self.output_weights.T) * d_sigmoid # This last oart is ofcourse the derivative of the sigmoid

        output_weights_gradient = np.matmul(a_h.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)

        hidden_weights_gradient = np.matmul(X.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)
        
        # a_h, self.probabilities = self.feed_forward(X)
        # a_o = self.probabilities
        # dLoss_a_o = y/a_o - (1-y)/(1-a_o)
        # dLoss_z_o = dLoss_a_o * a_o * (1 - a_o)
        # dLoss_a_h = self.output_weights.T @ dLoss_z_o
        # dLoss_w_o = 

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
        """
        Reads the credit card data
        Preprocessing the data.
        Returns the total design matrix X and the output data y.
        """

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

        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X)
        # robust_scaler = RobustScaler()        # RobustScaler ignores outliers
        # X = robust_scaler.fit_transform(X)

        return X, y






"""
Neural Network class for Regression Analysis.
"""

class NeuralNetRegression(NeuralNet):
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
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.read_data = False


    def cost(self, y_data, y_actual):
        """
        This method uses the regular Mean Squared Error to create the cost.
        """
        return self.MSE(y_data, y_actual)


    def backpropagation(self, X, y):
        """
        The backpropagation method is in charge of creating the gradients, 
        and uses matrix multiplication to make use of numpy's speed.
        """
        a_h, self.probabilities = self.feed_forward(X)

        error_output = self.probabilities - y
        d_sigmoid = a_h * (1 - a_h)
        error_hidden = np.matmul(error_output, self.output_weights.T) * d_sigmoid

        output_weights_gradient = np.matmul(a_h.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)

        hidden_weights_gradient = np.matmul(X.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)

        return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient


    def fit(self, X_train, y_train, X_test, y_test):
        """
        This fit method uses the regular gradient descent to train the network. 
        It also stores the loss and scores for each learning iteration for test and train sets.
        The method calls the backpropagation method.
        """
        train_loss, test_loss = [], []
        train_score, test_score = [], []
        eta = self.learning_rate
        lmbd = self.L2_penalty
        counter = 0
        a_h, self.probabilities = self.feed_forward(X_train)
        # self.probabilities = np.zeros(len(y_train))

        for i in range(self.max_iter):

            cost1 = self.cost(self.probabilities, y_train)
            train_loss.append(cost1)
            train_score.append(self.R2(y_train, self.probabilities))

            test_predict = self.predict(X_test)
            test_loss.append( self.cost( test_predict , y_test) )
            test_score.append( self.R2(y_test, test_predict) )
            if self.verbose:
                print ("Cost:", cost1)

            dWo, dBo, dWh, dBh = self.backpropagation(X_train, y_train)

            cost2 = self.cost(self.probabilities, y_train)

            if abs(cost1-cost2)<self.tol:
                if counter==1:
                    print("tolerance value reached")
                    break
                counter += 1
            else:
                counter = 0

            dWo += lmbd * self.output_weights
            dWh += lmbd * self.hidden_weights

            # update weights and biases
            self.output_weights -= eta * dWo
            self.output_bias -= eta * dBo
            self.hidden_weights -= eta * dWh
            self.hidden_bias -= eta * dBh
        return train_loss, test_loss, train_score, test_score


    def predict(self, X):
        """
        This method returns the prediction from the neural network.
        """
        a_h, probabilities = self.feed_forward(X)
        return probabilities

    def var(self,l):
        """
        returns the variance of an 1-dimensional array
        """
        return np.sum((l-np.mean(l))**2)/len(l)

    def MSE(self, y_data, y_model):
        """
        takes the MSE of two arrays, y_data being the data and y_model being
        the model fitted on the data
        """
        return np.mean( (y_model-y_data)**2 )

    def R2(self, y_data, y_model):
        """
        Calculates the R2 score of the function of the y_model
        comaped to the y_data. 1 is the best score and 0 is the worst.
        """
        return 1 - np.sum((y_data - y_model) ** 2) /\
         np.sum((y_data - np.mean(y_data)) ** 2)


