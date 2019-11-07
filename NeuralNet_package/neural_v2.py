import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import  train_test_split


class NeuralNetRegression:
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


    def setX(self,):
        xy_zip = np.array(list(zip(self.x,self.y)))
        poly = PolynomialFeatures(self.degree)   # using sklearn.preprocessing
        self.X_ = poly.fit_transform(xy_zip)

    def X(self):
        return self.X_


    def train_test_split(self, X, y, test_size=0.3, random_state=4):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test

    def initialize_weights(self, X, y):
        n_inputs, n_features = X.shape
        n_outputs = 1

        self.hidden_weights = np.random.randn(n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, n_outputs)
        self.output_bias = np.zeros(n_outputs) + 0.01

    def sigmoid(self,t):
        # Implement sigmoid
        return 1/(1+np.exp(-t))

    def cost(self, y_data, y_actual):
        return self.MSE(y_data, y_actual)

    def feed_forward(self, X):
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        # print("X:",np.shape(X))
        # print("w_h",self.hidden_weights.shape)
        # print("b_h",self.hidden_bias.shape)
        # print("z_h",z_h.shape)
        a_h = self.sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        ## Softmax activation:
            # exp_term = np.exp(z_o)
            # self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        ## Sigmoid activation
        self.probabilities = self.sigmoid(z_o)
        return a_h, self.probabilities

    def backpropagation(self, X, y):
        # Dont copy this. Dependent on cost function. Prolly easier though
        a_h, self.probabilities = self.feed_forward(X)

        # error in the output layer
        error_output = self.probabilities - y
        # print("error_o",error_output.shape)

        # error in the hidden layer
        d_sigmoid = a_h * (1 - a_h)
        # print("d_sig",d_sigmoid.shape)
        # print("w_o:",self.output_weights.shape)   
        error_hidden = np.matmul(error_output, self.output_weights.T) * d_sigmoid 

        # gradients for the output layer
        output_weights_gradient = np.matmul(a_h.T, error_output)
        output_bias_gradient = np.sum(error_output, axis=0)

        # # gradient for the hidden layer
        hidden_weights_gradient = np.matmul(X.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)

        return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient


    # def fit(self, X_train, y_train, X_test, y_test):
    #     # Perhaps pretty similar?
    #     self.backpropagation(X_train, y_train)
    #     return 0, 1, 2, 3


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
            train_score.append(self.R2(y_train, self.probabilities>0.5))

            test_predict = self.predict(X_test)
            test_loss.append( self.cost( test_predict , y_test) )
            test_score.append( self.R2(y_test, test_predict>0.5) )
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

        

    def predict(self, X):
        # This is basically feed forward without setting global variables
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = self.sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        ## Softmax activation:
            # exp_term = np.exp(z_o)
            # self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        ## Sigmoid activation
        probabilities = self.sigmoid(z_o)
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


#x,y,z,x_mesh,y_mesh, data = molding_data(veggli)
