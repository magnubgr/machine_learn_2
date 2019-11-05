import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures

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

    def train_test_split(self,test_size=0.3, random_state=4):
        X_train, X_test, y_train, y_test = train_test_split(self.X_, self.y_data,\
        test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test

    def fix_and_maybe_scale_data_input(self):
        pass

    def initialize_weights(self, X, y):
        n_inputs, n_features = X.shape
        n_outputs = 1

        self.hidden_weights = np.random.randn(n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, n_outputs)
        self.output_bias = np.zeros(n_outputs) + 0.01

    def sigmoid(self):
        # Implement sigmoid
        pass

    def cost(self):
        # IMPLEMENT LEAST SQUARES COST
        #  SEE MORTENS SLIDE FOR MORE        
        pass

    def feed_forward(self):
        # 
        pass

    def backpropagation(self):
        
        pass

    def fit(self):
        pass

    def predict(self):
        pass




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
