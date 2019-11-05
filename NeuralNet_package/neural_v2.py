import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures

class NeuralNetRegression:
    def __init__(self,x,y,y_data, degree = 5):
        self.x = x
        self.y = y
        self.degree = degree
        self.n = len(x)
        self.setX()
        self.y_data =y.data

    def setX(self,):
        xy_zip = np.array(list(zip(self.x,self.y)))
        poly = PolynomialFeatures(self.degree)   # using sklearn.preprocessing
        self.X_ = poly.fit_transform(xy_zip)

    def X(self):
        return self.X_

    def Z(self,):
        return self.y_data

    def train_test_split(self,test_size=0.3, random_state=4):
        X_train, X_test, y_train, y_test = train_test_split(self.X_, self.y_data,\
        test_size=test_size, random_state=4)
        return X_train, X_test, y_train, y_test

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
