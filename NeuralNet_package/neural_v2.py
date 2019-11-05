import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures
from imageio import imread

class NeuralNetRegression:
    def __init__(self,x,y, degree = 5):
        self.y_data = self.FrankeFunction(x,y)
        self.x = x
        self.y = y
        self.degree = degree
        self.setX()

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

    def FrankeFunction(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

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
