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


class Classifier:
    def __init__(self,x,y,y_data):

        self.x = x
        self.y = y
        self.y_data = y_data
        #self.n = len(x)
        

    def X(self):
        return self.X_

    def var(self,l):
        """
        returns the variance of an 1-dimensional array
        """
        return np.sum((l-np.mean(l))**2)/len(l)

    def sigmoid(self, t):
        return np.exp(t)/(np.exp(t)+1)


obj = Classifier(1,23,4)
print(obj.sigmoid(0.23))



'''
    def getX(self):
        """
        Creates the design matrix X based on x and y data
        The degree can be adjusted from the init
        """
        xy_zip = np.array(list(zip(self.x,self.y)))
        poly = PolynomialFeatures(self.degree)      # using sklearn.preprocessing
        self.X_ = poly.fit_transform(xy_zip)
        return self.X_
'''
