#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler #, OneHotEncoder
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from functools import partial
import sys
from Classifier_package.classifier import Classifier
import pandas as pd
import sklearn
import os


"""
runs the classifier class with data from the data package
"""


xls_file = "default_credit_card_data.xls"
obj = Classifier(1,23,4)
obj.read_credit_card_file(xls_file)
obj.display_data()
X,y = obj.fit_data()