from RegressionMethod.reg_solver import RegSolver
import sklearn.linear_model as skl
import numpy as np


class Lasso(RegSolver):
    def __init__(self,x,y,y_data, degree = 5, _lambda=0.01):
        super().__init__(x,y,y_data, degree=degree)
        self._lambda = _lambda

    def predict(self, X, y_data):
        # X      = self.X      if len(X)==0      else X
        # y_data = self.y_data if len(y_data)==0 else y_data
        clf_lasso = skl.Lasso(alpha=self._lambda,  fit_intercept=True).fit(X, y_data)
        self.beta = clf_lasso.coef_
        self.beta[0] = clf_lasso.intercept_
        self.y_model = X @ self.beta# clf_lasso.predict(X)
        return self.y_model, self.beta
