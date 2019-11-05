from RegressionMethod.reg_solver import RegSolver
import numpy as np

class Ridge(RegSolver):

    def __init__(self, x, y, y_data, degree=5, _lambda=0.01):
        super().__init__(x, y, y_data, degree=degree)
        self._lambda = _lambda


    def predict(self, X, y_data):
        p = len(X[0])
        # X = self.X if len(X)==0 else X
        # y_data = self.y_data if len(y_data)==0 else y_data
        # _lambda = self._lambda if len(X)==0 else  _lambda

        self.beta = np.linalg.inv(X.T.dot(X) +self._lambda *np.identity(p)).dot(X.T).dot(y_data)
        self.y_model = X @ self.beta
        return self.y_model, self.beta
