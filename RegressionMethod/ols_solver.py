from RegressionMethod.reg_solver import RegSolver
import numpy as np

class OLS(RegSolver):
    def __init__(self,x,y,y_data, degree = 5):
        super().__init__(x,y,y_data, degree = degree)


    def predict(self, X, y_data):
        # X = self.X if len(X)==0 else X
        # y_data = self.y_data if len(y_data)==0 else y_data

        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_data)
        self.y_model = X @ self.beta
        return self.y_model, self.beta
