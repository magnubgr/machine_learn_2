import numpy as np
import unittest
from LinearRegressionMethod.reg_solver import RegSolver
"""
"""

class Regsolver_test(unittest.TestCase):
    """
    This class is a testfunction for Regsolver. It tests ...
    """
    def test_canary(self):
        "testing that the simplest case works"
        self.assertEqual(2, 2)

    def test_var(self):
        """
        check that the variance is calculated correctly
        """
        np.random.seed(2)
        n = 50000
        sigma = 2

        x = np.linspace(0,1,n)
        y = np.linspace(0,1,n)
        y_data = np.random.normal(0,sigma,n)

        obj = RegSolver(x,y,y_data)
        self.assertAlmostEqual(obj.var(x),0.08, places=2)
        self.assertAlmostEqual(obj.var(y_data), sigma**2, places=1)


    def test_MSE(self):
        """
        Checks that the error of identical arrays will be zero for the
        Mean square error
        """
        n = 50000
        x = np.linspace(0,1,n)
        y_data = np.linspace(0,1,n)
        y_model = np.linspace(0,1,n)
        # return np.mean( (self.y_model-self.y_data)**2 )
        obj = RegSolver(x,y_model,y_data)
        self.assertEqual(obj.MSE(y_data=y_data, y_model = y_model),0)

    def test_R2(self):
        """
        Checks that the score of identical arrays will be 1 for
        the R_squared
        """
        n = 50000
        x = np.linspace(0,1,n)
        y_data = np.linspace(0,1,n)
        y_model = np.linspace(0,1,n)
        # return np.mean( (self.y_model-self.y_data)**2 )
        obj = RegSolver(x,y_model,y_data)
        self.assertEqual(obj.R2(y_data=y_data, y_model = y_model),1)


if __name__ == '__main__':
    unittest.main()
