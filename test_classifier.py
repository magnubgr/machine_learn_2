from classifier import Classifier
import numpy as np
import unittest

class Regsolver_test(unittest.TestCase):
    """
    This class is a testfunction for Regsolver.
    """
    def test_canary(self):
        "testing that the simplest case works"
        self.assertEqual(2, 2)

    def test_sigmoid(self):
        """
        check that the sigmoid function is calculated correctly
        """
        np.random.seed(2)
        n = 50000;sigma = 2
        x = np.linspace(0,1,n)
        y = np.linspace(0,1,n)
        y_data = np.random.normal(0,sigma,n)
        obj = Classifier(x,y,y_data)
        self.assertAlmostEqual(obj.sigmoid(0),0.5)

if __name__ == '__main__':
    unittest.main()
