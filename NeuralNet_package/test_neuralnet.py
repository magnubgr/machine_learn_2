from NeuralNet_package.neuralnet import NeuralNetClassifier
from NeuralNet_package.neural_v2 import NeuralNetRegression
import numpy as np
import unittest

class NeuralNetClassifier_test(unittest.TestCase):
    """
    This class is a testfunction for NeuralNetClassifier.
    """
    def test_canary(self):
        """testing that the simplest case works."""
        self.assertEqual(2, 2)

    def test_sigmoid(self):
        """
        check that the sigmoid function is calculated correctly
        """
        obj = NeuralNetClassifier()
        self.assertEqual(obj.sigmoid(0),0.5)
        self.assertAlmostEqual(obj.sigmoid(1), 1/(1+np.exp(-1)))

    def test_accuracy(self):
        """
        checks the accuracy function that checks the model with actual data
        """
        obj = NeuralNetClassifier()
        y_actual = np.array([1,1,0,0])
        y_model = np.array([1,1,1,0])
        with self.assertRaises(ValueError):
            obj.accuracy(y_actual, np.array([1,2,2]))
        self.assertEqual(obj.accuracy(y_actual,y_model),0.75)
        self.assertEqual(obj.accuracy(y_actual,y_actual),1)

    def test_display_data(self):
        """
        Test for display_data, that checks that it raises an exception, when
        trying to display the data without having read any data.
        """
        obj = NeuralNetClassifier()
        with self.assertRaises(SyntaxError):
            obj.display_data()


class NeuralNetRegression_test(unittest.TestCase):
    """
    This class is testfunction for NeuralNetRegression.
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

        obj = NeuralNetRegression(x,y)
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
        obj = NeuralNetRegression(x,y_model)
        self.assertEqual(obj.MSE(y_data=y_data, y_model = y_model),0)

    def test_R2(self):
        """
        Checks that the score of identical arrays will be 1 for
        the R_squared
        """
        n = 50000
        x = np.linspace(0,1,n)
        y1 = np.linspace(0,1,n)
        y2 = np.linspace(0,1,n)
        # return np.mean( (self.y_model-self.y_data)**2 )
        obj = NeuralNetRegression(x,y1)
        self.assertEqual(obj.R2(y_data=y1, y_model = y2),1)


if __name__ == '__main__':
    unittest.main()
