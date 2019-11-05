from Classifier_package.classifier import Classifier
import numpy as np
import unittest

class Classifier_test(unittest.TestCase):
    """
    This class is a testfunction for Regsolver.
    """
    def test_canary(self):
        """testing that the simplest case works."""
        self.assertEqual(2, 2)

    def test_sigmoid(self):
        """
        check that the sigmoid function is calculated correctly
        """
        obj = Classifier()
        self.assertEqual(obj.sigmoid(0),0.5)
        self.assertAlmostEqual(obj.sigmoid(1), 1/(1+np.exp(-1)))

    def test_accuracy(self):
        """
        checks the accuracy function that checks the model with actual data
        """
        obj = Classifier()
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
        obj = Classifier()
        with self.assertRaises(SyntaxError):
            obj.display_data()


if __name__ == '__main__':
    unittest.main()
