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
        obj = Classifier(1,1,1)
        self.assertEqual(obj.sigmoid(0),0.5)
        self.assertAlmostEqual(obj.sigmoid(1), 1/(1+np.exp(-1)))


if __name__ == '__main__':
    unittest.main()
