import unittest
import numpy
from ..EM_algorithm import EMAlgorithm


class TestEMAlgorithm(unittest.TestCase):
    def setUp(self) -> None:
        self.algo = EMAlgorithm()
        self.data = numpy.array([[1., 2.], [-1., -2.]])
        self.weights = numpy.array([0.3, 0.7])
        self.means = [numpy.array([0., 0.]), numpy.array([1., 1.])]
        self.covs = [numpy.array([[1.5, 0.], [0., 2.5]]),
                     numpy.array([[1., 1.], [1., 2.]])]

    def test_compute_responsibilities(self):
        resp = self.algo.compute_responsibilities(self.data, self.weights, self.means, self.covs)

        self.assertEqual((2,2), resp.shape, "compute_responsibilities() method returned output format incorrectly")
        self.assertTrue(numpy.allclose(resp,
                                       numpy.array([[0.10512733, 0.89487267], [0.46468164, 0.53531836]])))