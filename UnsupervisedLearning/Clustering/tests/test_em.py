import unittest
import numpy
from ..algorithms import ExpectationMaximize


class TestEMAlgorithm(unittest.TestCase):
    def setUp(self) -> None:
        self.algo = ExpectationMaximize()
        self.data = numpy.array([[1., 2.], [-1., -2.]])
        self.weights = numpy.array([0.3, 0.7])
        self.means = [numpy.array([0., 0.]), numpy.array([1., 1.])]
        self.covs = [numpy.array([[1.5, 0.], [0., 2.5]]),
                     numpy.array([[1., 1.], [1., 2.]])]

    def test_compute_responsibilities(self):
        resp = self.algo.compute_responsibilities(self.data, self.weights, self.means, self.covs)

        self.assertEqual((2,2), resp.shape, "compute_responsibilities() method returned output format incorrectly")
        self.assertTrue(numpy.allclose(resp, numpy.array([[0.10512733, 0.89487267],
                                                          [0.46468164, 0.53531836]])),
                        "compute_responsibilities() method returned results incorrect")

    def test_compute_cluster_weights(self):
        data = numpy.array([[1., 2.], [-1., -2.], [0, 0]])
        resp = self.algo.compute_responsibilities(data, self.weights, self.means, self.covs)
        counts = self.algo.compute_soft_counts(resp)
        weights = self.algo.compute_cluster_weights(counts)

        self.assertTrue(numpy.allclose(weights, [0.27904865942515705, 0.720951340574843]), "updated weights incorrect")

    def test_compute_means(self):
        resp = self.algo.compute_responsibilities(self.data, self.weights, self.means, self.covs)
        counts = self.algo.compute_soft_counts(resp)

        means = self.algo.compute_means(self.data, resp, counts)

        self.assertTrue(numpy.allclose(means, numpy.array([[-0.6310085, -1.262017], [0.25140299, 0.50280599]])),
                        "returned means incorrect.")

    def test_compute_covariances(self):
        resp = self.algo.compute_responsibilities(self.data, self.weights, self.means, self.covs)
        counts = self.algo.compute_soft_counts(resp)

        means = self.algo.compute_means(self.data, resp, counts)

        covariances = self.algo.compute_covariances(self.data, resp, counts, means)

        self.assertTrue(numpy.allclose(covariances[0], numpy.array([[0.60182827, 1.20365655], [1.20365655, 2.4073131]])),
                        "first covariance incorrect")
        self.assertTrue(numpy.allclose(covariances[1], numpy.array([[ 0.93679654, 1.87359307], [1.87359307, 3.74718614]])),
                        "second covariance incorrect")
