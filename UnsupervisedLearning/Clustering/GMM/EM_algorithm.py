import numpy


class EMAlgorithm:
    """
    This class implements the EM algorithm.
    """
    def log_likelihood(self, data_points, cluster_weights, means, covs):
        """
        Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters.

        :param data_points: A sequence of tuple of data point coordinates.
        :param cluster_weights: A sequence of cluster weights.
        :param means: A sequence of means.
        :param covs: A sequence of covariances.
        """
        num_clusters = len(means)
        num_dim = len(data_points[0])

        likelihood = 0
        for data in data_points:
            Z = numpy.zeros(num_clusters)
            for k in range(num_clusters):
                # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
                delta = numpy.array(data) - means[k]
                exp_term = numpy.dot(delta.T, numpy.dot(numpy.linalg.inv(covs[k]), delta))

                # Compute log-likelihood contribution for this data point and this cluster
                Z[k] += numpy.log(cluster_weights[k])
                Z[k] -= 1 / 2. * (num_dim * numpy.log(2 * numpy.pi) + numpy.log(numpy.linalg.det(covs[k])) + exp_term)

            # Increment log-likelihood contribution of this data point across all clusters
            likelihood += self._log_sum_exp(Z)

        return likelihood

    def _log_sum_exp(self, Z):
        """
        Compute cumulative sum of log  for array Z.

        :param Z: A 1D ``numpy.ndarray`` of log-likelihood.
        """
        return numpy.max(Z) + numpy.log(numpy.sum(numpy.exp(Z - numpy.max(Z))))

    def compute_responsibilities(self, data_points, cluster_weights, means, covs):
        """
        E-step: compute responsibilities, given the current parameters

        :param data_points: A sequence of tuple of data point coordinates.
        :param cluster_weights: A sequence of cluster weights.
        :param means: A sequence of means.
        :param covs: A sequence of covariances.
        """
        num_data = len(data_points)
        num_clusters = len(means)
        resp = numpy.zeros((num_data, num_clusters))

        # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
        for i in range(num_data):
            for k in range(num_clusters):
                # YOUR CODE HERE
                resp[i, k] = ...

        # Add up responsibilities over each data point and normalize
        row_sums = resp.sum(axis=1)[:, numpy.newaxis]
        resp = resp / row_sums

        return resp

    def process(self):
        ...