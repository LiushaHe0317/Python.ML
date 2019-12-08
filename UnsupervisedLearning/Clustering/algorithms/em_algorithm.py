import numpy
from scipy.stats import multivariate_normal


class ExpectationMaximize:
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
        This method computes cluster responsibilities.

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
                resp[i, k] = multivariate_normal.pdf(data_points[i], means[k], covs[k]) * cluster_weights[k]

        # Add up responsibilities over each data point and normalize
        row_sums = resp.sum(axis=1)[:, numpy.newaxis]
        resp = resp / row_sums

        return resp

    def compute_soft_counts(self, resp):
        """
        This method compute the total responsibility assigned to each cluster, which will be useful when
        implementing M-steps.

        :param resp: A 2D ``numpy.ndarray`` of cluster responsibilities, where first dimension indicates the number of
        data points and the second dimension indicates the number of clusters.
        :return: A sequence of soft count for each cluster.
        """
        return numpy.sum(resp, axis=0)

    def compute_cluster_weights(self, counts):
        """
        This method update the cluster weights. The cluster weights The weight of cluster k is given by the ratio of
        the soft count N to the total number of data points N.

        :param counts: A 1D ``numpy.ndarray`` representing soft count of each cluster.
        :return: A sequence of cluster weights.
        """
        return [c / sum(counts) for c in counts]

    def compute_means(self, data_points, resp, counts):
        """
        This method computes the mean of each cluster. The mean of each cluster is set to the weighted average of all
        data points, weighted by the cluster responsibilities.

        :param data_points: A sequence of tuples representing all data points.
        :param resp: A sequence of tuples representing cluster responsibilities.
        :param counts: A sequence of cluster soft counts.
        :return: A sequence of cluster means.
        """
        num_clusters = len(counts)
        num_data = len(data_points)
        means = [numpy.zeros(len(data_points[0]))] * num_clusters

        for k in range(num_clusters):
            # Update means for cluster k using the M-step update rule for the mean variables.
            weighted_sum = 0.
            for i in range(num_data):
                weighted_sum += resp[i, k] * data_points[i]
            means[k] = weighted_sum / counts[k]

        return means

    def compute_covariances(self, data_points, resp, counts, means):
        """
        This method computes covariances, The covariance of each cluster is set to the weighted average of all outer
        products, weighted by the cluster responsibilities.

        :param data_points: A sequence of tuples representing all data points assigned to the cluster.
        :param resp: A sequence of tuples representing cluster responsibilities.
        :param counts: A sequence of cluster soft counts.
        :param means: A sequence of means.
        :return:
        """
        num_clusters = len(counts)
        num_dim = len(data_points[0])
        num_data = len(data_points)
        covariances = [numpy.zeros((num_dim, num_dim))] * num_clusters

        for k in range(num_clusters):
            # Update covariances for cluster k using the M-step update rule for covariance variables.
            weighted_sum = numpy.zeros((num_dim, num_dim))
            for i in range(num_data):
                weighted_sum += numpy.outer(data_points[i] - means[k], data_points[i] - means[k]) * resp[i, k]
            covariances[k] = weighted_sum / counts[k]

        return covariances

    def process(self, data_points, init_means, init_covariances, init_weights, max_iter=1000, thresh=1e-4):
        """
        This method implement EM algorithm.

        :param data_points: A sequence of tuples representing all data points assigned to the cluster.
        :param init_means:
        :param init_covariances:
        :param init_weights:
        :param max_iter:
        :param thresh:
        """
        # Make copies of initial parameters, which we will update during each iteration
        means = init_means[:]
        covariances = init_covariances[:]
        weights = init_weights[:]

        # Infer dimensions of dataset and the number of clusters
        num_data = len(data_points)
        num_clusters = len(means)

        # Initialize some useful variables
        resp = numpy.zeros((num_data, num_clusters))
        ll = self.log_likelihood(data_points, weights, means, covariances)
        ll_trace = [ll]

        for i in range(max_iter):

            # E-step: compute responsibilities
            resp = self.compute_responsibilities(data_points, weights, means, covariances)

            # M-step
            # Compute the total responsibility assigned to each cluster, which will be useful when
            # implementing M-steps below. In the lectures this is called N^{soft}
            counts = self.compute_soft_counts(resp)

            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            weights = self.compute_cluster_weights(counts)

            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            means = self.compute_means(data_points, resp, counts)

            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
            covariances = self.compute_covariances(data_points, resp, counts, means)

            # Compute the log-likelihood at this iteration
            ll_latest = self.log_likelihood(data_points, weights, means, covariances)
            ll_trace.append(ll_latest)

            # Check for convergence in log-likelihood and store
            if (ll_latest - ll) < thresh and ll_latest > -numpy.inf:
                print(f'stop at iteration {i}')
                break
            ll = ll_latest

        return {'weights': weights, 'means': means, 'covs': covariances, 'log-likelihood': ll_trace,
                'responsibilities': resp}
