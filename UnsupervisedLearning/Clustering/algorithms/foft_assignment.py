from copy import deepcopy
import numpy
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances


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

    def log_pdf_diagonal_gaussian(self, x, mean, cov):
        """
        This method Computes log probability of a multivariate Gaussian distribution with diagonal covariance at a given
        data point. A multivariate Gaussian distribution with a diagonal covariance is equivalent to a collection of
        independent Gaussian random variables.

        :param x: A sparse matrix. The logpdf will be computed for each row of x.
        :param mean: A 1D ``numpy.ndarray``.
        :param cov: A 1D ``numpy.ndarray``.
        """

        n = x.shape[0]
        dim = x.shape[1]
        assert (dim == len(mean) and dim == len(cov))

        # multiply each i-th column of x by (1/(2*sigma_i)), where sigma_i is sqrt of variance of i-th variable.
        scaled_x = x.dot(self._diag(1. / (2 * numpy.sqrt(cov))))

        # multiply each i-th entry of mean by (1/(2*sigma_i))
        scaled_mean = mean / (2 * numpy.sqrt(cov))

        # sum of pairwise squared Eulidean distances gives SUM[(x_i - mean_i)^2/(2*sigma_i^2)]
        return -numpy.sum(numpy.log(numpy.sqrt(2 * numpy.pi * cov))) - pairwise_distances(scaled_x, [scaled_mean],
                                                                                          'euclidean').flatten() ** 2

    def _log_sum_exp(self, x, axis=0):
        """
        Compute the log of a sum of exponentials.

        :param x: A ``numpy.ndarray`` of log-likelihood.
        :param axis: An integer that indicates dimension.
        :return: A ``numpy.ndarray`` indicating the log of a sum of exponentials.
        """
        x_max = numpy.max(x, axis=axis)
        if axis == 1:
            return x_max + numpy.log(numpy.sum(numpy.exp(x - x_max[:, numpy.newaxis]), axis=1))
        else:
            return x_max + numpy.log(numpy.sum(numpy.exp(x - x_max), axis=0))

    def _diag(self, array):
        """
        A helper function which creates a sparse diagonal matrix.

        :param array: An array of digits.
        """
        n = len(array)
        return spdiags(array, 0, n, n)

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
        :param init_means: A sequence of tuples of mean coordinates for each cluster.
        :param init_covariances: A sequence of ``numpy.ndarray`` of covariance for each cluster.
        :param init_weights: A sequence of weights for each cluster.
        :param max_iter: Maximum number of iterations.
        :param thresh: A digit indicating the threshold.
        :return: A dictionary that caches updated log-likelihood, means, covariances, weights and cluster responsibility
            matrix.
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

    def process_for_high_dimensions(self, data_points, init_means, init_covs, init_weights, cov_smoothing=1e-5,
                                    max_iter=int(1e3),
                                    thresh=1e-4, verbose=False):
        """
        This method implement EM algorithm for high dimensions data.

        :param data_points: A sequence of tuples representing all data points assigned to the cluster.
        :param init_means: A sequence of tuples of mean coordinates for each cluster.
        :param init_covs: A sequence of ``numpy.ndarray``s of covariance for each cluster.
        :param init_weights: A sequence of weights for each cluster.
        :param cov_smoothing: A sequence of ``numpy.ndarray``s which specifies the default variance assigned to absent
            features in a cluster.
        :param max_iter: Maximum number of iterations.
        :param thresh: A digit indicating the threshold.
        :param verbose: Verbosity, True or False.
        :return: A dictionary that caches updated log-likelihood, means, covariances, weights and cluster responsibility
            matrix.
        """
        n = data_points.shape[0]
        dim = data_points.shape[1]

        mu = deepcopy(init_means)
        Sigma = deepcopy(init_covs)
        K = len(mu)
        init_weights = numpy.array(init_weights)
        resp = numpy.zeros((n, K))
        ll = None
        ll_trace = []

        for i in range(max_iter):
            # E-step: compute responsibilities
            logresp = numpy.zeros((n, K))
            for k in range(K):
                logresp[:, k] = numpy.log(init_weights[k]) + \
                                self.log_pdf_diagonal_gaussian(data_points, mu[k], Sigma[k])
            ll_new = numpy.sum(self._log_sum_exp(logresp, axis=1))

            if verbose:
                print(ll_new)

            logresp -= numpy.vstack(self._log_sum_exp(logresp, axis=1))
            resp = numpy.exp(logresp)
            counts = numpy.sum(resp, axis=0)

            # M-step: update weights, means, covariances
            init_weights = counts / numpy.sum(counts)
            for k in range(K):
                mu[k] = (self._diag(resp[:, k]).dot(data_points)).sum(axis=0) / counts[k]
                mu[k] = mu[k].A1

                Sigma[k] = self._diag(resp[:, k]).dot(
                    data_points.multiply(data_points) - 2 * data_points.dot(self._diag(mu[k]))).sum(axis=0) \
                           + (mu[k] ** 2) * counts[k]
                Sigma[k] = Sigma[k].A1 / counts[k] + cov_smoothing * numpy.ones(dim)

            # check for convergence in log-likelihood
            ll_trace.append(ll_new)
            if ll is not None and (ll_new - ll) < thresh and ll_new > -numpy.inf:
                ll = ll_new
                break
            else:
                ll = ll_new

        return {'weights': init_weights, 'means': mu, 'covs': Sigma, 'log-likelihood': ll, 'resp': resp}
