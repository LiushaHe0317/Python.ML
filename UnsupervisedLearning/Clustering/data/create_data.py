import numpy


class DataCreator:
    def generate_gmm_data(self, means, covariances, weights, n_of_data):
        """
        THis method Creates a list of data points.
        """
        data = []
        for i in range(n_of_data):
            k = numpy.random.choice(len(weights), 1, p=weights)[0]
            x = numpy.random.multivariate_normal(means[k], covariances[k])
            data.append(x)

        return data