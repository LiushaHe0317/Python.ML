import numpy
import matplotlib.pyplot as plt


class HMM:
    """
    This object implements a hidden Markov model.

    :param M: An integer indicating the number of states.
    """
    def __init__(self, M):
        self.M = M

    def fit(self, X, max_iter=30, seed=42, verbose=0):
        """
        The method which trains a hidden Markov model.

        :param X:
        :param max_iter: An integer which defines how many iterations expectation maximization will do.
        :return:
        """
        numpy.random.seed(seed)

        # calculate the given the vocabulary size
        V = max(max(x) for x in X) + 1

        # calculate the given number of sequences
        N = len(X)

        # state initialization
        self.pi = numpy.ones(self.M) / self.M
        self.A = self._random_initialized(self.M, self.M)
        self.B = self._random_initialized(self.M, V)

        costs = []
        for it in range(max_iter):
            if it % 10 == 0:
                print(f"iteration: {it}")
                alphas = []
                betas = []
                P = numpy.zeros(N)

                for n in range(N):
                    x = X[n]
                    T = len(x)

                    # calculate alpha: forward
                    alpha = numpy.zeros((T, self.M))
                    alpha[0] = self.pi * self.B[:, x[0]]
                    for t in range(1, T):
                        alpha[t] = numpy.dot(alpha[t-1], self.A) * self.B[:, x[t]]
                    P[n] = numpy.sum(alpha[-1])
                    alphas.append(alpha)

                    # calculate beta: backward
                    beta = numpy.zeros((T, self.M))
                    beta[-1] = 1
                    for t in range(T-2, -1, -1):
                        beta[t] = numpy.dot(self.A, self.B[:, x[t+1]])
                    betas.append(beta)

                # calculate cost
                cost = numpy.sum(numpy.log(P))
                costs.append(cost)

                # update pi
                self.pi = numpy.sum((alphas[n][0] + betas[n][0])/P[n] for n in range(N)) / N

                den1 = numpy.zeros((self.M, 1))
                den2 = numpy.zeros((self.M, 1))

                a_num = 0
                b_num = 0

                for n in range(N):
                    x = X[n]
                    T = len(x)

                    den1 += numpy.sum(alphas[n][:-1] * betas[n][:-1], axis=0, keepdims=True).T / P[n]
                    den2 += numpy.sum(alphas[n] * betas[n], axis=0, keepdims=True).T / P[n]

                    a_num_n = numpy.zeros((self.M, self.M))
                    for i in range(self.M):
                        for j in range(self.M):
                            for t in range(T-1):
                                a_num_n[i,j] += alphas[n][t,i]*self.A[i,j] * self.B[j, x[t+1]] * betas[n][t+1,j]
                    a_num += a_num_n / P[n]

                    b_num_n = numpy.zeros((self.M, V))
                    for i in range(self.M):
                        for j in range(V):
                            for t in range(T):
                                if x[t] == j:
                                    b_num_n[i, j] = alphas[n][t, i] * betas[n][t,i]
                    b_num += b_num_n / P[n]
                self.A = a_num / den1
                self.B = b_num / den2

        if verbose:
            print(f"pi: {self.pi}")
            print(f"A: {self.A}")
            print(f"B: {self.B}")

            plt.plot(costs)
            plt.show()

    def likelihood(self, x):
        """
        This method calculates the likelihood for a single observation.

        :param x:
        :return:
        """
        T = len(x)
        alpha = numpy.zeros((T, self.M))
        alpha[0] = self.pi * self.B[:, x[0]]
        for t in range(1,T):
            alpha[t] = numpy.dot(alpha[t-1], self.A) * self.B[:, x[t]]

        return numpy.sum(alpha[-1])

    def likelihood_multi(self, X):
        """

        :param X:
        :return:
        """
        return numpy.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        return numpy.log(self.likelihood_multi(X))

    def get_stats_sequence(self, x):
        """
        

        :param x:
        :return:
        """
        T = len(x)
        delta = numpy.zeros((T, self.M))
        psi = numpy.zeros((T, self.M))


    def _random_initialized(self, d1, d2):
        """
        A helper function for a 2D matrix random initialization.

        :param d1: An integer which indicates the size of dimension 1.
        :param d2: An integer which indicates the size of dimension 2.
        :return: A 2D matrix.
        """
        x = numpy.random.random((d1, d2))
        return x / numpy.sum(x, axis=1, keepdims=True)
