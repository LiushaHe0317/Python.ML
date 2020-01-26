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
        This method trains a hidden Markov model.

        :param X:
        :param max_iter:
        :param seed:
        :param verbose:
        """
        # set default
        numpy.random.seed(seed)   # seed
        self.V = max(max(x) for x in X) + 1    # calculate the given the vocabulary size
        N = len(X)    # calculate the given number of sequences
        self.pi = numpy.ones(self.M) / self.M    # initial state distribution
        self.A = self._random_initialized(self.M, self.M)   # state transition matrix
        self.B = self._random_initialized(self.M, self.V)    # output distribution

        print(f"initial A: {self.A}")
        print(f"initial B: {self.B}")
        print(f"initial pi: {self.pi}")

        costs = []
        for it in range(max_iter):
            if it % 10 == 0:
                print(f"iteration: {it}")

            forwards = self.compute_alphas_and_betas(X, N, self.A, self.B, self.pi)
            alphas = forwards['alphas']
            betas = forwards['betas']
            P = forwards['P']
            self.A = forwards['A']
            self.B = forwards['B']
            self.pi = forwards['pi']

            assert (numpy.all(P > 0))

            # calculate cost
            cost = numpy.sum(numpy.log(P))
            costs.append(cost)

            # now re-estimate pi, A, B
            self.pi, self.A, self.B = self.reestimate(X, N, alphas, betas, self.A, self.B, P)

        if verbose:
            print(f"pi after training: {self.pi}")
            print(f"A after training: {self.A}")
            print(f"B after training: {self.B}")

        plt.figure()
        plt.plot(costs)
        plt.ylabel('costs')
        plt.xlabel('# of iteration')
        plt.savefig('result.png')

    def compute_alphas_and_betas(self, X, N, A, B, pi):
        """


        :param X:
        :param N:
        :param A:
        :param B:
        :param pi:
        :return:
        """
        forwards = {}

        alphas = []
        betas = []
        P = numpy.zeros(N)
        for n in range(N):
            x = X[n]
            T = len(x)

            # calculate alpha: forward
            alpha = numpy.zeros((T, self.M))
            alpha[0] = pi * B[:, x[0]]
            for t in range(1, T):
                tmp1 = alpha[t - 1].dot(A) * B[:, x[t]]
                alpha[t] = tmp1

            P[n] = numpy.sum(alpha[-1])
            alphas.append(alpha)

            # calculate beta: backward
            beta = numpy.zeros((T, self.M))
            beta[-1] = 1
            for t in range(T - 2, -1, -1):
                beta[t] = A.dot(B[:, x[t + 1]] * beta[t + 1])
            betas.append(beta)

        forwards['alphas'] = alphas
        forwards['betas'] = betas
        forwards['P'] = P
        forwards['A'] = A
        forwards['B'] = B
        forwards['pi'] = pi

        return forwards

    def reestimate(self, X, N, alphas, betas, A, B, P):
        """
        This method re-estimate pi, A, and B.

        :param X:
        :param N:
        :param alphas:
        :param betas:
        :param A:
        :param B:
        :param P:
        :return:
        """
        pi = numpy.sum((alphas[n][0] + betas[n][0]) / P[n] for n in range(N)) / N
        den1 = numpy.zeros((self.M, 1))
        den2 = numpy.zeros((self.M, 1))
        a_num, b_num = 0, 0

        for n in range(N):
            x = X[n]
            T = len(x)

            den1 += numpy.sum((alphas[n][:-1] * betas[n][:-1]), axis=0, keepdims=True).T / P[n]
            den2 += numpy.sum((alphas[n] * betas[n]), axis=0, keepdims=True).T / P[n]

            a_num_n = numpy.zeros((self.M, self.M))
            for i in range(self.M):
                for j in range(self.M):
                    for t in range(T - 1):
                        a_num_n[i, j] += alphas[n][t, i] * A[i, j] * B[j, x[t + 1]] * betas[n][t + 1, j]
            a_num += a_num_n / P[n]

            b_num_n = numpy.zeros((self.M, self.V))
            for i in range(self.M):
                for j in range(self.V):
                    for t in range(T):
                        if x[t] == j:
                            b_num_n[i, j] += alphas[n][t, i] * betas[n][t, i]
            b_num += b_num_n / P[n]

        A, B = a_num / den1, b_num / den2

        return pi, A, B

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

    def get_state_sequence(self, x):
        """
        This method get the most likely state sequence.

        :param x: A list of observation.
        :return: A list of integers which indicates the sequence of states.
        """
        T = len(x)
        delta = numpy.zeros((T, self.M))
        psi = numpy.zeros((T, self.M))
        delta[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = numpy.max(delta[t-1]*self.A[:,j])*self.B[j,x[t]]
                psi[t,j] = numpy.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = numpy.zeros(T, dtype=numpy.int32)
        states[T-1] = numpy.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def _random_initialized(self, d1, d2):
        """
        A helper function for a 2D matrix random initialization.

        :param d1: An integer which indicates the size of dimension 1.
        :param d2: An integer which indicates the size of dimension 2.
        :return: A 2D matrix.
        """
        x = numpy.random.random((d1, d2))
        return x / numpy.sum(x, axis=1, keepdims=True)


if __name__=="__main__":
    data_file = r"../data/coin_data.txt"
    X = []
    for line in open(data_file):
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    hmm = HMM(2)
    hmm.fit(X, seed=123, verbose=2)

    print(f"LL with fitted parameters: {numpy.sum(hmm.log_likelihood_multi(X))}")
    print(f"best state sequence for: {numpy.array(X[0])}")
    print(hmm.get_state_sequence(X[0]))
