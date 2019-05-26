import json
import os
import numpy
import matplotlib.pyplot as plt
from utils import get_wikipedia_data, find_analogies


class Glove:
    def __init__(self, dimension, vocab_size, context_size, path_to_matrix=None):
        self.dim = dimension
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.fx = numpy.zeros([self.vocab_size, self.vocab_size])
        self.w = numpy.random.randn(self.vocab_size, self.dim) / numpy.sqrt(self.vocab_size + self.dim)
        self.U = numpy.random.randn(self.vocab_size, self.dim) / numpy.sqrt(self.vocab_size + self.dim)
        self.b = numpy.zeros(self.vocab_size)
        self.c = numpy.zeros(self.vocab_size)
        self.path_to_matrix = path_to_matrix

    def fit(self, sentences, learning_rate=10e-3, reg=0.1, xmax=100, alpha=0.85, epochs=10,
            verbose=0):

        cc_matrix = self._get_cc_matrx(self.path_to_matrix, sentences)
        self.fx[cc_matrix < xmax] = (cc_matrix[cc_matrix < xmax] / float(xmax))**alpha
        self.fx[cc_matrix >= xmax] = 1
        self.logx = numpy.log(cc_matrix + 1)
        self.mu = self.logx.mean()

        # gradient descent
        costs = []
        for epoch in range(epochs):
            delta = self.W.dot(self.U.T) + self.b.reshape(self.vocab_size,1) + self.c.reshape(1, self.vocab_size) + \
                    self.mu - self.logx
            cost = (self.fx*delta*delta).sum()

            if verbose > 0:
                print('training epoch {} of {} epochs; cost {}'.format(epoch, epochs, cost))

            costs.append(cost)

            oldW = self.w.copy()
            for i in range(self.vocab_size):
                self.w[i] -= learning_rate*(self.fx[i,:]*delta[i,:]).dot(self.U)
                self.b[i] -= learning_rate*self.fx[i,:].dot(delta[i,:])
                self.U[i] -= learning_rate*(self.fx[:,i]*delta[:,i]).dot(oldW)
                self.c[i] -= learning_rate*self.fx[:,i].dot(delta[:,i])
            self.w -= learning_rate * reg * self.w
            self.b -= learning_rate * reg * self.b
            self.U -= learning_rate * reg * self.U
            self.c -= learning_rate * reg * self.c

        plt.plot(costs)
        plt.show()

    def _get_cc_matrx(self, path_to_matrix, sentences):
        if not os.path.exists(path_to_matrix):
            thematrix = numpy.zeros([self.vocab_size, self.vocab_size])

            for sent in sentences:
                for i in range(len(sent)):
                    wi = sent[i]

                    start = max(0, i - self.context_size)
                    end = min(i + self.context_size, len(sent))

                    if i - self.context_size < 0:
                        points = 1.0/(i+1)
                        thematrix[wi,0] += points
                        thematrix[0,wi] += points
                    if i + self.context_size > len(sent):
                        points = 1.0/(n-i)
                        thematrix[wi,1] = points
                        thematrix[1,wi] = points

                        for j in range(start, i):
                            wj = sent[j]
                            points = 1.0/(i - j)
                            thematrix[wi,wj] = points
                            thematrix[wj,wi] = points

                        for j in range(i+1, end):
                            wj = sent[j]
                            points = 1.0/(j-i)
                            thematrix[wi, wj] = points
                            thematrix[wj, wi] = points

            numpy.save(path_to_matrix, thematrix)
        else:
            thematrix = numpy.load(path_to_matrix)

        return thematrix

    def save(self, fn):
        arr = [self.w, self.U.T]
        numpy.save(fn, *arr)

def main(we_file, w2i_file, n_files=100):
    cc_matrix = "cc_matrix_%s.npy" % n_files

    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)
        sentences = []  # dummy - we won't actually use it
    else:
        sentences, word2idx = get_wikipedia_data(n_files=n_files, n_vocab=2000)
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    # model initialization
    vocab_size = len(word2idx)
    model = Glove(100, vocab_size, 10, cc_matrix=cc_matrix)

    # train the model
    model.fit(sentences, epochs=20)
    model.save(we_file)

if __name__=='__main__':

    we_path = 'glove_model_50.npz'
    w2i_path = 'glove_word2idx_50.json'
    main(we__path, w2i_path)

    # load back embeddings
    npz = numpy.load(we)

    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i: w for w, i in word2idx.items()}

    for concat in (True, False):
        print("** concat:", concat)

        if concat:
            We = numpy.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2

        # find analogies between words
        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)