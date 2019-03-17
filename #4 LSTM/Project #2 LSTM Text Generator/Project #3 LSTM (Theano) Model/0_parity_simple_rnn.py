
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def all_parity_pairs_with_sequence_labels(nbit):
    # total number of samples (Ntotal) will be a multiple of 100
    # why did I make it this way? I don't remember.
    N = 2**nbit
    Ntotal = N + 100 - (N % 100)
    
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    
    for ii in range(Ntotal):
        i = ii % N
        
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
    Y[ii] = X[ii].sum() % 2
    
    N, t = X.shape

    # we want every time step to have a label
    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n,i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n,i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)
    return X, Y_t

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

class SimpleRNN:
    def __init__(self, M):
        self.M = M
    
    def train(self, 
              X, Y, learning_rate = 0.1, mu = 0.99, reg = 1.0,
              activation = T.tanh, epochs = 100, show_fig = False):
        
        D = X[0].shape[1] # X is of size N x T(n) x D
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        
        self.f = activation

        # initial weights
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        Wo = init_weight(M, K)
        
        bh = np.zeros(M)
        h0 = np.zeros(M)
        bo = np.zeros(K)

        # make them theano shared
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        
        ## the recurrence function #1
        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            
            print(x_t)
            print(x_t.dot(self.Wx))
            
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps=thX.shape[0],
            )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        ## updates model weights
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
         ] + [
             (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
             ]
        
        self.predict_op = theano.function(
            inputs=[thX], outputs=prediction,
            )
        self.train_op = theano.function(
            inputs=[thX, thY],outputs=[cost, prediction, y],updates=updates,
        )

        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(N):
                c, p, rout = self.train_op(X[j], Y[j])
                # print "p:", p
                cost += c
                if p[-1] == Y[j,-1]:
                    n_correct += 1
            
            print("shape y:", rout.shape)
            print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
            
            costs.append(cost)
            #if n_correct == N:
            #    break

        if show_fig:
            plt.plot(costs)
            plt.show()

X, Y = all_parity_pairs_with_sequence_labels(12)

def solve_parity_problem(
        X = X, Y = Y, learning_rate = 1e-4, epochs = 200):
    
    
    
    rnn_model = SimpleRNN(4)
    rnn_model.train(X, Y, learning_rate = learning_rate, epochs = epochs,
                  activation = T.nnet.relu, show_fig = True)
    
if __name__ == '__main__':
    solve_parity_problem()
