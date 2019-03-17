
# packages and libraries
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import init_weight, all_parity_pairs
from sklearn.utils import shuffle

# Build the hidden layer
class HiddenLayer:
    
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        w = init_weight(M1, M2)
        b = np.zeros(M2)
        
        self.w = theano.shared(w, 'w_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.w, self.b]
        
    def forward(self, X):
        return T.nnet.relu(X.dot(self.w) + self.b)

# build the ANN
class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
    
    def fit(self, X, Y, learning_rate = 1e-2, mu = 0.99, reg = 1e-12,
            epochs = 400, batch_size = 20, print_period = 1, show_fig = False):
        
        Y = Y.astype(np.int32)
        
        # initialise the hidden layer
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        count = 0
        
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
            
        w = init_weight(M1, K)
        b = np.zeros(K)
        
        self.w = theano.shared(w, 'w_logreg')
        self.b = theano.shared(b, 'b_logreg')
        
        # collect params for later use
        self.params = [self.w, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        
        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        
        # for rmsprop
        cache = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]
        
        # set up theano functions and variables
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)
        
        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        prediction = self.predict(thX)
        grads = T.grad(cost, self.params)
            
        # momentum only
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]
            
        train_op = theano.function(
                inputs = [thX, thY], 
                outputs = [cost, prediction], 
                updates = updates
        )
        
        n_batches = N // batch_size
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_size:(j*batch_size + batch_size)]
                Ybatch = Y[j*batch_size:(j*batch_size + batch_size)]
                
                c, p = train_op(Xbatch, Ybatch)
                
                if j % print_period == 0:
                    costs.append(c)
                    err = np.mean(Ybatch != p)
                    print('i:', i, 'j:', j, 'nb:', n_batches, 'cost:', c, 'error rate:', err)
                    
        if show_fig:
            plt.plot(costs)
            plt.show()
            
            print(len(costs))

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.w) + self.b)
        
    def predict(self, X):
        pY = self.forward(X)
        return T.argmax(pY, axis = 1)
        
def wide():
    X, Y = all_parity_pairs(12)
    model = ANN([2048])
    model.fit(X, Y, learning_rate = 1e-4, print_period = 10, epochs = 300, show_fig = True)           
    
def deep():
    X, Y = all_parity_pairs(12)
    model = ANN([2048])
    model.fit(X, Y, learning_rate = 10e-3, print_period = 10, epochs = 100, show_fig = True)
             
if __name__ == '__main__':
        wide()
        # deep()
        