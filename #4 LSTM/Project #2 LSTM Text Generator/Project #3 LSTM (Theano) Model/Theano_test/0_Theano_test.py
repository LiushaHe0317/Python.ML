
import theano
import theano.tensor as T
import numpy as np

c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')

m = A.dot(v)

matrix_times_vector = theano.function(inputs = (A,v), 
                                      outputs = m)

A_vol = np.array([[1,2],[3,4]])
v_vol = np.array([5,6])
m_vol = matrix_times_vector(A_vol, v_vol)

print(m_vol)

# train function
x = theano.shared(20.0, 'x')
cost = x*x + x + 1
x_updates = x - 0.3*T.grad(cost,x)
train = theano.function(inputs = [], outputs = cost,
                        updates = [(x, x_updates)])

# loop to call the train function
for i in range(25):
    cost_vol = train()
    print(i,cost_vol)
    
x.get_value()