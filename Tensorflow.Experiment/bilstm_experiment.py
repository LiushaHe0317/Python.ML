from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional
import numpy

T = 8
D = 2
M = 3

X = numpy.random.randn(1, T, D)

_input = Input(shape=(T,D))
_output = Bidirectional(LSTM(M, return_sequences=False, return_state=True))(_input)
model = Model(inputs=_input, outputs=_output)

o, h1, c1, h2, c2 = model.predict(X)

print('o: ', o.shape)
print('h1: ', h1.shape)
print('c1: ', c1.shape)
print('h2: ', h2.shape)
print('c2: ', c2.shape)