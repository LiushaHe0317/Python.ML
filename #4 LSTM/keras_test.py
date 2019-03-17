
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be
# no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

print(input_array.shape)

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)

## plot matrix
f = plt.figure()
#f.add_subplot(121)
plt.imshow(input_array)
plt.title('Input')
plt.show()

#f.add_subplot(122)
#plt.imshow(output_array)
#plt.title('Output')
#plt.show()

assert output_array.shape == (32, 10, 64)
