
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
Y = data.target

## categorical labelling
# class 0 --> [1 0 0]
# class 1 --> [0 1 0]
# class 2 --> [0 0 1]
from keras.utils import to_categorical

Y = to_categorical(Y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size = 0.33, random_state = 42)

## data scaling
from sklearn.preprocessing import MinMaxScaler
myscaler = MinMaxScaler()
myscaler.fit(x_train)

scaled_x_train = myscaler.transform(x_train)
scaled_x_test = myscaler.transform(x_test)

## build up the network model
# simple model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
## add layers
model.add(Dense(
        units = 8, input_dim = 4, activation = 'relu',
                ))
model.add(Dense(
        units = 8, input_dim = 4, activation = 'relu',
                ))
model.add(Dense(
        units = 3,activation = 'softmax', 
        ))

## compile the model
model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy'])
## check the model
model.summary()

## training
model.fit(
        scaled_x_train, y_train, epochs = 350, verbose = 2,
        )

## prediction
pr = model.predict_classes(scaled_x_test)
actual = y_test.argmax(axis = 1)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(actual, pr))
print('\n')
print(classification_report(actual, pr))
print('\n')
print(accuracy_score(actual, pr))

## save the model
model.save('iris_model_1.h5')