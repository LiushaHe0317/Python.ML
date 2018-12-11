
# Step 1: Data preparation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# input (dependent) and output (independent) variables
Inp = dataset.iloc[:, 3:13].values
Outp = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_I_1 = LabelEncoder()
Inp[:, 1] = labelencoder_I_1.fit_transform(Inp[:, 1])
labelencoder_I_2 = LabelEncoder()
Inp[:, 2] = labelencoder_I_2.fit_transform(Inp[:, 2])

# relates variables
onehotencoder = OneHotEncoder(categorical_features = [1])
Inp = onehotencoder.fit_transform(Inp).toarray()
Inp = Inp[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Inp_train, Inp_test, Outp_train, Outp_test = train_test_split(Inp, Outp, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Inp_train = sc.fit_transform(Inp_train)
Inp_test = sc.transform(Inp_test)

# step 2: Make the ANN

# importing Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialise the ANN
classifier = Sequential()

# addint input layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the ANN to training set
classifier.fit(Inp_train, Outp_train, batch_size = 10, nb_epoch = 100)

# step 3: making prediction and evaluating the model

# Predicting the Test set results
Outp_pred = classifier.predict(Inp_test)
Outp_pred = (Outp_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Outp_test, Outp_pred)


