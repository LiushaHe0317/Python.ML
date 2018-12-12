
# Part 1 data preparation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 time steps and 1 output
X_Train = []
Y_Train = []

for i in range(60,1258):
     X_Train.append(training_set_scaled[i-60:i,0])
     Y_Train.append(training_set_scaled[i,0])
X_Train, Y_Train = np.array(X_Train), np.array(Y_Train)

# reshaping
X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1],1))

# Part 2 Building RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_Train.shape[1],1)))
regressor.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_Train, Y_Train, epochs = 120, batch_size = 32)

# Part 3 Predictions and Visualisation

# real stock price for 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# predicted stock price
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
# scale inputs
inputs = sc.transform(inputs)

X_Test = []
for i in range(60,80):
     X_Test.append(inputs[i-60:i,0])
X_Test = np.array(X_Test)
# reshaping
X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1],1))

# model prediction
predicted_stock_price = regressor.predict(X_Test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualise the results
plt.plot(real_stock_price, color = 'black', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'gray', label = 'Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


















