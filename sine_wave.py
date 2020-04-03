import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

mi = 0
ma = 100
data_points = np.linspace(mi, ma, 1460)
dataset_un = np.sin(data_points)
plt.plot(dataset_un)

from sklearn.preprocessing import MinMaxScaler

dataset_un = dataset.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset_un)

n_steps=10
n_outputs=1

dataX, dataY = [], []
for i in range(10, 1200):
  x = dataset[i-10: i]
  y = dataset[i]
  dataX.append(x)
  dataY.append(y)
dataX, dataY =  np.array(dataX), np.array(dataY)
dataX = np.reshape(dataX, (dataX.shape[0], dataX.shape[1], 1))
print(dataX.shape)
print(dataY.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(1, activation='tanh', input_shape= (dataX.shape[1], 1)))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse') 

model.fit(dataX, dataY, epochs=200,verbose=2)

testX, testY = [], []

for i in range(1200, 1460):
  x = dataset[i-10: i]
  y = dataset[i]
  testX.append(x)
  testY.append(y)
testX, testY =  np.array(testX), np.array(testY)
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

pred = model.predict(testX, batch_size=16, verbose=2)
pred = scaler.inverse_transform(pred)

final = dataset[:1200]
final = list(final)
for i in pred:
  final.append(i)

plt.plot(final)