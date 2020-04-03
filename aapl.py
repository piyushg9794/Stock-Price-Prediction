# Importing libraries

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Loading dataset

df = pd.read_csv('AAPL.csv')
df = df.dropna()

training_set_unscaled = df.iloc[:-260, 4:5].values
test_set_unscaled = df.iloc[-260:, 4:5].values

print(training_set.shape)
print(test_set.shape)

# Scaling data

sc = MinMaxScaler(feature_range = (0, 1))
training_set = sc.fit_transform(training_set_unscaled)
test_set = sc.transform(test_set_unscaled)

# Visualization function

def final_plot(prediction, df):
   
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df.iloc[:, 4:5]['Close'], mode='lines', name='original'))
    fig.add_trace(go.Scatter(x=df.iloc[-200:]['Date'], y=prediction, mode='lines', name='predicted'))
    fig.show()
	
# Model 
window_size = 60

m = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences = True, input_shape= (window_size, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1),
])

m.compile(optimizer='adam', loss='mean_squared_error')
m.summary()

# Training method - 2 ( using tf.data.Dataset (recommended way) )

def make_window_dataset(df, window_size, shift, stride=1):
    
  window_size+=1
  ds = tf.data.Dataset.from_tensor_slices(df)
  ds = ds.window(window_size, shift=shift, stride=stride)
  
  def make_timestep(sub):
    return sub.batch(window_size, drop_remainder=True)

  def make_label(sub):
    return sub[:-1], sub[-1]

  ds = ds.flat_map(make_timestep) 
  win = ds.map(make_label).batch(32)

  return win

window_size = 60
train = make_window_dataset(training_set, window_size, 1)
test = make_window_dataset(test_set, window_size, 1)

m.fit(train, epochs=100)

# Predction using m-2

pred2 = m.predict(X_test)
pred2 = sc.inverse_transform(pred2)
pred2 = pred2.flatten()
final_plot(pred2, df)