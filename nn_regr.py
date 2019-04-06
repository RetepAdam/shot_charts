from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import pandas as pd
from sklearn.model_selection import train_test_split
import theano
import cv2
import os

#Variables
df = pd.read_csv('2016-17_advanced.csv')
df = df[df['Yr'] == 2017]
df = df[df['MP'] >= 1000]
y = np.array(df['Yr'].values)
pid = np.array(df['Player_ID'].values.astype('int'))
images = []
for i in range(len(pid)):
   images.append(cv2.imread('images/{0}_2016-17.png'.format(pid[i]), 0).flatten())
X = np.array(images)
X = np.array(X)
X = np.array(df['Yr'].values)
y = np.array(y).reshape(-1, 1)
# scaler = MinMaxScaler()
# print(scaler.fit(y))
# y = scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

theano.config.floatX = 'float32'
y_train = y_train.astype(theano.config.floatX)
y_test = y_test.astype(theano.config.floatX)
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

model = Sequential()
model.add(Dense(units=1, input_dim=1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=5,  verbose=1, validation_split=0.2)
