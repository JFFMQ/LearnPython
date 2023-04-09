# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:25:59 2023

@author: gamem
"""

import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense
from pprint import pprint
from pylab import plt, mpl


plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
#pd.set_option('precision', 4)
pd.options.display.precision=4
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'

def set_seeds(seed=100): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seeds()

a = np.arange(100)
a = a.reshape(len(a),-1)

lags = 3
g = TimeseriesGenerator(a,a,length=lags,batch_size=5)
pprint(list(g)[0])

model = Sequential()
model.add(SimpleRNN(100, activation='relu',input_shape=(lags, 1))) 
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adagrad', loss='mse',metrics=['mae'])

model.summary()

%time model.fit(g, epochs=1000, steps_per_epoch=5,verbose=False)

#The performance metrics might show relatively erratic behavior when training RNNs
res = pd.DataFrame(model.history.history)
res.tail(3)

res.iloc[10:].plot(figsize=(10, 6), style=['--', '--'])

#generates in-sample and out-of-sample predictions
x = np.array([21, 22, 23]).reshape((1, lags, 1))
y = model.predict(x, verbose=False) 
print(int(round(y[0, 0])))

x = np.array([87, 88, 89]).reshape((1, lags, 1))
y = model.predict(x, verbose=False) 
print(int(round(y[0, 0])))

x = np.array([187, 188, 189]).reshape((1, lags, 1))
y = model.predict(x, verbose=False) 
print(int(round(y[0, 0])))

x = np.array([1187, 1188, 1189]).reshape((1, lags, 1))
y = model.predict(x, verbose=False) 
print(int(round(y[0, 0])))