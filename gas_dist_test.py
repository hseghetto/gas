# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:51:36 2020

@author: hsegh
"""
import sys
import os

import tensorflow as tf
from tensorflow import keras

import gas

gas.path = os.getcwd()

data_og = gas.read("gas_im_extendido_1.txt")
data = gas.read("gas_im_extendido_1.txt")

data = data[0:100] 
data[0][-1]=0

data = gas.sample(data)
data = gas.delta_times(data,True)
data = gas.square_pressures(data,False)

data = gas.gauss_noise(data, 0.01)
data_standarized = gas.standarize(data)

data_shaped,label=gas.preprocess(data_standarized)


"=============================================================================================="
reg1=0.00
reg2=0.01
layer_size = 16
BATCH_SIZE = 16
EPOCHS = 100

model = keras.Sequential()
model.add(keras.layers.GRU(layer_size, kernel_regularizer=keras.regularizers.l1_l2(reg1,reg2),
                           input_shape=(data_shaped.shape[1], data_shaped.shape[2])),
                           bias_regularizer=keras.regularizers.l1_l2(reg1,reg2))
#model.add(keras.layers.Flatten(input_shape=(train_data_norm.shape[1], train_data_norm.shape[2])))
#model.add(keras.layers.Dense(layer_size,activation="tanh"))
#model.add(keras.layers.Dense(layer_size,activation="tanh"))
model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l2(reg2),
                 activity_regularizer=keras.regularizers.l1(reg1)))

model.compile(optimizer='adam',
              loss=keras.losses.mse,
              metrics=['mae','mse','mape'])

"=============================================================================================="
