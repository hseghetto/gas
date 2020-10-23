# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:51:36 2020

@author: hsegh
"""
import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import gas

gas.path = os.getcwd()

data_og = gas.read("gas_im_extendido_1.txt")
data = gas.read("gas_im_extendido_1.txt")

data = data[0:8000] 
#data[0][-1]=0

data = gas.sample(data)
data = gas.delta_times(data,True)
data = gas.square_pressures(data,False)

data = gas.gauss_noise(data, 0.00)

data_stats = gas.stats(data)
data_norm = gas.standarize(data,data_stats)

data_shaped,label = gas.preprocess(data)
data_shaped_norm,label_norm = gas.preprocess(data_norm)

reg1=0.00
reg2=0.00
layer_size = 16
shape=(data_shaped.shape[1], data_shaped.shape[2])

model = gas.rnn_network(layer_size, reg1, reg2, shape)

epochs = [1000]
batch_size = [32]
patience = np.max(epochs)
print("Training")
history = gas.train(epochs,batch_size,data_shaped_norm,label,patience,model)

print("Testing")
test_results, test_errors = gas.test(data_shaped_norm,label,model)

print("Predicting")
prediction_results,prediction_errors = gas.predict(data_shaped_norm  ,label,model,data_stats)

gas.save()