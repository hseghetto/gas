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

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import time as tm

import gas

# pip install -e os.getcwd gas_dist
gas.path = os.getcwd()
gas.initial_pressure=300
gas.last=2

# gas.tr1=0
# gas.tr2=0
# gas.noise1=0.01
# gas.noise2=0
# gas.train_percent = 0 
# gas.layer_size = 16
# gas.reg1 = 0.
# gas.reg2 = 0.
# gas.first_pred_time = 0
# gas.time_delta=True
# gas.sqrp = False

gas.flow_type="IM"
data = gas.read("gas_im_extendido_1.txt")
data_og = data.copy()

data = data[0:8000] #to be deleted 
#data[0][-1]=0      #to be deleted

data = gas.sample(data)

train_start = 0
train_end = gas.getIndex(data,64)

val_start = gas.getIndex(data,64)
val_end = gas.getIndex(data,72)

indexes = gas.getIndex(data[0:val_end])

data = gas.calc_time_delta(data)
# data = gas.calc_square_pressures(data)

#data = gas.gauss_noise(data, 0.05)

data_stats = gas.stats(data)
data_norm = gas.standarize(data,data_stats)

data_shaped,label = gas.preprocess(data)
data_shaped_norm,label_norm = gas.preprocess(data_norm)

train_data = data_shaped_norm[train_start:train_end]
train_label = label[train_start:train_end]

val_data = data_shaped_norm[val_start:val_end]
val_label = label[val_start:val_end]

layer_size = 16
reg1 = 0.
reg2 = 0.
shape=(data_shaped.shape[1], data_shaped.shape[2])

model = gas.rnn_network(layer_size,reg1,reg2,shape)

epochs = [200]
epochs1=[50]
batch_size = [32]
patience = np.max(epochs)

print("Training")
# transitions = np.vstack([data_shaped_norm[0:4406],data_shaped_norm[0:200]])

transition_size=50
transitions = list(range(0,transition_size))
for x in indexes:
    aux = list(range(x,x+transition_size))
    transitions.extend(aux)
    
history = gas.train(epochs,batch_size,train_data,train_label,val_data,val_label,patience,model)
history2 = gas.train(epochs1,batch_size,train_data[transitions],train_label[transitions],val_data,val_label,patience,model)
# history = gas.train(epochs1,batch_size,data_shaped_norm[0:200],label[0:200],patience,model)

print("Testing")
test_results, test_errors = gas.test(data_shaped_norm[0:4406],label[0:4406],model)

print("Predicting")
prediction_results,prediction_errors = gas.predict(data_shaped_norm[0:4406],label[0:4406],model ,data_stats)

gas.saveRunResults(test_errors,prediction_errors,prediction_results[-1])

gas.plotResults(prediction_results,label[0:4406],data_shaped[0:4406])
prediction_results,prediction_errors = gas.predict(data_shaped_norm[4406:],label[4406:],model ,data_stats)
gas.plotResults(prediction_results,label[4406:],data_shaped[4406:])

