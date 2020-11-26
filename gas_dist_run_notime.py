# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:40:21 2020

@author: Humberto
"""

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

# pip install -e <os.getcwd>/gas_dist
# gas.path = os.getcwd()
gas.Parameters.initial_pressure=300
gas.Parameters.last=2

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

gas.Parameters.flow_type="IM"
gas.Parameters.model_type="RNN"
data = gas.read("gas_im_extendido_1.txt")
data_og = data.copy()

 #to be deleted 
#data[0][-1]=0      #to be deleted

data = gas.sample(data)

train_start = 0
train_end = gas.getIndex(data,60)

val_start = gas.getIndex(data,60)
val_end = gas.getIndex(data,72)

indexes = gas.getIndex(data[0:val_end])

prediction_start = gas.getIndex(data,72)
prediction_end = gas.getIndex(data,120)

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

predict_data = data_shaped_norm[prediction_start:prediction_end]
predict_label=label[prediction_start:prediction_end]

layer_size= 16
reg1 = 0.0
# reg2 = 0.0
shape=(data_shaped.shape[1], data_shaped.shape[2]-1)


# for reg2 in [int(sys.argv[1])]:
for reg2 in [0,2]:
    for layer_size in [16]:
        for epochs in [[150],[200],[250]]:
            for epochs1 in [[0]]:
                for transition_size in [50]:
                    for batch_size in [[32]]:
                        result_runs = [[],[],[],[]]
                        for seed in range(5):
                            
                            gas.Parameters.Epochs = epochs
                            gas.Parameters.Epochs1 = epochs1
                            gas.Parameters.Batch_size = batch_size
                            gas.Parameters.transition_size = transition_size
                            
                            np.random.seed(seed)
                            reg = 0.0005*reg2 # 0, 1 ,2 ,10
                            # model = gas.rnn_network(layer_size,reg1,reg,shape)
                            model = gas.rnn_network(layer_size,reg1,reg,shape)                            
                            
                            patience = np.max(epochs)
                            
                            print("Training")
                            # transitions = np.vstack([data_shaped_norm[0:4406],data_shaped_norm[0:200]])
                            transitions = list(range(0,transition_size))
                            for x in indexes:
                                aux = list(range(x,x+transition_size))
                                transitions.extend(aux)
                            
                            history = gas.train(epochs,batch_size,train_data[:,:,1:],train_label,val_data[:,:,1:],val_label,patience,model)
                            history2 = gas.train(epochs1,batch_size,train_data[transitions,:,1:],train_label[transitions],val_data[:,:,1:],val_label,patience,model)
                            # history = gas.train(epochs1,batch_size,data_shaped_norm[0:200],label[0:200],patience,model)
                            
                            history = history.to_numpy()
                            train_error = history[-1,1:4]
                            
                            history2 = history2.to_numpy()
                            trans_error = history[-1,1:4]
                            
                            print("Testing")
                            test_results, test_errors = gas.predict(data_shaped_norm[0:4406,:,1:],label[0:4406],model,data_stats)
                            
                            print("Predicting")
                            prediction_results,prediction_errors = gas.predict(predict_data[:,:,1:],predict_label,model ,data_stats)
                            
                            gas.saveRunResults(test_errors,prediction_errors,prediction_results[-1])
                            
                            result_runs[0].append(np.mean(prediction_errors[0]))
                            result_runs[1].append(np.mean(prediction_errors[1]))
                            result_runs[2].append(np.mean(prediction_errors[2]))
                            result_runs[3].append(prediction_results[-1])
                            
                            
                        gas.saveMeanResults(result_runs)
