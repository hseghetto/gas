# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 00:50:18 2020

@author: Humberto
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
gas.path = "/home/math1656/gas2/gas/"
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


gas.Parameters.flow_type = "DS"
gas.Parameters.model_type="RNN"
data = gas.read("gas_sd_extendido_1.txt")
data_og = data.copy()

transition_size = 0

# for reg2 in [int(sys.argv[1])]:
for layer_size in [30]:
    for epochs in [[100]]:
        for reg2 in [0.1]:
            for reg1 in [0]:
                for noise in [0,1,2]:
                    for batch_size in [[8]]:
                        result_runs = [[],[],[],[]]
                        for seed in range(100):
                            np.random.seed(seed)
                            gas.seed(seed)

                            data = gas.sample(data_og)
                            
                            train_start = 0
                            train_end = gas.getIndex(data,72)-gas.Parameters.last
                            
                            known_indexes = range(train_start,train_end,1)
                            known_indexes = list(known_indexes)
                            
                            percent = int(len(known_indexes)*0.7)
                            
                            prediction_start = gas.getIndex(data,72)-gas.Parameters.last
                            prediction_end = gas.getIndex(data,120)-gas.Parameters.last
                            
                            data = gas.calc_time_delta(data)
                            data = gas.calc_square_pressures(data)
                            #data = gas.calc_square_pressures(data)
                            
                            data = gas.gauss_noise(data, noise/100)
                            gas.Parameters.gauss_noise = noise
                            
                            data_stats = gas.stats(data[known_indexes])
                            data_norm = gas.standarize(data,data_stats)
                            
                            data_shaped,label = gas.preprocess(data)
                            data_shaped_norm,label_norm = gas.preprocess(data_norm)
                            
                            predict_data = data_shaped_norm[prediction_start:prediction_end]
                            predict_label=label[prediction_start:prediction_end]
                            
                            shape=(data_shaped.shape[1], data_shaped.shape[2])
                            
                            epochs1 = [0]
                            
                            np.random.shuffle(known_indexes)
                            train_indexes = known_indexes[0:percent]
                            val_indexes = known_indexes[percent:]
                            
                            train_data = data_shaped_norm[train_indexes]
                            train_label = label[train_indexes]
                            
                            val_data = data_shaped_norm[val_indexes]
                            val_label = label[val_indexes]
                            
                            gas.Parameters.Epochs = epochs
                            gas.Parameters.Epochs1 = epochs1
                            gas.Parameters.Batch_size = batch_size
                            gas.Parameters.transition_size = transition_size
                            gas.Parameters.train_percent = 0.7 
                            
                            reg = 0.005*reg2 # 0, 1 ,2 ,10
                            model = gas.rnn_network(layer_size,reg1,reg,shape)
                            
                            patience = np.max(epochs)
                            
                            print("Training")
                                                        
                            history = gas.train(epochs,batch_size,train_data,train_label,val_data,val_label,patience,model)
                            # history = gas.train(epochs1,batch_size,data_shaped_norm[0:200],label[0:200],patience,model)
                            
                            history = history.to_numpy()
                            train_error = history[-1,1:4]
                            
                            val_error = history[-1,5:8]
                            
                            # history2 = history2.to_numpy()
                            # trans_error = history2[-1,5:8]                            
                            
                            
                            print("Testing")
                            test_results, test_errors = gas.predict(data_shaped_norm[0:1],label[0:1],model,data_stats)
                            
                            prediction_results,prediction_errors = gas.predict(train_data,train_label,model ,data_stats)
                            print("Predicting")
                            prediction_results,prediction_errors = gas.predict(predict_data,predict_label,model ,data_stats)
                            trans_error = prediction_errors
                            
                            gas.saveRunResults(train_error,val_error,trans_error,test_errors,prediction_errors,prediction_results[-1])
                            
                            result_runs[0].append(np.mean(prediction_errors[0]))
                            result_runs[1].append(np.mean(prediction_errors[1]))
                            result_runs[2].append(np.mean(prediction_errors[2]))
                            result_runs[3].append(prediction_results[-1])
                            
                        gas.saveMeanResults(result_runs)

