# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:51:22 2020

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

path = ""
last = 2
pfactor=1
initial_pressure = 300
x=1


def read(file):
    a=pd.read_csv(path+"data/"+file,sep=" ")
    a=a.to_numpy()
    return a

def sample(data,tr=0,tr2=0):
    r=[]
    
    r.append(data[0])
    for i in range(len(data)-1):
        if(np.abs(data[i][1]-r[-1][1])>tr or np.abs(data[i][1]-data[i+1][1])>tr or np.abs(data[i][0]-r[-1][0])>tr2):
            r.append(data[i])
    r=np.array(r)
    return r

def delta_times(data,aux=True):
    if(aux):
        #for i in range(1,len(data)):
        for i in range(len(data)-1,0,-1):
            data[i][0]=data[i][0]-data[i-1][0]
    return data

def square_pressures(data,aux=False):
    if(aux):
        global pfactor
        pfactor = initial_pressure
        for i in range(len(data)):
            data[i][1]=data[i][1]**2
    return data

def stats(data):
    data_df=pd.DataFrame(data)
    data_df=data_df.describe()
    data_stats=data_df.to_numpy()
    return data_stats

def standarize(data,data_stats):
    data = np.copy(data)
    for i in range(0,len(data[0])):
        data[:,i]=(data[:,i]-data_stats[1,i])/data_stats[2,i] #standarization
    return data
        
def preprocess(data): #returns the arrays to be fed the model
    #When using an RNN data must be fed with an {N, timestep, features} shape
    #In our case this means N is the number of datapoints that can be used to predict (we cant predict if there are not at least L past datapoints)
    #timestep is the number of L past points used for each prediction
    #we used [timestamp, past pressure, flow rate] for each timestep giving us features=3
    input_data=[]
    label=[]
    for i in range(0,data.shape[0]-last-1):
        input_data.append([])
        for j in range(last):
            input_data[-1].append([])
            input_data[-1][-1].append(data[i+j+1][0]) #appending time
            input_data[-1][-1].append(data[i+j,1]) #appending pressure
            input_data[-1][-1].append(data[i+j+1,2]) #appending flow rate

        label.append(data[i+last,1]/pfactor) #appending pressure target

    input_data=np.array(input_data)
    label=np.array(label)

    return input_data,label

def gauss_noise(data,sigma): #white noise
    for i in range(len(data)):
        noise=np.random.normal(0,sigma,1)
        data[i][1]=data[i][1]*(1+noise)
    return data

def feedfoward_network(layer_size,reg1,reg2,shape):
    model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape=shape))
    model.add(keras.layers.Dense(layer_size,activation="tanh",kernel_regularizer=keras.regularizers.l1_l2(reg1,reg2),
                           bias_regularizer=keras.regularizers.l1_l2(reg1,reg2)))
    model.add(keras.layers.Dense(layer_size,activation="tanh",kernel_regularizer=keras.regularizers.l1_l2(reg1,reg2),
                           bias_regularizer=keras.regularizers.l1_l2(reg1,reg2)))
    model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l1_l2(reg1,reg2),
                           bias_regularizer=keras.regularizers.l1_l2(reg1,reg2)))

    model.compile(optimizer='adam',
              loss=keras.losses.mse,
              metrics=['mae','mse','mape'])
    
    return model
    
def rnn_network(layer_size,reg1,reg2,shape):
    model = keras.Sequential()
    model.add(keras.layers.GRU(layer_size, input_shape=shape,
                           kernel_regularizer=keras.regularizers.l1_l2(reg1,reg2),
                           bias_regularizer=keras.regularizers.l1_l2(reg1,reg2)))
    model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l1_l2(reg1,reg2),
                           bias_regularizer=keras.regularizers.l1_l2(reg1,reg2)))

    model.compile(optimizer='adam',
              loss=keras.losses.mse,
              metrics=['mae','mse','mape'])
    
    return model

def train(Epochs,Batch_size,train_data_shaped_norm,train_label,val_shaped_norm,val_label,Patience,model):
    if(len(val_label)):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=Patience) #CHANGE TO VAl_MSE
    else:
        early_stop = keras.callbacks.EarlyStopping(monitor='mse', patience=Patience)
    h=0
    
    for i in range(len(Epochs)):
       
        history = model.fit(train_data_shaped_norm, train_label, epochs=Epochs[i],
                            validation_data=(val_shaped_norm,val_label),
                            verbose=0, callbacks=[early_stop], batch_size=Batch_size[i])

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        
        if(type(h) == type(int(1))):
            h=hist
        else:
            h=pd.concat([h,hist])
        
    return h

def test(data_shaped_norm,label,model):
    prediction = model.predict(data_shaped_norm)
    
    mse=np.zeros(len(prediction))
    mae=np.zeros(len(prediction))
    mape=np.zeros(len(prediction))

    for i in range(0,len(prediction)):
        mse[i]=np.square(prediction[i]-label[i])
        mae[i]=np.abs(prediction[i]-label[i])
        mape[i]=mae[i]/label[i]*100
        
    errors = np.transpose(np.vstack((mse,mae,mape)))
    return prediction,errors

def predict(data_shaped_norm,label,model,data_stats):
    global pfactor
    size = len(label)
    prediction = np.zeros(size)
    
    for i in range(size):
        #print(np.array([data_shaped_norm[i]]).shape)
        prediction[i] = model.predict(np.array([data_shaped_norm[i]]))[0][0]
        #print(prediction[i])
        norm = (prediction[i]*pfactor- data_stats[1,1])/data_stats[2,1]
        for j in range(min(last,size-i)):
            data_shaped_norm[i+j][-j][1] = norm
        
    mse=np.zeros(size)
    mae=np.zeros(size)
    mape=np.zeros(size)

    for i in range(0,size):
        mse[i]=np.square(prediction[i]-label[i])
        mae[i]=np.abs(prediction[i]-label[i])
        mape[i]=mae[i]/label[i]*100
        
    errors = np.transpose(np.vstack((mse,mae,mape)))
    return prediction,errors

def getIndex(series,t=0):
    if(t==0):
        r = []
        for i in range(1,len(series)):
            if(series[i][-1] != series[i-1][-1]):
                r.append(i)
        return r
    for i in range(len(series)):
        if(series[i][0]>t):
            return i-1
        

def save():
    p=";"+"IM" #Flow Type
    p+=";"+str(first_pred_time) #Predicting from
    p+=";"+str(tr1)
    p+=";"+str(tr2)
    p+=";"+str(noise1)
    p+=";"+str(noise2)
    p+=";"+str(train_percent)
    p+=";"+str(layer_size)
    p+=";"+str(reg1)
    p+=";"+str(reg2)
    p+=";"+str([x for x in Epochs])
    p+=";"+str([x for x in Batch_size])
    p+=";"+str(int(sqrp))
    p+=";"+str(int(time_delta))
    p+=";"+str(last)
    
    s = ";"
    for x in errors_testing:
        s += ";"+str(np.mean(x))
    for x in errors_prediction:
        s += ";"+str(np.mean(x))
    
    s = s + p + "\n"
    
    with open(path+"results/"+str(hash(p))+".txt",'a') as arq:
        arq.write(s)
    return 1

def saveMeanResults(result_runs,tr1,tr2,noise1,noise2,train_percent,layer_size,reg1,reg2,
         Epochs,Batch_size,first_pred_time,time_delta,sqrp):

    s=";"
    s+=str(np.mean(result_runs[0])) #MAE
    s+=";"+str(np.mean(result_runs[1])) #MSE
    s+=";"+str(np.mean(result_runs[2])) #MAPE
    s+=";"+str(np.mean(result_runs[3])) #Mean AOF
    s+=";"+str(np.median(result_runs[3])) #Median AOF
    s+=";"+str(np.var(result_runs[3])) #Var AOF
    p=";"+"IM" #Flow Type
    p+=";"+str(first_pred_time) #Predicting from
    p+=";"+str(tr1)
    p+=";"+str(tr2)
    p+=";"+str(noise1)
    p+=";"+str(noise2)
    p+=";"+str(train_percent)
    p+=";"+str(layer_size)
    p+=";"+str(reg1)
    p+=";"+str(reg2)
    p+=";"+str([x for x in Epochs])
    p+=";"+str([x for x in Batch_size])
    p+=";"+str(int(sqrp))
    p+=";"+str(int(time_delta))
    p+=";"+str(last)
    
    s=str(abs(hash(p)))+s+p

    s+="\n"
    with open(path+"results/results.txt",'a') as arq:
        arq.write(s)
        
    return 1