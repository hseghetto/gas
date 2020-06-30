# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:40:10 2020

@author: hsegh
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm

import tensorflow as tf

from tensorflow import keras

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def noise(x):
    return np.sin(x*np.pi)

def gauss_noise(train_data,last):
    noise=np.random.normal(0,0.1,[len(train_data),last])
    for i in range(len(train_data)):
        for j in range(last):
            train_data[i][j] += noise[i][j]
    return train_data

def arqScatter(arq):
    plt.scatter([i[0] for i in arq],[j[1] for j in arq],c=[k[-1] for k in arq],cmap="Set1")
    plt.xlabel("Time")
    plt.ylabel("Presssure")
    plt.show() 

def modelGraphs(hist):
    
    #Loss function graphs
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'][Patience:], hist['loss'][Patience:],label='Train Loss Function')
    plt.plot(hist['epoch'][Patience:], hist['val_loss'][Patience:],label = 'Val Loss Functionr')
    plt.legend()
    
    #Mean absolute error graphs
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'][Patience:], hist['mae'][Patience:],label='Train Error')
    plt.plot(hist['epoch'][Patience:], hist['val_mae'][Patience:],label = 'Val Error')
    plt.legend()
    
    #Mean square error graphs
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'][Patience:], hist['mse'][Patience:],label='Train Error')
    plt.plot(hist['epoch'][Patience:], hist['val_mse'][Patience:],label = 'Val Error')
    plt.legend()
    
    plt.show()
    print(np.mean(np.abs(target)))

def resultGraphs(prediction):
    plt.scatter([i[0] for i in time[last+1:]],[j for j in prediction],c="red")
    plt.scatter([i[0] for i in time[last+1:]],[j for j in target],c="blue")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    #plt.ylim([100,300])
    plt.show()
    plt.scatter([i[0] for i in prediction],[j for j in target],c=[k[0][-1] for k in train_data_norm],cmap="Set1")
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.plot([100,300],[100,300],c="purple")
    #plt.ylim([100,300])
    #plt.xlim([100,300])
    plt.grid(True)
    plt.show()
    
def sampling(arq):
    r=[]
    
    r.append(arq[0])
    for i in range(len(arq)):
        if(np.abs(arq[i][1]-r[-1][1])>0.5):
            r[-1][0]=arq[i][0]
            r.append(arq[i])
    r=np.array(r)
    return r
            
def preprocess(arq):
    data=[]
    label=[]
    for i in range(0,arq.shape[0]-last-1):
        data.append([])
        for j in range(last):
            data[-1].append([])
            data[-1][-1].append(arq[i+j+1][0]) #appending time
            data[-1][-1].append(arq[i+j,1]) #appending pressure
            data[-1][-1].append(arq[i+j+1,2]) #appending flow rate
            
        label.append(arq[i+last,1]/300) #appending pressure target
        #label.append(arq[i+last,1]-arq[i+last-1,1])
        
    data=np.array(data)
    label=np.array(label)
    
    return data,label
#--------------------------------
t0 =tm.perf_counter()

tf.executing_eagerly()
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

last=3

seed=6
tf.random.set_seed(seed)
np.random.seed(seed)

time=pd.read_csv("data/time.txt",sep=" ")
time=time.to_numpy()

#a=pd.read_csv("data/gas_terceiro_caso_interpolado.txt",sep=" ",usecols=[0,1,2])
#a=pd.read_csv("data/gas_segundo_caso_variavel.txt",sep=" ")
a=pd.read_csv("data/gas_terceiro_caso_variavel.txt",sep=" ")
#a=pd.read_csv("data/gas_quarto_caso_variavel.txt",sep=" ")

a=a.to_numpy()

#a=sampling(a)
arqScatter(a)

squareP=True
if(squareP==True):
    for i in range(len(a)):
        a[i][1]=np.square(a[i][1])
        
deltaT=True
if(deltaT==True):
    for i in range(1,len(a)):
        a[-i][0]=np.abs(a[-i][0]-a[-i-1][0]) #replacing timestamps with time delta
    
data_df=pd.DataFrame(a)
data_df=data_df.describe()
data_stats=data_df.to_numpy()

train_data,target=preprocess(a)
for i in range(0,len(a[0])):
    a[:,i]=(a[:,i]-data_stats[1,i])/data_stats[2,i] #standarization
    #a[:,i]=a[:,i]/data_stats[-1,i] #normalization
    
train_data_norm,target_norm=preprocess(a)

train_split=int(len(train_data)*0.8)

index=list(range(len(train_data)))
np.random.shuffle(index)
train_index=index[0:train_split]
val_index=index[train_split:]

#------------ BUILDING THE NETWORK -------------------------------------------
print(train_data.shape)

layer_size=16

reg=0.005

model = keras.Sequential()
model.add(keras.layers.GRU(layer_size, kernel_regularizer=keras.regularizers.l2(reg),
                 activity_regularizer=keras.regularizers.l1(0.), batch_input_shape=(1, train_data_norm.shape[1], train_data_norm.shape[2])))
#model.add(keras.layers.Dense(layer_size, activation='relu'))
model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l2(reg),
                 activity_regularizer=keras.regularizers.l1(0.)))

model.compile(optimizer='adam',
              loss=tf.keras.losses.mse,
              metrics=['mae','mse','mape'])

EPOCHS = 1000
Patience=30
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=Patience)
history = model.fit(train_data_norm[train_index], target[train_index], epochs=EPOCHS,
                    validation_data=(train_data_norm[val_index], target[val_index]), 
                    verbose=0, callbacks=[PrintDot(),early_stop])

#---------------- EVALUATING and TESTING -------------------------------------------------
hist = pd.DataFrame(history.history)

print("-/-")
hist['epoch'] = history.epoch
print(hist.tail(1))

modelGraphs(hist)

s=["data/gas_primeiro_caso_variavel.txt","data/gas_segundo_caso_variavel.txt","data/gas_terceiro_caso_variavel.txt","data/gas_quarto_caso_variavel.txt","data/gas_quinto_caso_variavel.txt"]
for k in range(len(s)):    
    a=pd.read_csv(s[k],sep=" ")    
    a=a.to_numpy()
    
    if(squareP==True):
        for i in range(len(a)):
            a[i][1]=np.square(a[i][1])

    if(deltaT==True):
        for i in range(1,len(a)):
            a[-i][0]=np.abs(a[-i][0]-a[-i-1][0]) #replacing timestamps with time delta
    
    train_data,target=preprocess(a)
    
    for i in range(0,len(a[0])):
        a[:,i]=(a[:,i]-data_stats[1,i])/data_stats[2,i] #standarization
        #a[:,i]=a[:,i]/data_stats[-1,i] #normalization
        
    train_data_norm,target_norm=preprocess(a)
    prediction = model.predict(train_data_norm)
    resultGraphs(prediction)

#-------------------Predictions---------------------------
a=pd.read_csv("data/gas_quinto_caso_alterado.txt",sep=" ")
a=a.to_numpy()
a=a[0:150]

if(squareP==True):
    for i in range(len(a)):
        a[i][1]=np.square(a[i][1])

if(deltaT==True):
    for i in range(1,len(a)):
        a[-i][0]=np.abs(a[-i][0]-a[-i-1][0])#replacing timestamps with time delta

predict_data,predict_target=preprocess(a)
#predict data will contain the data resulting from the prediction
#we initialize it with the preprocess result of the case to be predicted
#test will be receive the normed values of each entry to be used as input
test=np.zeros(predict_data[0].shape)
r=0

for i in range(len(predict_data)-last):
    for j in range(len(predict_data[0][0])):
        test[:,j]=(predict_data[i,:,j]-data_stats[1,j])/data_stats[2,j]
        
    r=model.predict([np.array([test])])[0][0] #resulting pressure prediction
    
    i=i+1
    for j in range(last):
        #we feed the prediction result back into the predict data to be used in future iterations
        predict_data[i+j][-j-1][1]=r*300

            
plt.scatter([i[0][1]/300 for i in predict_data[0:len(predict_data)]],[j for j in predict_target[0:len(predict_data)]],c=[k[0] for k in time[0:len(predict_data)]],cmap="Set1")
plt.xlabel("Prediction")
plt.ylabel("Target")
plt.plot([100,300],[100,300],c="purple")
#plt.ylim([100,300])
#plt.xlim([100,300])
plt.grid(True)
plt.show()

plt.scatter([i[-1] for i in time[0:len(predict_data)]],[j for j in predict_target[0:len(predict_data)]],c="b")
plt.scatter([i[-1] for i in time[0:len(predict_data)]],[j[0][1]/300 for j in predict_data[0:len(predict_data)]],c=[k[0] for k in time[0:len(predict_data)]],cmap="Set1")
plt.xlabel("Time")
plt.ylabel("Presssure")
#plt.ylim([100,300])
plt.show()

t1=tm.perf_counter()
print("Time elapsed:",t1-t0)

mse=np.zeros(len(predict_target))
mae=np.zeros(len(predict_target))
mape=np.zeros(len(predict_target))

for i in range(1,len(predict_data)):
    mse[i]=np.square(predict_data[i,0,1]/300-predict_target[i-1])
    mae[i]=np.abs(predict_data[i,0,1]/300-predict_target[i-1])
    mape[i]=mae[i]/predict_target[i-1]*100

print(np.max(mse))
print(np.max(mae))
print(np.max(mape))
print(np.mean(mse))
print(np.mean(mae))
print(np.mean(mape))

bourdet=np.zeros((len(predict_target),3))
bourdet[0][0]=300-predict_data[0][0][1]
for i in range(1,len(predict_data)-last+1):
    bourdet[i][0]=300-predict_data[i][0][1]
    bourdet[i][1]=np.abs((bourdet[i][0]-bourdet[i-1][0])/np.log(time[i+last-1][-1]/time[i+last-2][-1]))
    
plt.scatter([i[0] for i in time[0:len(bourdet)]],[j[0] for j in bourdet[0:len(bourdet)]],c="blue")
plt.scatter([i[0] for i in time[0:len(bourdet)]],[j[1] for j in bourdet[0:len(bourdet)]],c="red")
plt.show()

plt.scatter([i[0] for i in time[0:len(bourdet)]],[j[0] for j in bourdet[0:len(bourdet)]],c="blue")
plt.scatter([i[0] for i in time[0:len(bourdet)]],[j[1] for j in bourdet[0:len(bourdet)]],c="red")
#plt.scatter([i[0] for i in time],[j[-2] for j in arq],c="green")
#plt.scatter([i[0] for i in time],[j[-1] for j in arq],c="orange")
plt.xscale("log")
plt.yscale("log")
plt.ylim([10,10000])
plt.xlim([0.0001,100])
plt.show()
