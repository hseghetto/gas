# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 23:17:31 2020

@author: hsegh
"""

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
    for i in range(len(x)):
        x[i][1]=x[i][1]+np.sin(time[i][0]*3.1415)*3
    return x

def gauss_noise(data):
    noise=np.random.normal(0,0.01,len(data))
    print(noise.shape)
    for i in range(len(data)):
        data[i][1]=data[i][1]*(1+noise[i])
    return data

def arqScatter(arq):
    plt.scatter([i[0] for i in arq],[j[1] for j in arq],c=[k[-1] for k in arq],cmap="Set1")
    plt.xlabel("Time")
    plt.ylabel("Presssure")
    plt.show() 

def modelGraphs(hist):
    plt.figure(figsize=[2*6.4/1,3*4.8/2])
    P=max(len(hist)-Patience*2,10,len(hist)-100)
    
    #Loss function graphs
    plt.subplot(321)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'][10:], hist['loss'][10:],label='Train Loss Function')
    plt.plot(hist['epoch'][10:], hist['val_loss'][10:],label = 'Val Loss Functionr')
    plt.legend()
    
    #Mean absolute error graphs
    plt.subplot(323)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(hist['epoch'][10:], hist['mae'][10:],label='Train Error')
    plt.plot(hist['epoch'][10:], hist['val_mae'][10:],label = 'Val Error')
    plt.legend()
    
    #Mean square error graphs
    plt.subplot(325)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(hist['epoch'][10:], hist['mse'][10:],label='Train Error')
    plt.plot(hist['epoch'][10:], hist['val_mse'][10:],label = 'Val Error')
    plt.legend()
    
    #Loss function graphs
    plt.subplot(322)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'][P:], hist['loss'][P:],label='Train Loss Function')
    plt.plot(hist['epoch'][P:], hist['val_loss'][P:],label = 'Val Loss Functionr')
    plt.legend()
    
    #Mean absolute error graphs
    plt.subplot(324)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(hist['epoch'][P:], hist['mae'][P:],label='Train Error')
    plt.plot(hist['epoch'][P:], hist['val_mae'][P:],label = 'Val Error')
    plt.legend()
    
    #Mean square error graphs
    plt.subplot(326)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(hist['epoch'][P:], hist['mse'][P:],label='Train Error')
    plt.plot(hist['epoch'][P:], hist['val_mse'][P:],label = 'Val Error')
    plt.legend()
    
    plt.show()
    print(np.mean(np.abs(target)))
    
def resultGraphs(prediction,target):
    plt.scatter([i[0] for i in time[last+1:len(target)+last+1]],[j for j in target],c="blue",marker="x")
    plt.scatter([i[0] for i in time[last+1:len(prediction)+last+1]],[j for j in prediction],c="red",marker="+")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    #plt.ylim([100,300])
    plt.show()
    plt.scatter([i[0] for i in prediction],[j for j in target],c=[k[0] for k in time[last+1:len(target)+last+1]],cmap='nipy_spectral')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.plot([100,300],[100,300],c="magenta")
    #plt.ylim([100,300])
    #plt.xlim([100,300])
    plt.grid(True)
    plt.show()
    
def sampling(arq):
    r=[]
    tr=0.5
    tr2=2
    
    r.append(arq[0])
    for i in range(len(arq)-1):
        if(np.abs(arq[i][1]-r[-1][1])>tr or np.abs(arq[i][1]-arq[i+1][1])>tr or np.abs(arq[i][0]-r[-1][0])>tr2):
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

def calcTime(a):
    t=[]
    t.append([a[0][0],0])
    for i in range(1,len(a)):
        t.append([a[i][0],a[i][0]-a[i-1][0]])
    return t

def split(a):
    i=0
    while(a[i][0]<144 or a[i][-1]!=200000):
        i=i+1
    return i

def BuildModel(fullData, Data):
    print("Data",Data.shape)
    input1 = tf.keras.layers.Input(shape = fullData.shape)
    input2 = tf.keras.layers.Input(shape = [Data.shape[1],Data.shape[2]])
    
    x = tf.keras.layers.Conv1D(5, 5, strides=5, padding="same", use_bias=False)(input1)
    x = tf.keras.layers.MaxPooling1D(5, padding="same")(x)
    x = tf.nn.leaky_relu(x)
    x = tf.keras.layers.Conv1D(5, 5, strides=5, padding="same", use_bias=False)(input1)
    x = tf.keras.layers.MaxPooling1D(5, padding="same")(x)
    x = tf.nn.leaky_relu(x)
    x = tf.keras.layers.Conv1D(5, 5, strides=5, padding="same", use_bias=False)(input1)
    x = tf.keras.layers.MaxPooling1D(5, padding="same")(x)
    x = tf.nn.leaky_relu(x)
    x = tf.keras.layers.Dense(last, activation="relu")(x)
    
    #x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(1)(x)
    #x = tf.keras.layers.RepeatVector(last)(x)
    
    x = tf.keras.layers.Permute((2,1))(x)
    
    #x = tf.keras.layers.Reshape((5,3))(x)
    
    
    
    print(x)
    y = tf.keras.layers.Concatenate(axis=-1)([x, input2])
    print(y)
    
    z = tf.keras.layers.GRU(32)(y)
    z = tf.keras.layers.Dense(1)(z)
    
    return tf.keras.Model(inputs=[input1, input2], outputs= z)


def loss(x_train,full_x, pressure_to_predict):
    full_x = np.expand_dims(full_x, axis=0)
    #x_train = np.expand_dims(x_train, axis=0)
    full_x = full_x.astype("float32")
    l=(model([full_x,x_train])[0]-pressure_to_predict)**2

    return l

optimizer=tf.keras.optimizers.Adam(0.01)

def train_step(x_train,full_x,pressure_to_predict):
    with tf.GradientTape() as tape:  
        l = loss(x_train,full_x,pressure_to_predict)
      
    gradient = tape.gradient(l,model.trainable_variables)
    optimizer.apply_gradients(zip(gradient,model.trainable_variables))
    
    return l


def fit(train_dataset,val_dataset, x_full):
    tf.print("Begin training...")
    total_loss=[[],[]]
    for epoch in range(EPOCHS):
        # Train Loss
        full_loss=[]
        if(epoch%100==0):
            print("")
        print('.', end='')
        for n, (train_x, label) in train_dataset.enumerate():
              l = train_step(train_x,x_full,label)
              full_loss.append(l)
        total_loss[0].append(np.mean(full_loss))
        full_loss=[]
        for n, (val_x, label) in val_dataset.enumerate():
              l=loss(val_x,x_full,label)
              full_loss.append(l)
        total_loss[1].append(np.mean(full_loss))
    print(total_loss)
    total_loss=np.array(total_loss)
    return np.transpose(total_loss)

#--------------------------------
t0 =tm.perf_counter()

pd.set_option('display.max_columns', None)

save=False
for seed in [0,1,2]: 
    tf.executing_eagerly()
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    last=3
    
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    a=pd.read_csv("data/gas_im_extendido_1.txt",sep=" ")
    
    a=a.to_numpy()
    a=a.astype("float32")
    
    a=sampling(a)
    #a=noise(a)
    arqScatter(a)
    
    index=split(a)
    time=calcTime(a)
    a=a[0:index]
    print(a[-2:])
    
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
    
    r=list(range(len(train_data)))
    np.random.shuffle(r)
    train_index=r[0:train_split]
    val_index=r[train_split:]
    
    #------------ BUILDING THE NETWORK -------------------------------------------
    print(train_data.shape)
    
    layer_size=16
    
    reg=0.000
    
    model = BuildModel(a, train_data_norm)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.mse,
                  metrics=['mae','mse','mape'])
    
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    
    EPOCHS = 200
    Patience=0
    
    batch_size=1
    target=target.astype("float32")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data_norm[train_index],target[train_index]))
    train_dataset = train_dataset.batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((train_data_norm[val_index],target[val_index]))
    val_dataset = val_dataset.batch(batch_size)
    
    l = fit(train_dataset, val_dataset, a)
    #history = model.fit([[a]*len(train_data_norm), train_data_norm], target, epochs=EPOCHS, verbose=1)
    
    #---------------- EVALUATING and TESTING -------------------------------------------------

    prediction = model.predict([[a]*len(train_data_norm),train_data_norm])
    resultGraphs(prediction,target[0:len(prediction)])
    
    plt.plot([i for i in range(len(l))], [j[0] for j in l])
    plt.plot([i for i in range(len(l))], [j[1] for j in l])
    plt.show()
    
    
    #-------------------Predictions---------------------------
    b=pd.read_csv("data/gas_im_extendido_1.txt",sep=" ")
    b=b.to_numpy()
    b=sampling(b)
    index=split(b)
    
    print(b[-5:])
    
    b=b[index-last:] #use index-last: to predict from the last point before 144h 
    #b=b[index-1:] #use index-1: to predict from the first L points after 144h 
    time=calcTime(b)
    
    if(squareP==True):
        for i in range(len(b)):
            b[i][1]=np.square(b[i][1])
    
    if(deltaT==True):
        for i in range(1,len(b)):
            b[-i][0]=np.abs(b[-i][0]-b[-i-1][0])#replacing timestamps with time delta
            
    predict_data,predict_target=preprocess(b)
    #predict data will contain the data resulting from the prediction
    #we initialize it with the preprocess result of the case to be predicted
    #test will be receive the normed values of each entry to be used as input
    test=np.zeros(predict_data[0].shape)
    r=[]
    
    for i in range(0,len(predict_data)):
        for j in range(len(predict_data[0][0])):
            test[:,j]=(predict_data[i,:,j]-data_stats[1,j])/data_stats[2,j]
        
        x=[np.array([a]),np.array([test])]
        r.append(model.predict(x)[0][0]) #resulting pressure prediction
        
        i=i+1
        for j in range(min(last,len(predict_data)-i)):
            #we feed the prediction result back into the predict data to be used in future iterations
            predict_data[i+j][-j-1][1]=r[-1]*300
    
                
    plt.scatter([i for i in r[0:len(predict_data)]],[j for j in predict_target[0:len(predict_data)]],c=[k[0] for k in time[0:len(predict_data)]],cmap='nipy_spectral')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.plot([np.min(r)-10,10+np.max(r)],[np.min(r)-10,10+np.max(r)],c="magenta")
    #plt.ylim([100,300])
    #plt.xlim([100,300])
    plt.grid(True)
    plt.show()
    
    plt.scatter([i[0] for i in time[last:last+len(predict_data)]], [j for j in predict_target[0:len(predict_data)]], c="magenta",marker="x")
    plt.scatter([i[0] for i in time[last:last+len(predict_data)]], [j for j in r[0:len(predict_data)]], c=[k[0] for k in time[last:last+len(predict_data)]],cmap='nipy_spectral',marker="+")
    plt.xlabel("Time")
    plt.ylabel("Presssure")
    #plt.ylim([100,300])
    if(save==True):
        plt.savefig("plots/"+str(seed)+'.png')
    plt.show()
    
    t1=tm.perf_counter()
    print("Time elapsed:",t1-t0)
    
    mse=np.zeros(len(predict_target))
    mae=np.zeros(len(predict_target))
    mape=np.zeros(len(predict_target))
    
    for i in range(0,len(predict_data)):
        mse[i]=np.square(r[i]-predict_target[i])
        mae[i]=np.abs(r[i]-predict_target[i])
        mape[i]=mae[i]/predict_target[i]*100
    
    print(np.max(mse))
    print(np.max(mae))
    print(np.max(mape))
    print(np.mean(mse))
    print(np.mean(mae))
    print(np.mean(mape))
    
    if(save==True):
        with open("plots/"+str(seed)+".txt",'w+') as arq: #appending results to hist.txt
            #arq.write(str(hist.tail(1)))
            arq.write("\n")
            arq.write("Max mse:"+str(np.max(mse)))
            arq.write("\n")
            arq.write("Max mae:"+str(np.max(mae)))
            arq.write("\n")
            arq.write("Max mape:"+str(np.max(mape)))
            arq.write("\n")
            arq.write("Mean mse:"+str(np.mean(mse)))
            arq.write("\n")
            arq.write("Mean mae:"+str(np.mean(mae)))
            arq.write("\n")
            arq.write("Mean mape:"+str(np.mean(mape)))
    
    bourdet=np.zeros((len(r),2))
    bourdet[0][0]=300-r[0]
    for i in range(1,len(r)):
        bourdet[i][0]=300-r[i]
        bourdet[i][1]=np.abs((bourdet[i][0]-bourdet[i-1][0])/np.log(time[i+last-1][0]/time[i+last-2][0]))
        
    plt.scatter([i[-1] for i in time[0:len(bourdet)]],[j[0] for j in bourdet[0:len(bourdet)]],c="blue")
    plt.scatter([i[-1] for i in time[0:len(bourdet)]],[j[1] for j in bourdet[0:len(bourdet)]],c="red")
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
    