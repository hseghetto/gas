# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 02:06:36 2020

@author: hsegh
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm

import tensorflow as tf

from tensorflow import keras

class PrintDot(keras.callbacks.Callback): #Helps tracking progress in epochs during training
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def noise(x): #sin noise
    for i in range(len(x)):
        x[i][1]=x[i][1]+np.sin(time[i][0]*3.1415)*3
    return x

def gauss_noise(data): #white noise
    for i in range(len(data)):
        noise=np.random.normal(0,0.01,1)
        data[i][1]=data[i][1]*(1+noise)
    return data

def arqScatter(arq): #Displays Pressure x Time for a dataset, colorcoded for extraction rate
    plt.scatter([i[0] for i in arq],[j[1] for j in arq],c=[k[-1] for k in arq],cmap="Set1")
    plt.xlabel("Time")
    plt.ylabel("Presssure")
    plt.show() 

def modelGraphs(hist): #Receives model training history and displays error graphs
    plt.figure(figsize=[2*6.4/1,3*4.8/2])
    P=max(len(hist)-Patience*2,10)
    
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
    
def resultGraphs(prediction,target): #Plots predicted pressures together with respective targets
    #Pressure x Time graph 
    plt.scatter([i[0] for i in time[last+1:len(target)+last+1]],[j for j in target],c="blue",marker="x")
    plt.scatter([i[0] for i in time[last+1:len(prediction)+last+1]],[j for j in prediction],c="red",marker="+")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    #plt.ylim([100,300])
    plt.show()
    
    #Target x Prediction graph
    plt.scatter([i[0] for i in prediction],[j for j in target],c=[k[0] for k in time[last+1:len(target)+last+1]],cmap='nipy_spectral')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.plot([100,300],[100,300],c="magenta")
    #plt.ylim([100,300])
    #plt.xlim([100,300])
    plt.grid(True)
    plt.show()
    
def sampling(arq): 
    #This function is used to reduce the dataset size, picking points only if either
    # 1: pressure difference from last chosen point and a given i-th point is bigger than a Pthreshold
    # 2: pressure difference from a given i-th point and the next point is bigger than the Pthreshold
    # 3: time difference from last chosen point and a given i-th point is bigger than a Tthreshold
    # This was needed to use with the RNN models, otherwise time spent training and making predictions would be prohibitive
    # This may or may not negatively influence a given model`s capacity to propperly learn the data
    r=[]
    tr=0.5
    tr2=0.5
    
    r.append(arq[0])
    for i in range(len(arq)-1):
        if(np.abs(arq[i][1]-r[-1][1])>tr or np.abs(arq[i][1]-arq[i+1][1])>tr or np.abs(arq[i][0]-r[-1][0])>tr2):
            r.append(arq[i])
    r=np.array(r)
    return r
            
def preprocess(arq): #returns the arrays to be fed the model
    #When using an RNN data must be fed with an {N, timestep, features} shape
    #In our case this means N is the number of datapoints that can be used to predict (we cant predict if there are not at least L past datapoints)
    #timestep is the number of L past points used for each prediction
    #we used [timestamp, past pressure, flow rate] for each timestep giving us features=3
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

def calcTime(a):#returns an array with the absolute time and deltaT of data
    t=[]
    t.append([a[0][0],0])
    for i in range(1,len(a)):
        t.append([a[i][0],a[i][0]-a[i-1][0]])
    return t

def split(a):#return the index used to split the dataset between training+validation and prediction
    ind=[]
    for i in range(1,len(a)):
        if(a[i-1][-1]!=a[i][-1]):
            ind.append(i)
    return ind

def aof(pressures,flow_rates):
    
    pressures=(300**2-np.array(pressures)**2)
    flow_rates=np.array(flow_rates)/1000
    
    model=np.polyfit(flow_rates, pressures, 1)
    print(pressures)
    print(model)
    print(np.poly1d(model))
    
    aof = ((300**2-1.033**2)-model[1])/model[0]
    
    return aof
"""
def loss(x_train,full_x, pressure_to_predict):
    full_x = np.expand_dims(full_x, axis=0)
    #x_train = np.expand_dims(x_train, axis=0)
    full_x = full_x.astype("float32")
    l=(model([full_x,x_train])[0]-pressure_to_predict)**2

    return l

def train_step_2(x_train,x_train_next,pressure_to_predict):
    with tf.GradientTape() as tape:  
        y=model(x_train)
        x_train_next[-1][1]=y
        l = (model(x_train_next)-pressure_to_predict)**2
      
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
        
        for i in np.shuffle(list(range(1,len(train_data)))):
              l = train_step(train_data_norm[i-1],train_data_norm[i],target[i])
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
"""
#--------------------------------
t0 =tm.perf_counter()

pd.set_option('display.max_columns', None)

save=False
for seed in range(0,3): 
    tf.executing_eagerly()
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    last=2   #L previous pressure points to be used for each prediction 
    next_steps=2
    
    #Setting seed to ensure reproducebility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    #Reading dataset as an array
    a=pd.read_csv("data/gas_si_extendido_1.txt",sep=" ")   
    a=a.to_numpy()
    
    #reducing dataset, aplying noise and graphing
    a=sampling(a)
    #a=noise(a)
    arqScatter(a)
    
    index=split(a)
    #saving data for AOF
    pressures=[]
    rates=[]
    for i in index:
        if(a[i-1][-1]!=0):
            pressures.append(a[i-1][1])
            rates.append(a[i-1][-1])
            
    #removing datapoints not to be used for training
    a=a[0:index[1]]
    time=calcTime(a)
    print(a[-2:])
    arqScatter(a)
    
    print(aof(pressures,rates))
    
    squareP=True
    if(squareP==True):
        for i in range(len(a)):
            a[i][1]=np.square(a[i][1]) #squaring pressures
            
    deltaT=True
    if(deltaT==True):
        for i in range(1,len(a)):
            a[-i][0]=np.abs(a[-i][0]-a[-i-1][0]) #replacing timestamps with time delta
    
    #saving data about the sequence used
    data_df=pd.DataFrame(a)
    data_df=data_df.describe()
    data_stats=data_df.to_numpy()
    
    #shaping the data correctly
    train_data,target=preprocess(a) #we use the model to predict the real pressure, hence we need the target array before normalizing data
    
    #normalizing/standarizing the sequence
    for i in range(0,len(a[0])):
        a[:,i]=(a[:,i]-data_stats[1,i])/data_stats[2,i] #standarization
        #a[:,i]=a[:,i]/data_stats[-1,i] #normalization
        
    #shaping the data correctly
    train_data_norm,target_norm=preprocess(a) #train_data_norm will be used as input for the model, using train_data would significantly hinder the model
    
    #[0:train_split] will be used for traing, [train_split:] will be used for validation
    train_split=int(len(train_data)*0.8)
    
    #crating the lists of indexes for training and validation 
    r=list(range(len(train_data)))
    np.random.shuffle(r) 
    train_index=r[0:train_split]
    val_index=r[train_split:]
    
    #------------ BUILDING THE NETWORK -------------------------------------------
    print(train_data.shape)
    
    layer_size=16
    
    reg=0.05
    
    model = keras.Sequential()
    model.add(keras.layers.GRU(layer_size, kernel_regularizer=keras.regularizers.l2(reg),
                     activity_regularizer=keras.regularizers.l1(0.), batch_input_shape=(1, train_data_norm.shape[1], train_data_norm.shape[2])))
    #model.add(keras.layers.Flatten(input_shape=(train_data_norm.shape[1], train_data_norm.shape[2])))
    #model.add(keras.layers.Dense(layer_size,activation="tanh"))
    #model.add(keras.layers.Dense(layer_size,activation="tanh"))    
    model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l2(reg),
                     activity_regularizer=keras.regularizers.l1(0.)))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.mse,
                  metrics=['mae','mse','mape'])
    
    EPOCHS = 1000
    Patience=100
    early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=Patience)
    history = model.fit(train_data_norm[train_index], target[train_index], epochs=EPOCHS,
                        validation_data=(train_data_norm[val_index], target[val_index]), 
                        verbose=0, callbacks=[PrintDot()])
    
    #---------------- EVALUATING and TESTING -------------------------------------------------
    hist = pd.DataFrame(history.history)
    
    print("-/-")
    hist['epoch'] = history.epoch
    print(hist.tail(1))
    
    modelGraphs(hist)
    
    #s=["data/gas_primeiro_caso_variavel.txt","data/gas_segundo_caso_variavel.txt","data/gas_terceiro_caso_variavel.txt","data/gas_quarto_caso_variavel.txt","data/gas_quinto_caso_variavel.txt"]
    s=["data/gas_si_extendido_1.txt",
       "data/gas_sd_extendido_1.txt",
       "data/gas_im_extendido_1.txt"]
    
    for k in range(len(s)):#this is meant to test model results when fed real data
        #we do all the same data treatment for each dataset
        a=pd.read_csv(s[k],sep=" ")    
        a=a.to_numpy()
        
        a=sampling(a)
        index=split(a)
        time=calcTime(a)
            
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
        resultGraphs(prediction,target[0:len(prediction)])
        
    #-------------------Predictions---------------------------
    #From now on we will make dynamic predictions using the result from previous predictions 
    a=pd.read_csv("data/gas_si_extendido_1.txt",sep=" ")
    a=a.to_numpy()
    a=sampling(a)
    index=split(a)
    
    print(a[-5:])
    
    a=a[index[1]-last:] #use index-last: to predict from the last point before 144h 
    #a=a[index-1:] #use index-1: to predict from the first L points after 144h 
    time=calcTime(a)
    
    if(squareP==True):
        for i in range(len(a)):
            a[i][1]=np.square(a[i][1])
    
    if(deltaT==True):
        for i in range(1,len(a)):
            a[-i][0]=np.abs(a[-i][0]-a[-i-1][0])#replacing timestamps with time delta
            
    predict_data,predict_target=preprocess(a)
    #predict data will contain the data resulting from each prediction
    #we initialize it with the preprocessed data of the case to be predicted
    #test will be receiving the normed values of each entry to be used as input
    test=np.zeros(predict_data[0].shape)
    r=[]
    
    for i in range(0,len(predict_data)):
        for j in range(len(predict_data[0][0])):
            test[:,j]=(predict_data[i,:,j]-data_stats[1,j])/data_stats[2,j]
            
        r.append(model.predict([np.array([test])])[0][0]) #resulting pressure prediction
        
        i=i+1
        for j in range(min(last,len(predict_data)-i)):
            #we feed the prediction result back into the predict data to be used in future iterations
            predict_data[i+j][-j-1][1]=r[-1]*300
    
    #plotting results
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
    
    #calculating error
    mse=np.zeros(len(predict_target))
    mae=np.zeros(len(predict_target))
    mape=np.zeros(len(predict_target))
    
    for i in range(0,len(predict_data)):
        mse[i]=np.square(r[i]-predict_target[i])
        mae[i]=np.abs(r[i]-predict_target[i])
        mape[i]=mae[i]/r[i]*100
    
    print(np.max(mse))
    print(np.max(mae))
    print(np.max(mape))
    print(np.mean(mse))
    print(np.mean(mae))
    print(np.mean(mape))
    
    #saving results
    if(save==True):
        with open("plots/"+str(seed)+".txt",'w+') as arq: #appending results to hist.txt
            arq.write(str(hist.tail(1)))
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
            arq.write("\n")
            arq.write("Final pressure:"+str(r[-1])) 
            
    
            
    """
    #calculating bourdet
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
    """