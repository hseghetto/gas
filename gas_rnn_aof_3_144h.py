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
    if epoch % 1000 == 0: print('')
    if epoch % 10 == 0: print('.', end='')

def sin_noise(x,intensity,frequency=3.1415): #sin noise
    for i in range(len(x)):
        x[i][1]=x[i][1]+np.sin(time[i][0]*frequency)*intensity
    return x

def gauss_noise(data,sigma): #white noise
    for i in range(len(data)):
        noise=np.random.normal(0,sigma,1)
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
    #plt.ylim([100,initial_pressure])
    plt.show()
    
    #Target x Prediction graph
    plt.scatter([i[0] for i in prediction],[j for j in target],c=[k[0] for k in time[last+1:len(target)+last+1]],cmap='nipy_spectral')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.plot([100,initial_pressure],[100,initial_pressure],c="magenta")
    #plt.ylim([100,initial_pressure])
    #plt.xlim([100,initial_pressure])
    plt.grid(True)
    plt.show()
    
def sampling(arq,tr1,tr2): 
    #This function is used to reduce the dataset size, picking points only if either
    # 1: pressure difference from last chosen point and a given i-th point is bigger than a Pthreshold
    # 2: pressure difference from a given i-th point and the next point is bigger than the Pthreshold
    # 3: time difference from last chosen point and a given i-th point is bigger than a Tthreshold
    # This was needed to use with the RNN models, otherwise time spent training and making predictions would be prohibitive
    # This may or may not negatively influence a given model`s capacity to propperly learn the data
    r=[]
    
    r.append(arq[0])
    for i in range(len(arq)-1):
        if(np.abs(arq[i][1]-r[-1][1])>tr1 or np.abs(arq[i][1]-arq[i+1][1])>tr1 or np.abs(arq[i][0]-r[-1][0])>tr2):
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
        
        label.append(arq[i+last,1]/pfactor) #appending pressure target
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
    ind=[0]
    for i in range(1,len(a)):
        if(a[i-1,-1]!=a[i,-1]):
            ind.append(i)
        if(a[ind[-1]][0]<143 and a[i][0]>143):
            ind.append(i)
    return ind

def calc_aof(pressures,flow_rates):
    
    pressures=(initial_pressure**2-np.array(pressures)**2)
    flow_rates=np.array(flow_rates)
    
    model=np.polyfit(flow_rates, pressures, 1)
    
    aof = ((initial_pressure**2-1.033**2)-model[1])/model[0]
    
    return aof

def loss(x,pressure_to_predict):  
    for i in range(PRED):
        #print(x[:,i,:,:].shape)
        y=model(x[:,i,:,:])
        y=np.squeeze((y*pfactor-data_stats[1][1])/data_stats[2][1])
        for j in range(1,min(PRED+1-i,last+1)):
            x[:,i+j,-j,1]=y
    
    l = (model(x[:,-1])-pressure_to_predict)**2

    return l

def train_step(x,pressure_to_predict):
    with tf.GradientTape() as tape:  
        l=loss(x,pressure_to_predict)
        
        """y=model(np.expand_dims(x_train,axis=0))
        x_train_next[-1][1]=(y*initial_pressure-data_stats[1][1])/data_stats[2][1]
        l = (model(np.expand_dims(x_train_next,axis=0))-pressure_to_predict)**2"""
      
    gradient = tape.gradient(l,model.trainable_variables)
    optimizer.apply_gradients(zip(gradient,model.trainable_variables))
    
    return l

optimizer=tf.keras.optimizers.Adam(0.01)

def fit(train_x,train_label,val_x,val_label):
    train_size=len(train_x)
    val_size=len(val_x)
    
    tf.print("Begin training...")
    total_loss=[[],[]]
    for epoch in range(EPOCHS):
        if(epoch%100==0):
            print("")
        print('.', end='')
        
        full_loss=0
        for i in range(0,train_size,BATCH_SIZE):
            l = train_step(train_x[i:i+BATCH_SIZE], train_label[i:i+BATCH_SIZE])
            full_loss += np.sum(l)
        total_loss[0].append(full_loss/train_size)
        
        full_loss=0
        for i in range(0,val_size,BATCH_SIZE):
            l = np.ndarray.flatten(np.array(loss(val_x[i:i+BATCH_SIZE], val_label[i:i+BATCH_SIZE])))
            full_loss += np.sum(l)
        total_loss[1].append(full_loss/val_size)
        
    total_loss=np.array(total_loss)
    return np.transpose(total_loss)

def customFitPreprocess():
    r=[]
    for i in range(PRED,len(train_data)):
        aux=np.array([x for x in [y for y in [z for z in train_data_norm[i-PRED:i+1]]]])
        r.append(aux)
    return np.array(r)
#--------------------------------
t0 =tm.perf_counter()


pd.set_option('display.max_columns', None)

initial_pressure=300
pfactor=1

verbose=False #set this to true if you want to see testing and prediction graphs
"""
This will use data from 0-144h to predict the last 144h
Do notice that running this code would result in training the network 2*4*4*5*6*4*10 = 38400 times
I do not expect this to be completed in one day (actually I guess you can make about 500 runs per day, if each run takes no more than 5 minutes)
I did, however, put the hyperparameters i consider the most important deeper in the loops, such that there will be variance in their valuer with less runs
You may want to change a few of these parameter ranges such that it doenst take so long
There are also other hyperparameter wich i didnt include here, most notably the reg1 for L_1 regularization, but also tr1 and tr2 for the adaptative timestep  
I also left tup with 100 epoch with 8 of batchsize + 100 with 64, this is barelly enough to achieve convergence, so results may be subpar
"""

for squareP in [True,False]:
    for last in [2]:
        for noise1 in [0.01]:
            for train_percent in [0.6,0.7,0.8]:
                for reg2 in [0.01,0.05,0.1]:
                    for layer_size in [8,16,24,32]:
                        result_runs=[[],[],[],[]]
                        for (EPOCHS,BATCH_SIZE) in [[75,8],[100,8],[125,8],[150,8]]:
                            for seed in range(10): #number of runs per hyperparameter set
                                print(seed)
                                
                                tf.executing_eagerly()
                                #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
                                PRED = 1
                                #last= 3  #L previous pressure points to be used for each prediction 
                                
                                #Setting seed to ensure reproducebility
                                tf.random.set_seed(seed)
                                np.random.seed(seed)
                                
                                #Reading dataset as an array
                                a=pd.read_csv("data/gas_im_extendido_1.txt",sep=" ")  
                                #a=pd.read_csv("data/gas_terceiro_caso_interpolado.txt",sep=" ",usecols=[0,1,2])  
                                a=a.to_numpy()
                                
                                #reducing dataset, aplying noise and graphing
                                tr1=0.0
                                tr2=0.0
                                a=sampling(a,tr1,tr2)
                                time=calcTime(a)
                                
                                index=split(a)
                               
                                #removing datapoints not to be used for training
                                a=a[0:index[-1]]
                                #a=a[0:index[4]]
                                #a=a[index[0]:index[3]]
                                #a=a[index[1]:index[5]]
                            
                                #noise1=0.01
                                a=gauss_noise(a,noise1)
                                noise2=0
                                a=sin_noise(a,noise2)
                                
                                #squareP=False
                                pfactor=1
                                if(squareP==True):
                                    pfactor=initial_pressure
                                    for i in range(len(a)):
                                        a[i][1]=np.square(a[i][1]) #squaring pressures
                                        
                                deltaT=True
                                if(deltaT==True):
                                    for i in range(1,len(a)):
                                        a[-i][0]=np.abs(a[-i][0]-a[-i-1][0]) #replacing timestamps with time delta
                                
                                if(verbose):
                                    arqScatter(a)
                                
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
                                #train_percent=0.8
                                train_split=int(len(train_data)*train_percent)
                                
                                #crating the lists of indexes for training and validation 
                                r=list(range(PRED,len(train_data)-1))
                                np.random.shuffle(r) 
                                train_index=r[0:train_split]
                                val_index=r[train_split:]
                                
                                #------------ BUILDING THE NETWORK -------------------------------------------
                                
                                #layer_size=16
                                
                                reg1=0.00
                                #reg2=0.05
                                
                                model = keras.Sequential()
                                model.add(keras.layers.GRU(layer_size, kernel_regularizer=keras.regularizers.l1_l2(reg1,reg2),
                                                           input_shape=(train_data_norm.shape[1], train_data_norm.shape[2])))
                                #model.add(keras.layers.Flatten(input_shape=(train_data_norm.shape[1], train_data_norm.shape[2])))
                                #model.add(keras.layers.Dense(layer_size,activation="tanh"))
                                #model.add(keras.layers.Dense(layer_size,activation="tanh"))    
                                model.add(keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l2(reg2),
                                                 activity_regularizer=keras.regularizers.l1(reg1)))
                                
                                model.compile(optimizer='adam',
                                              loss=tf.keras.losses.mse,
                                              metrics=['mae','mse','mape'])
                            
                                for (EPOCHS,BATCH_SIZE) in tup:
                                    Patience=EPOCHS//10
                                    history = model.fit(train_data_norm[train_index], target[train_index], epochs=EPOCHS,
                                                        validation_data=(train_data_norm[val_index], target[val_index]), 
                                                        verbose=0, callbacks=[], batch_size=BATCH_SIZE)
                                
                                
                            
                                #---------------- EVALUATING and TESTING -------------------------------------------------
                                
                                if(verbose): 
                                    
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
                                        
                                        a=sampling(a,tr1,tr2)
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
                                a=pd.read_csv("data/gas_im_extendido_1.txt",sep=" ")
                                #a=pd.read_csv("data/gas_quinto_caso_alterado.txt",sep=" ")
                                a=a.to_numpy()
                                a=sampling(a,tr1,tr2)
                                index=split(a)
                                
                                #a=a[0:index[1]]
                                #a=a[index[3]-last:index[-3]] #use index-last: to predict from the last point before 144h 
                                a=a[index[-1]-last:] #use index-1: to predict from the first L points after 144h 
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
                                        predict_data[i+j][-j-1][1]=r[-1]*pfactor
                                
                                #calculating error
                                mse=np.zeros(len(predict_target))
                                mae=np.zeros(len(predict_target))
                                mape=np.zeros(len(predict_target))
                                
                                for i in range(0,len(predict_data)):
                                    mse[i]=np.square(r[i]-predict_target[i])
                                    mae[i]=np.abs(r[i]-predict_target[i])
                                    mape[i]=mae[i]/predict_target[i]*100
                                    
                                aof_rates=[200,400,600]
                                aof_pressures=[249.3,192.1,120.7] # Modified Isochronous
                                
                                last_pressure=predict_data[-1][-1][1]/pfactor
                                last_rate=predict_data[-1][-1][-1]/1000
                                
                                for (i,rate) in enumerate(aof_rates):
                                    if (rate==int(last_rate)):
                                        aof_pressures[i]=last_pressure
                                
                                aof=calc_aof(aof_pressures, aof_rates)
                                
                                result_runs[0].append(np.mean(mse))
                                result_runs[1].append(np.mean(mae))
                                result_runs[2].append(np.mean(mape))
                                result_runs[3].append(aof)
                            
                                if(verbose):
                                    #plotting results
                                    plt.scatter([i for i in r[0:len(predict_data)]],[j for j in predict_target[0:len(predict_data)]],c=[k[0] for k in time[0:len(predict_data)]],cmap='nipy_spectral')
                                    plt.xlabel("Prediction")
                                    plt.ylabel("Target")
                                    plt.plot([np.min(r)-10,10+np.max(r)],[np.min(r)-10,10+np.max(r)],c="magenta")
                                    #plt.ylim([100,initial_pressure])
                                    #plt.xlim([100,initial_pressure])
                                    plt.grid(True)
                                    plt.show()
                                    
                                    plt.scatter([i[0] for i in time[last:last+len(predict_data)]], [j for j in predict_target[0:len(predict_data)]], c="magenta",marker="x")
                                    plt.scatter([i[0] for i in time[last:last+len(predict_data)]], [j for j in r[0:len(predict_data)]], c=[k[0] for k in time[last:last+len(predict_data)]],cmap='nipy_spectral',marker="+")
                                    plt.xlabel("Time")
                                    plt.ylabel("Presssure")
                                    plt.show()
                                
                            with open("results.txt",'a') as arq: 
                                s=";"+str(np.mean(result_runs[0]))
                                s+=";"+str(np.mean(result_runs[1]))
                                s+=";"+str(np.mean(result_runs[2]))
                                s+=";"+str(np.mean(result_runs[3]))
                                s+=";"+"IM"
                                s+=";"+str(a[0][0])
                                s+=";"+str(tr1)
                                s+=";"+str(tr2)
                                s+=";"+str(noise1)
                                s+=";"+str(noise2)
                                s+=";"+str(train_percent)
                                s+=";"+str(layer_size)
                                s+=";"+str(reg1)
                                s+=";"+str(reg2)
                                s+=";"+str([x[0] for x in tup])
                                s+=";"+str([x[1] for x in tup])
                                s+=";"+str(int(squareP))
                                s+=";"+str(last)
                                
                                s=str(abs(hash(s)))+s
                            
                                s+="\n"
                                arq.write(s)
    #print(tup,reg2,np.mean(np.array(mae_runs)))