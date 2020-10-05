# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:05:01 2020

@author: hsegh
"""


import pandas as pd
import seaborn as sns
import numpy as np

def norm_aof(aof_array, aof):
    r = np.tanh((aof_array - aof)/aof)
    return r

def calc_aof(pressures,flow_rates):
    
    pressures=(initial_pressure**2-np.array(pressures)**2)
    flow_rates=np.array(flow_rates)
    
    model=np.polyfit(flow_rates, pressures, 1)
    
    aof = ((initial_pressure**2-1.033**2)-model[1])/model[0]
    
    return aof

a=pd.read_csv("results2.txt",sep=";")  
s="Median_aof"
a=a[a[s]>0]
a=a[a["Mean_aof"]>0]

#sns.heatmap(a[[s]])
#aa=a[[s,"layer_size","reg1","train_percent"]]
#sns.pairplot(a[[s,"layer_size","reg1","train_percent","epochs"]],hue="epochs",diag_kind="hist")

reg1=a.reg1.unique()
epochs=a.epochs.unique()
layers=a.layer_size.unique()
train=a.train_percent.unique()

reg1_stats={}
for x in reg1:
    #print(a.loc(a["reg1"]==x))
    reg1_stats[x]=a[a["reg1"]==x].describe()
for i,el in reg1_stats.items():
    print(" ")
    print("Reg1 ",i)
    print(el.loc[["mean","std","50%"],["Mean_aof","Var","Median_aof"]])
print("===========================")



epochs_stats={}
for x in epochs:
    #print(a.loc(a["reg1"]==x))
    epochs_stats[x]=a[a["epochs"]==x].describe()
for i,el in epochs_stats.items():
    print(" ")
    print("Epochs ",i)
    print(el.loc[["mean","std","50%"],["Mean_aof","Var","Median_aof"]])
print("===========================")


    
layers_stats={}
for x in layers:
    #print(a.loc(a["reg1"]==x))
    layers_stats[x]=a[a["layer_size"]==x].describe()
for i,el in layers_stats.items():
    print(" ")
    print("Layer size ",i)
    print(el.loc[["mean","std","50%"],["Mean_aof","Var","Median_aof"]])
print("===========================")



train_stats={}
for x in train:
    #print(a.loc(a["reg1"]==x))
    train_stats[x]=a[a["train_percent"]==x].describe()
for i,el in train_stats.items():
    print(" ")
    print("Train percent ",i)
    print(el.loc[["mean","std","50%"],["Mean_aof","Var","Median_aof"]])
print("===========================")



b=np.array([x for x in a[s]])
r= norm_aof(b,727)
a[s]=r

aof=727
tol=5/100
lower=aof*(1-tol)
upper=aof*(1+tol)
lower_norm= norm_aof(lower,aof)
upper_norm= norm_aof(727*1.1,aof)

#sns.pairplot(a[[s,"layer_size","reg1","train_percent","epochs"]],hue="epochs",diag_kind="hist")
"""
initial_pressure=300
aof_rates=[200,400,600]
aof_pressures=[249.3,192.1,120.7]

aof= calc_aof(aof_pressures,aof_rates)

aof_rates=[200,400]
aof_pressures=[249.3,192.1]

aof2= calc_aof(aof_pressures,aof_rates)

plin=600*(192.1-249.3)/200+(249.3*2-192.1)
aof_rates=[200,400,600]
aof_pressures=[249.3,192.1,plin]

aof3= calc_aof(aof_pressures,aof_rates)

aof_ex=[]
for p in range(-100,300):
    aof_rates=[200,400,600]
    aof_pressures=[249.3,192.1,p]
    
    aof_ex.append([p,calc_aof(aof_pressures,aof_rates)])
aof_ex=np.array(aof_ex)
"""