# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:05:01 2020

@author: hsegh
"""


import pandas as pd
import seaborn as sns
import numpy as np
import os

def norm_aof(aof_array, aof):
    r = np.tanh((aof_array - aof)/aof)
    return r

def calc_aof(pressures,flow_rates):
    
    pressures=(initial_pressure**2-np.array(pressures)**2)
    flow_rates=np.array(flow_rates)
    
    model=np.polyfit(flow_rates, pressures, 1)
    
    aof = ((initial_pressure**2-1.033**2)-model[1])/model[0]
    
    return aof

initial_pressure=300
aof_rates=[200,400,600]
aof_pressures=[249.3,192.1,120.7]

aof= calc_aof(aof_pressures,aof_rates)
tol=5/100
lower=aof*(1-tol)
upper=aof*(1+tol)

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


# Reading the data
a=pd.read_csv("results 19-11/results.txt",sep=";",index_col=0)  

def func(x):
    return int(x[1:-1])

a["epochs"]=a["epochs"].apply(func)
a["epochs_1"]=a["epochs_1"].apply(func)

# sns.pairplot(a[["Variance","Median_aof","Mean_mse","Mean_mape"]])

print(a.shape)
# sns.pairplot(a[["Variance","Median_aof","trans_size","epochs_1","epochs"]])
# sns.pairplot(a[["Mean_mse","Mean_mape","trans_size","epochs_1","epochs"]],palette="tab10")

# a=a.loc[(a["Mean_aof"]>0) & (a["Mean_aof"]<250)]
# a=a[(a["Median_aof"]>0) & (a["Median_aof"]<250)]

# sns.pairplot(a[["Variance","Median_aof","trans_size","epochs_1","epochs"]])
# sns.pairplot(a[["Mean_mse","Mean_mape","trans_size","epochs_1","epochs"]],palette="tab10")

s = a["Variance"].idxmin()
b = pd.read_csv("results 19-11/520964197112933607.txt",skiprows=[0],header=None,sep=";")
c = pd.read_csv("results 19-11/"+str(s)+".txt",sep=";")

lista = os.listdir("results 19-11/")
lista.pop()

comp = pd.read_csv("results 19-11/"+lista[-1],skiprows=[0],header=None,sep=";")
lista.pop()

for s in lista:
    aux = pd.read_csv("results 19-11/"+s,skiprows=[0],header=None,sep=";")
    comp = pd.concat([comp,aux])
    
sns.pairplot(comp[[0,1,2,3,4,5,6]])
# print(a.loc[5252807442750585345])

# a=a.loc[(a["Mean_aof"]>100) & (a["Mean_aof"]<150)]
# a=a[(a["Median_aof"]>100) & (a["Median_aof"]<150)]

for x in ["trans_size","epochs_1","epochs"]:
    print(a[x].value_counts())

# sns.pairplot(a[["Variance","Median_aof","trans_size","epochs_1","epochs"]])