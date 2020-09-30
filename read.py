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

a=pd.read_csv("results.txt",sep=";")  
s="Median_aof"

sns.pairplot(a[[s,"layer_size","reg2","last","train_percent"]])

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


b=np.array([x for x in a[s]])
r= norm_aof(b,727)
a[s]=r

tol=5/100
lower=aof*(1-tol)
upper=aof*(1+tol)
lower_norm= norm_aof(lower,aof)
upper_norm= norm_aof(727*1.1,aof)

sns.pairplot(a[[s,"layer_size","reg2","last","train_percent"]])

aof_ex=[]
for p in range(-100,300):
    aof_rates=[200,400,600]
    aof_pressures=[249.3,192.1,p]
    
    aof_ex.append([p,calc_aof(aof_pressures,aof_rates)])
aof_ex=np.array(aof_ex)