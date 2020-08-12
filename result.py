# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 02:16:34 2020

@author: hsegh
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm

string=[]


#1200-1225: 200 epochs
#1225-1250: 100 epochs
#1250-1275: 500 epochs
#1275-1300: 1000 epochs
for seed in range(2100,2200): #set this range to the range correspondent to the test8
    string.append([])
    with open("plots/"+str(seed)+".txt",'r+') as arq:
        for s in arq:
            string[-1].append(s)


df = pd.DataFrame(columns=["Max Mse","Max Mae","Max mape","Mean Mse","Mean Mae","Mean mape"])

model=[]
for i in range(len(string)):
    model.append([])
    for j in string[i][1].split():
        model[-1].append(float(j))
    #string[i]=string[i][0:-1] # uncomment for AOF
    df.loc[i]=[float(string[i][-6][8:]),float(string[i][-5][8:]),float(string[i][-4][9:]),float(string[i][-3][9:]),float(string[i][-2][9:]),float(string[i][-1][10:])]
    
print("Epochs:"+str(model[0][0]))
data_df=df.describe()
print(data_df)

df=df.to_numpy()
for j in [0,3]:
    plt.scatter([x for x in range(len(df))],[y[j] for y in df])

plt.grid(True)
plt.show()

for j in [2,5]:
    plt.scatter([x for x in range(len(df))],[y[j] for y in df])

plt.grid(True)
plt.ylim(0,100) #uncomment this line to exclude the outliers
#plt.yscale("log")
plt.show()

for j in range(1,len(model[-1])-1):
    plt.scatter([x for x in range(len(model))],[y[j] for y in model])
plt.grid(True)
plt.show()