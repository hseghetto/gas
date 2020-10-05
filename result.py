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

def CalcAOF(pressures,flow_rates):
    initial_pressure=300
    pressures=(initial_pressure**2-np.array(pressures)**2)
    flow_rates=np.array(flow_rates)
    
    model=np.polyfit(flow_rates, pressures, 1)
    
    
    aof = ((initial_pressure**2-1.033**2)-model[1])/model[0]
    
    return aof

#1200-1225: 200 epochs
#1225-1250: 100 epochs
#1250-1275: 500 epochs
#1275-1300: 1000 epochs

for run in range(12):    
    for seed in range(10): #set this range to the range correspondent to the test8
        string.append([])
        with open("plots/0"+str(run)+str(seed)+".txt",'r+') as arq:
            for s in arq:
                string[-1].append(s)
    
    df = pd.DataFrame(columns=["Max Mse","Max Mae","Max mape","Mean Mse","Mean Mae","Mean mape"])
    
    model=[]
    final_pressure=[]
    aof=True
    
    for i in range(len(string)):
        model.append([])

        if(aof==True):
            final_pressure.append(float(string[i][-1][15:]))

    pressures=[249.3,192.1,120.7]
    
    flow=[200,400,600]
    
    if(aof==True):
        
        real_aof=CalcAOF(pressures,flow)
        
        aof_results=[]
        
        for i in final_pressure:
            pressures[-1]=i
            #print(pressures)
            aof_results.append(CalcAOF(pressures, flow))
            
        plt.scatter([x for x in range(len(aof_results))],[y for y in aof_results])
        for x in range(len(aof_results)):
            plt.scatter(x,real_aof, c="orange")
        plt.show()
