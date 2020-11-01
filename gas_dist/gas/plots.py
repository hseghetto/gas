# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:38:14 2020

@author: Humberto
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

def plotResults(results,label,data_shaped,initial_pressure=300):
    plt.scatter([i for i in results],[j for j in label])
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.plot([50,initial_pressure],[50,initial_pressure],c="magenta")
    plt.grid(True)
    plt.show()
    
    plt.scatter([i[-1][0] for i in data_shaped],[j for j in label],c="blue",marker="x")
    plt.scatter([i[-1][0] for i in data_shaped],[j for j in results],c="red",marker="+")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    #plt.ylim([100,initial_pressure])
    plt.show()
    
def modelGraphs(hist,x=0.9): #Receives model training history and displays error graphs
    plt.figure(figsize=[2*6.4/1,3*4.8/2])
    P=int(len(hist)*x)
    
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