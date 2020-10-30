# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:38:14 2020

@author: Humberto
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm

import tensorflow as tf

from tensorflow import keras

import sys

def plotResults(results,label,data_shaped):
    plt.scatter([i for i in results],[j for j in label])
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    #plt.plot([100,initial_pressure],[100,initial_pressure],c="magenta")
    #plt.ylim([100,initial_pressure])
    #plt.xlim([100,initial_pressure])
    plt.grid(True)
    plt.show()
    
    plt.scatter([i[-1][0] for i in data_shaped],[j for j in label],c="blue",marker="x")
    plt.scatter([i[-1][0] for i in data_shaped],[j for j in results],c="red",marker="+")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    #plt.ylim([100,initial_pressure])
    plt.show()