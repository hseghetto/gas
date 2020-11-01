# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:29:47 2020

@author: hsegh
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as tm

import tensorflow as tf

from tensorflow import keras

import sys



a=pd.read_csv("data/gas_im_extendido_1.txt",sep=" ").to_numpy()

t=np.zeros(len(a))
for x in range(1,len(a)):
    t[x]=a[x,0]-a[x-1,0]