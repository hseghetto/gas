# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:05:01 2020

@author: hsegh
"""


import pandas as pd
import seaborn as sns

a=pd.read_csv("results.txt",sep=";")  
sns.pairplot(a[["Mean_mse","layer_size","reg2","sqrp","last"]])
