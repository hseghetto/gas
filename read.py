# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:05:01 2020

@author: hsegh
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


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

# old results
# Results = pd.read_csv("results 19-11/results.txt",sep=";",index_col=0)
# Results = pd.read_csv("results 09-12/results.txt",sep=";",index_col=0)
# Results = pd.read_csv("results 11-12/results.txt",sep=";",index_col=0)  

Results = pd.read_csv("results/results.txt",sep=";",index_col=0)
Results_copy = Results.copy()

Results.drop(["hour","last","tr1","tr2"],axis=1,inplace=True)

seq = list(Results.sequence.unique())

Results["Median_p_error"] = 0
Results["Mean_p_error"] = 0
Results["Median_a_error"] = 0
Results["Mean_a_error"] = 0

def erro_press(x):
    tipo = x.sequence
    target = 0
    
    if("IM" in tipo):
        target = 120.7
    elif("DS" in tipo):
        target = 95.8
    elif("IS" in tipo):
        target = 238.2
    if(x.sqrp == 1 ):
        target = target**2/300
    x.Median_p_error = (x.Median_aof - target)/target*100
    x.Mean_p_error = (x.Mean_aof - target)/target*100
    
    return x

def erro_aof(x):
    tipo = x.sequence
    target = 0
    
    median_press = x.Median_aof
    mean_press = x.Mean_aof
    if(x.sqrp):
        median_press = (max(x.Median_aof,0)*300)**0.5
        mean_press = (max(x.Mean_aof,0)*300)**0.5
        
    if("IM" in tipo):
        target = 718.3
        median = calc_aof([249.3,192.1,median_press],[200,400,600])
        mean = calc_aof([249.3,192.1,mean_press],[200,400,600])  
    elif("DS" in tipo):
        target = 666.1
        median = calc_aof([249.3,185.5,median_press], [200,400,600])
        mean = calc_aof([249.3,185.5,mean_press], [200,400,600])
    elif("IS" in tipo):
        target = 747.4
        median = calc_aof([median_press,181.5,127.1], [200,400,600])
        mean = calc_aof([mean_press,181.5,127.1], [200,400,600])        
        
    x.Median_a_error = (median - target)/target*100
    x.Mean_a_error = (mean - target)/target*100
    
    return x

Results = Results.apply(erro_press, axis=1)
Results = Results.drop( Results.loc[(Results.Median_aof<0)|(Results.Mean_aof<0)].index)
Results = Results.apply(erro_aof, axis=1)

def check(x, s):
    if s in x:
        return True
    return False
Results_IM = Results.loc[Results["sequence"].apply(check, s="IM")]
Results_SD = Results.loc[Results["sequence"].apply(check, s="DS")]
Results_SI = Results.loc[Results["sequence"].apply(check, s="IS")]

Results["epochs"]=Results["epochs"].apply(lambda x: int(x[1:-1]))
Results["epochs_1"]=Results["epochs_1"].apply(lambda x: int(x[1:-1]))

# sns.pairplot(Results[["reg2","Median_p_error","Mean_p_error","Median_a_error","Mean_a_error","sqrp"]],hue = "sqrp")
# sns.pairplot(Results[["layer_size","Median_p_error","Mean_p_error","Median_a_error","Mean_a_error","sqrp"]],hue = "sqrp")
sns.pairplot(Results[["epochs","Median_p_error","Mean_p_error","Median_a_error","Mean_a_error","sqrp"]],hue = "sqrp")

Results["sequence"] = Results["sequence"].apply(lambda x: x[0:2])

# sns.pairplot(Results[["reg2","Median_p_error","Mean_p_error","Median_a_error","Mean_a_error","sequence"]],hue = "sequence")
# sns.pairplot(Results[["layer_size","Median_p_error","Mean_p_error","Median_a_error","Mean_a_error","sequence"]],hue = "sequence")
sns.pairplot(Results[["epochs","Median_p_error","Mean_p_error","Median_a_error","Mean_a_error","sequence"]],hue = "sequence")

'''
print("====="*20)
print("full dataset")
print(Results.shape)
sns.pairplot(Results[["Variance","Median_aof","Mean_aof","Mean_mse","Mean_mape"]])
plt.show()
sns.pairplot(Results[["Variance","Median_aof","trans_size","epochs_1","epochs"]])
plt.show()
print("====="*20)

print("====="*20)
print("0<aof<250")
Results=Results.loc[(Results["Mean_aof"]>0) & (Results["Mean_aof"]<250)]
Results=Results[(Results["Median_aof"]>0) & (Results["Median_aof"]<250)]

sns.pairplot(Results[["Variance","Median_aof","Mean_aof","Mean_mse","Mean_mape"]])
plt.show()
sns.pairplot(Results[["Variance","Median_aof","trans_size","epochs_1","epochs"]])
plt.show()
print("====="*20)

print("====="*20)
print("100<aof<150")
Results=Results.loc[(Results["Mean_aof"]>100) & (Results["Mean_aof"]<150)]
Results=Results[(Results["Median_aof"]>100) & (Results["Median_aof"]<150)]

# sns.pairplot(Results[["Variance","Median_aof","Mean_aof","Mean_mse","Mean_mape"]])
# plt.show()
# sns.pairplot(Results[["Variance","Median_aof","trans_size","epochs_1","epochs"]])
# plt.show()
print("====="*20)

# =========================
# lista = os.listdir("Results/")
# lista.pop()

# comp = pd.read_csv("Results/"+lista[-1],skiprows=[0],sep=";")
# lista.pop()

# for s in lista:
#     aux = pd.read_csv("Results/"+s,skiprows=[0],sep=";")
#     comp = pd.concat([comp,aux])
    
# sns.pairplot(comp[["Train_MAE","Train_MSE","Train_MAPE"]])
# comp = comp.loc[(comp.AOF>0) &(comp.AOF<250)]
# sns.pairplot(comp[["Train_MAE","Train_MSE","Train_MAPE"]])
# # sns.jointplot(comp["Train_MAE"],comp["Train_MSE"],kind="reg",stat_func=r2)
# sns.pairplot(comp[["Train_MSE","Val_MSE","Trans_MSE","Test_MSE","Pred_MSE"]])

# for i in ["Train_MSE","Val_MSE","Trans_MSE","Test_MSE","Pred_MSE"]:
#     for j in ["Train_MSE","Val_MSE","Trans_MSE","Test_MSE","Pred_MSE"]:
#         if (i!=j):
#             sns.jointplot(comp[i],comp[j],kind="reg",stat_func=r2)
#             pass
#         pass
'''
