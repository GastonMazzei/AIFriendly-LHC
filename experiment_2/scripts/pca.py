#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:06:25 2020

@author: m4zz31
"""
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
import simple as s
seed = None
if True:
  ending = '.csv'
  dmsc = 'results_manu_sin_cuadrinorm_base'+ending
  dssc = 'results_sequi_sin_cuadrinorm_base'+ending
  dscc = 'results_sequi_con_cuadrinorm'+ending
else:
  dmsc = '50_50.csv'
  dssc = '50_2.csv'
  dscc = '2_2.csv'
dmsc = pd.read_csv(dmsc)
dssc = pd.read_csv(dssc)
dscc = pd.read_csv(dscc)
#---------------------------------------------------------
def filter0(pand,a,b):
  print('old len: ',len(pand))
  temp = pand[((pand['signal_noise_ratio_train']==a)&(pand['signal_noise_ratio_val']==b))]
  print('new len: ',len(temp))
  return temp
def n_eigenvectors(df,n):
  df = df.fillna(0)
  df[df.columns] = StandardScaler().fit_transform(df[df.columns])
  pca = PCA(n_components=n,
          random_state=seed )
  pca.fit(df)
  #return pca.components_ 
  return normalize(pca.components_)


def plotme(df,n,x,axis,string):
  print('EL LARGO ES ',len(df))
  p = n_eigenvectors(df,n)
  axis.bar(range(len(p[x])),abs(p[x]),alpha=0.2)
  if len(p[x])==len(df.columns):
    L = len(df.columns)
    axis.set_xticks(list(range(L)))
    names = [i for i in df.columns]
    #names[-3] = 'signal ratio'
    axis.set_xticklabels(names, rotation=15)
    axis.set_title(string)
  else: print('ERROR: MISMATCHED LEN')
  return 


def main(a,b,m=1,K=False):
    plt.close()
    name = {0:dssc, 1:dscc, 2:dmsc} 
    name_s = {0:'dssc', 1:'dscc', 2:'dmsc'}
    f,ax = plt.subplots(2,3,figsize=(24,14))
    wanted_1 = ['epochs', 'batch', 'neurons', 
                  #'activation',
                  #'auc', 
                  'soft_auc',
                  #'f1', 'test_acc',
                  #'accuracy',
                  #'signal_noise_ratio_val',
                  #'signal_noise_ratio_train',
                   #'signal_noise_ratio_test',
                   'scaler','paradigm',
                   ]
    #wanted_2 = ['signal_noise_ratio_train',]
    wanted = wanted_1
    def main_core(x):
        for q in range(3):
            tempo  = filter0(name[q],a,b)
            tempo = pd.concat([tempo,
                ((tempo['training_accuracy_mean']+
                tempo['validation_accuracy_mean']
                )/2).to_frame(name='accuracy')],axis=1)
            plotme(tempo[wanted],m,x,ax[0,q],
                   'Unfiltered '+name_s[q] +
                        f'\n(eig {K} de {m})')
        for q in range(3):
            try:
                tempo = s.filter_1(name[q],0.9,0.05)
                #tempo = filter0(tempo,a,b)
                tempo = pd.concat([tempo,
                ((tempo['training_accuracy_mean']+
                tempo['validation_accuracy_mean']
                )/2).to_frame(name='accuracy')],axis=1)
                plotme(tempo[wanted],m,x,
                   ax[1,q], 'Filtered '+name_s[q] + 
                            f'\n(eig {K} de {m})')
            except ValueError: print('ignoring the ',q)
        return
    if K: main_core(K-1)
    else:
        for x in range(m):
            main_core(x)
    return

casos = ((0.5,0.5),(0.5,0.02),(0.02,0.02))
(a,b) = casos[2]
main(a,b,1,1)
plt.savefig('temp.png')
        
