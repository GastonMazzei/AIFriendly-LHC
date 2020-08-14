import numpy as np
import pandas as pd
# Imports de estadistica
from uuid import uuid4
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
# Imports para las redes de Keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import backend
from AIMODEL import *
from math import ceil
from AIMODEL import AI_model

def process_ssc(**kwargs):
  process('sequi_sin_cuadrinorm', **kwargs)
def process_scc(**kwargs):
  process('sequi_con_cuadrinorm', **kwargs)
def process_msc(**kwargs):
  process('manu_sin_cuadrinorm', **kwargs)

def process(flag, **kwargs): 
  print('epochs are ',kwargs.get('epochs',100))
  s = kwargs.get('scaler', preprocessing.MinMaxScaler(),)
  dataset = dataset_initializer(kwargs.get('signal_noise_ratio_train', 0.5),
                                kwargs.get('signal_noise_ratio_val', 0.5),
                                kwargs.get('signal_noise_ratio_test', 0.5),
                                s, kwargs.get('paradigm',0), flag)
  
  
  experiment = AI_model(dataset, **kwargs)
  # here it prints that it has built a network under SPECS 
  # AND it can raise an error "CorruptedDatabase" si no 
  # se condice la "signal_noise_ratio_blabla" que queremos
  # y que tiene la database...
  
  # shared converters
  name_the_scaler = {str(preprocessing.MinMaxScaler()):0, 
                    str(preprocessing.StandardScaler()):1,}
  name_the_activation = {'relu':0, 'elu':1, 'exponential':2}  

  # do the experiment and save data!
  try: 
    experiment.trainModel()
    localdict = {
                'epochs': experiment.epochs,
                'batch': experiment.batch,
                'neurons': experiment.neurons,
                'activation': name_the_activation[experiment.activation],
                'signal_noise_ratio_train': experiment.signal_noise_ratio_train,
                'signal_noise_ratio_val': experiment.signal_noise_ratio_val,
                'signal_noise_ratio_test': experiment.signal_noise_ratio_test,
                'X_train_len': experiment.X_train_len,
                'X_val_len': experiment.X_val_len,
                'X_test_len': experiment.X_test_len,
                'auc': experiment.auc,
                'soft_auc': experiment.soft_auc, 
                'f1': experiment.f1,
                'test_acc': experiment.confidence,
                'training_accuracy_mean': experiment.training_accuracy_mean ,
                'validation_accuracy_mean': experiment.validation_accuracy_mean ,
                'training_accuracy_std' : experiment.training_accuracy_std ,
                'validation_accuracy_std': experiment.validation_accuracy_std ,
                'training_loss_mean' : experiment.training_loss_mean ,
                'validation_loss_mean' : experiment.validation_loss_mean ,
                'training_loss_std': experiment.training_loss_std ,
                'validation_loss_std': experiment.validation_loss_std,
                'scaler': name_the_scaler[str(s)],
                'paradigm': kwargs.get('paradigm',0),
                            }
    append_to_database(flag, **localdict)
    append_to_others(flag, experiment, **localdict) # NEW! 
  except Exception as ins:
    print('training failed with error code: ',ins.args)
    kwargs['scaler'] = name_the_scaler[str(s)]
    kwargs['activation'] = name_the_activation[kwargs['activation']]
    append_to_error(flag, **kwargs)

  # FINALLY: release memory please
  # currently not working inside docker
  backend.clear_session()
  del(experiment)
  # END
  return

def dataset_initializer(q1, q2, q3, scaler, paradigm, flag):
  c = {'sequi_sin_cuadrinorm':0, 'sequi_con_cuadrinorm':1, 
        'manu_sin_cuadrinorm':0}
  X_train = pd.read_csv(f'{flag}/{flag}_data/train_p{int(100*q1)}_c{c[flag]}_s1_.csv',low_memory=False).astype(np.float32)
  X_val = pd.read_csv(f'{flag}/{flag}_data/val_p{int(100*q2)}_c{c[flag]}_s1_.csv',low_memory=False).astype(np.float32)
  X_test = pd.read_csv(f'{flag}/{flag}_data/test_p{int(100*q3)}_c{c[flag]}_s1_.csv',low_memory=False).astype(np.float32)
  X_train, y_train = X_train.iloc[:,:-1].to_numpy(), X_train.iloc[:,-1].to_numpy() 
  X_val, y_val = X_val.iloc[:,:-1].to_numpy(), X_val.iloc[:,-1].to_numpy()
  X_test, y_test = X_test.iloc[:,:-1].to_numpy(), X_test.iloc[:,-1].to_numpy()

  if paradigm==2:
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)
    return [X_train, X_val, X_test, 
          y_train, y_val, y_test]
  elif paradigm==1:
    scaler.fit(X_train)    
  else:
    scaler.fit(np.concatenate((X_train,X_val,X_test)))
  X_train = scaler.transform(X_train)
  X_val = scaler.transform(X_val)
  X_test = scaler.transform(X_test)
  return [X_train, X_val, X_test, 
          y_train, y_val, y_test]

def append_to_database(flag, **kwargs):
  for x in kwargs.keys(): kwargs[x] = [kwargs[x]]
  with open(flag+'/results_'+flag+'.csv', 'a') as f:
    pd.DataFrame(kwargs).to_csv(f, header=False, index=False)
  return

def append_to_others(flag, experiment, **kwargs):
  q = {}
  vectors = [   
             experiment.vector_roc,
             experiment.vector_hardroc,
             experiment.vector_accuracy,
             experiment.vector_accuracyv,
             experiment.vector_loss,
             experiment.vector_lossv,   
               ]
  for x in kwargs.keys(): q[x] = [kwargs[x]] 
  t = pd.concat([pd.DataFrame(q),pd.DataFrame({'id':[uuid4()]})],axis=1)
  with open(flag+'/idmap_'+flag+'.csv', 'a') as f:
    t.to_csv(f, header=False, index=False)
  with open(flag+'/vectors_'+flag+'.txt', 'a') as f:
    f.write(f'ID: {t["id"].to_numpy()}\n')
    for L in vectors:
      f.write(str(L)+'\n')
  return


  for x in kwargs.keys(): kwargs[x] = [kwargs[x]]
  with open(flag+'/results_'+flag+'.csv', 'a') as f:
    pd.DataFrame(kwargs).to_csv(f, header=False, index=False)
  return


def append_to_error(flag, **kwargs):
  for x in kwargs.keys(): kwargs[x] = [kwargs[x]]
  with open(flag+'/errors_'+flag+'.csv', 'a') as f:
    pd.DataFrame(kwargs).to_csv(f, header=False, index=False)
  return
