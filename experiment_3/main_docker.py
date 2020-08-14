import pipeline_kernel
import datetime
from tensorflow.keras.backend import clear_session
from keras import backend
from random import choice
from sklearn import preprocessing
from numpy.random import randint, uniform

epochs_list =   [200]#list(randint(150,150,2000,dtype='int')) 
batch_list =  list(randint(10,60,2000,dtype='int')) 
neurons_list =  list(randint(1,30,2000,dtype='int')) 
scaler_list = [preprocessing.StandardScaler() ] #preprocessing.MinMaxScaler()], preprocessing.StandardScaler()]
paradigm_list = [0]#,1,2]
#signal_noise_ratio_train_list = [0.5,0.02]
#signal_noise_ratio_val_list   = [0.5,0.02]
#signal_noise_ratio_test_list  = [0.5]
activation_list = ['relu','elu']#, 'elu', 'exponential'] 
L = 700
for i in range(L):
  (A_,B_,C_) = [(0.5,0.5,0.5),(0.5,0.02,0.02),(0.02,0.02,0.02)][choice(list(range(3)))]
  print(f'Los train,val,test elegidos son: ',A_,B_,C_)
  a = datetime.datetime.now()
  try: 
    pipeline_kernel.process_ssc(epochs=choice(epochs_list),
                                 neurons=choice(neurons_list),
                                 batch=choice(batch_list),
                                 scaler=choice(scaler_list),
                                 paradigm=choice(paradigm_list),
                                 signal_noise_ratio_train=A_,
                                 signal_noise_ratio_val=B_,
                                 signal_noise_ratio_test=C_,
                                 activation=choice(activation_list),)
  except Exception as ins: 
    print('UNEXPECTED ERROR at sequi_sin_cuadrinorm. didnt save more info than this:')
    print('\n\n\n\n\n',ins.args,'\n\n\n\n\n')
  clear_session()
  b = datetime.datetime.now()
  c = b - a
  print('ended iteration SSC',i,f' of {L} in {c.seconds + round(c.microseconds/1E6,2)} seconds')

  #a = datetime.datetime.now()
  try:
    pipeline_kernel.process_scc(epochs=choice(epochs_list),
                                 neurons=choice(neurons_list),
                                 batch=choice(batch_list),
                                 scaler=choice(scaler_list),
                                 paradigm=choice(paradigm_list),
                                 signal_noise_ratio_train=A_,
                                 signal_noise_ratio_val=B_,
                                 signal_noise_ratio_test=C_,
                                 activation=choice(activation_list),)
  except Exception as ins: 
    print('UNEXPECTED ERROR at sequi_con_cuadrinorm. didnt save more info than this:')
    print('\n\n\n\n\n',ins.args,'\n\n\n\n\n')
  clear_session()
  b = datetime.datetime.now()
  c = b - a
  print('ended iteration SCC',i,f' of {L} in {c.seconds + round(c.microseconds/1E6,2)} seconds')

  a = datetime.datetime.now()
  try: 
    pipeline_kernel.process_msc(epochs=choice(epochs_list),
                                 neurons=choice(neurons_list),
                                 batch=choice(batch_list),
                                 scaler=choice(scaler_list),
                                 paradigm=choice(paradigm_list),
                                 signal_noise_ratio_train=A_,
                                 signal_noise_ratio_val=B_,
                                 signal_noise_ratio_test=C_,
                                 activation=choice(activation_list),)
  except Exception as ins: 
    print('UNEXPECTED ERROR at manu_sin_cuadrinorm. didnt save more info than this:')
    print('\n\n\n\n\n',ins.args,'\n\n\n\n\n')
  clear_session()
  b = datetime.datetime.now()
  c = b - a
  print('ended iteration MSC',i,f' of {L} in {c.seconds + round(c.microseconds/1E6,2)} seconds')

