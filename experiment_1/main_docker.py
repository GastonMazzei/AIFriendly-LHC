import pipeline_kernel
import datetime
from tensorflow.keras.backend import clear_session
from keras import backend
from random import choice
from sklearn import preprocessing
from numpy.random import randint, uniform


epochs_list =   list(randint(40,400,1000,dtype='int')) 
batch_list =  list(randint(5,400,1000,dtype='int')) 
neurons_list =  list(randint(1,50,200,dtype='int')) 
scaler_list = [preprocessing.MinMaxScaler(), preprocessing.StandardScaler()]
paradigm_list = [0,1,2]
signal_noise_ratio_train_list = uniform(0.01,0.99,100)
signal_noise_ratio_val_list   = uniform(0.01,0.99,100)
signal_noise_ratio_test_list  = uniform(0.01,0.99,100)
activation_list = ['relu', 'elu', 'exponential'] 

L = 5
for i in range(L):
  a = datetime.datetime.now()
  try: pipeline_kernel.process(epochs=choice(epochs_list),
                                 neurons=choice(neurons_list),
                                 batch=choice(batch_list),
                                 scaler=choice(scaler_list),
                                 paradigm=choice(paradigm_list),
                                 signal_noise_ratio_train=choice(signal_noise_ratio_train_list),
                                 signal_noise_ratio_val=choice(signal_noise_ratio_val_list),
                                 signal_noise_ratio_test=choice(signal_noise_ratio_test_list),
                                 activation=choice(activation_list),)
  except Exception as ins: 
    print('UNEXPECTED ERROR. didnt save more info than this:')
    print('\n\n\n\n\n',ins.args,'\n\n\n\n\n')
  clear_session()
  b = datetime.datetime.now()
  c = b - a
  print('ended iteration ',i,f' of {L} in {c.seconds + round(c.microseconds/1E6,2)} seconds')

