import numpy as np
import pandas as pd
# Imports de estadistica
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
# Imports para las redes de Keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import backend
from tensorflow.autograph.experimental import do_not_convert
class AI_model:
  def __init__(self, tp, **kwargs):
    [X_train, X_val, X_test,
    y_train, y_val, y_test] = tp
    def snr_answer(y_):
      return round(sum(y_)/len(y_),3)
    #---METODOS UTILES PARA LA CLASIFICACION A POSTERIORI
    self.neurons = kwargs.get('neurons',32)
    self.batch = kwargs.get('batch',32)
    self.epochs = kwargs.get('epochs',100)
    self.activation = kwargs.get('activation','relu')
    self.model = Sequential([Dense(self.neurons, activation=self.activation,
                                    input_shape=(len(X_train[0]),)),
                            Dense(self.neurons, activation=self.activation),
                            Dense(1, activation='sigmoid')]) 
    self.model.compile(optimizer='SGD',
            loss='binary_crossentropy',
            metrics=['accuracy'])
    self.signal_noise_ratio_train = snr_answer(y_train)
    self.signal_noise_ratio_val = snr_answer(y_val)
    self.signal_noise_ratio_test = snr_answer(y_test)
    #------------This Seems a Bit Spammy And Drastic jejeje...
    errors = 0
    tolerance = 1 / 100 # porciento
    try:
      if any([1+tolerance < snr_answer(y_train)/kwargs['signal_noise_ratio_train'], 
         snr_answer(y_train)/kwargs['signal_noise_ratio_train'] < 1 - tolerance]): errors += 1
    except KeyError: pass
    try:
      if any([1+tolerance < snr_answer(y_val)/kwargs['signal_noise_ratio_val'],
         snr_answer(y_val)/kwargs['signal_noise_ratio_val'] < 1 - tolerance]): errors += 1
    except KeyError: pass
    try:
      if any([1+tolerance < snr_answer(y_test)/kwargs['signal_noise_ratio_test'], 
          snr_answer(y_test)/kwargs['signal_noise_ratio_test'] < 1 - tolerance]): errors += 1
    except KeyError: pass
    if errors: 
      print(f'ERROR: database encountered {errors} problems')
      try: print(snr_answer(y_train)/kwargs['signal_noise_ratio_train'],kwargs['signal_noise_ratio_train'])
      except KeyError: pass
      try: print(snr_answer(y_val)/kwargs['signal_noise_ratio_val'],kwargs['signal_noise_ratio_val'] )  
      except KeyError: pass
      try: print(snr_answer(y_test)/kwargs['signal_noise_ratio_test'],kwargs['signal_noise_ratio_test'] )      
      except KeyError: pass
      raise Exception('CorruptedDatabase')
    #-----------But better being safe than sorry huh?-------
    #---weird info we also keep.....
    self.X_train_len = len(X_train)
    self.X_val_len = len(X_val)
    self.X_test_len = len(X_test)
    #----------
    self.X_train = X_train
    self.y_train = y_train
    self.X_val = X_val    
    self.y_val = y_val    
    self.X_test = X_test
    self.y_test = y_test
    self.history = None
    self.confidence = None
    print(f'\n\nNetwork summary is: {self.model.summary()}\n\n')
  #@do_not_convert
  def trainModel(self): # Should we use CAMMEL CASE for class_functions? vote NO
    def auc_and_f1(ytrue,ypred): # CHECK
      return (metrics.roc_auc_score(ytrue,ypred), metrics.f1_score(ytrue,ypred)) # CHECK
    self.history = self.model.fit(self.X_train, self.y_train,
        batch_size = self.batch, epochs = self.epochs, verbose=0,
        validation_data=(self.X_val, self.y_val))
    self.confidence = self.model.evaluate(self.X_test, self.y_test, verbose=0)[1]
    #
    self.training_accuracy_mean = np.mean(self.history.history['accuracy'][-10:])
    self.validation_accuracy_mean = np.mean(self.history.history['val_accuracy'][-10:])
    self.training_accuracy_std = np.std(self.history.history['accuracy'][-10:])
    self.validation_accuracy_std = np.std(self.history.history['val_accuracy'][-10:])
    #
    self.training_loss_mean = np.mean(self.history.history['loss'][-10:])
    self.validation_loss_mean = np.mean(self.history.history['val_loss'][-10:])
    self.training_loss_std = np.std(self.history.history['loss'][-10:])
    self.validation_loss_std = np.std(self.history.history['val_loss'][-10:])
    #
    try:  self.soft_auc = metrics.roc_auc_score(self.y_test, self.model.predict(self.X_test))
    except ValueError:
      print('PROBLEM WITH SOFT_AUC... replacing it for "None"')
      self.soft_auc = None
    self.auc, self.f1 = auc_and_f1(self.y_test, self.model.predict_classes(self.X_test) )  # CHECK
    return 

