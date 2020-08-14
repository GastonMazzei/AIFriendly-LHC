import pandas as pd
import numpy as np
import os

#--------------------------------------------------------------------------.
#                 FILTERING ALGORITHM                                      |
#                                                                          |
#--------------------------------------------------------------------------.
# Condition 1: "good accuracy"                                             |
#                                                                          |
# "keep models with training accuracy > treshold1"                         |
#--------------------------------------------------------------------------.
# Condition 2: "no overfitting"                                            |
#                                                                          |
# "keep models with relative difference between                            |
# training accuracy and validation accuracy < treshold2"                   |
#                                                                          |
# i.e.                                                                     |
#|training_accuracy - validation_accuracy| < treshold2 * training_accuracy |
#                                                                          |
#--------------------------------------------------------------------------.

# APPLY THIS TO FILTER
def filter(d,treshold1,treshold2):
  print('warning: "filter" is overwritting builtin function')
  l0 = len(d)
  d = d[(d['training_accuracy_mean']-d['training_accuracy_std']).between(treshold1,1) & 
               ((d['validation_accuracy_mean']-d['validation_accuracy_std'])/
                  (d['training_accuracy_mean']+d['training_accuracy_std'])).between(1-treshold2,1+treshold2)]
  print('Applying filter 1 (valid) contracted the database from ',l0,' to ',len(d))
  return d

# APPLY THIS TO RETRIEVE FILTERED
def anti_filter_1(d,treshold1,treshold2):
  l0 = len(d)
  d = d[~((d['training_accuracy_mean']-d['training_accuracy_std']).between(treshold1,1) & 
               ((d['validation_accuracy_mean']-d['validation_accuracy_std'])/
                  (d['training_accuracy_mean']+d['training_accuracy_std'])).between(1-treshold2,1+treshold2))]
  print('Applying anti-filter 1 (valid) contracted the database from ',l0,' to ',len(d))
  return d


