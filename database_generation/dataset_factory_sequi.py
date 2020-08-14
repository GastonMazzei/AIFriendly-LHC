import pandas as pd
import numpy as np
from itertools import combinations
from os import listdir
import os

def make_dataset(p_train, p_val, p_test, cuadrinorm = False,**kwargs):
    # (0) MACRO_OVERWRITE P_TRAIN == 0.5 
    #p_train = 0.5
    #DISABLE TRAIN LOOP
    # (es el mas costoso
    # computacionalmente)
    DTL = False 
   
    # (1) Define random seed and dir
    seed = kwargs.get('seed',93650115)
    dire = kwargs.get('dire',False)

    # (2) Define useful functions
    #
    # (2.a) make the cuadrinorm of P_1+P_2 
    #       con signature (-,+,+,+) 
    def spit_cuadrinorm(dft):
        return (-(dft[0]+dft[4])**2 + (dft[1]+dft[5])**2 
               + (dft[2]+dft[6])**2 + (dft[3]+dft[7])**2)

    # (2.b) mix signal with background
    #       under "key" (train,val,test)
    #       default values and "q" mix ratio 
    def mix_p_concentrated(data,key,q,**kwargs):
        # A,signal B,background
        A = data[0] 
        B = data[1] 
        # filtering constraint
        if len(A)==len(B):
            L = len(A)
        else:
            print('script not ready for different-sized'\
                  ' signals and background') 
            raise ValueError
        # def default values as a function of key
        if key=='train': Ln = kwargs.get(
                           'trainingset_length',
                                          28000,)
        elif key=='test': Ln = kwargs.get(
                            'testingset_length',
                                           3000,)
        else: Ln = kwargs.get(
                         'validationset_length',
                                           3000,)
        # main          
        C = pd.concat([A[:int(Ln*q)], B[:int(Ln*(1-q))]])
        if seed: return C.sample(frac=1, random_state=seed)
        else: return C.sample(frac=1)

    # (2.c) well-behaved saver: don't 
    #       overwrite if it exists
    def saver(df,tag,optionaldir=False):
        if optionaldir:
            base = os.getcwd() 
            os.chdir(optionaldir)
        if tag+'.csv' in listdir(): 
            print('WARNING: file already existed (no-overwrite-policy)')
        else:  
            df.to_csv(tag+'.csv',index=False)        
            print(f'STATUS: --((successfully saved {tag}))--')
        if optionaldir: os.chdir(base)
        return
    # (3) Load the Data
    if seed:
        signal = pd.read_csv('sequi/signal_sequi.dat',
                                delim_whitespace=True, header=None).sample(frac=1,random_state=seed)
        background = pd.read_csv('sequi/background_sequi.dat', 
                                delim_whitespace=True, header=None).sample(frac=1, random_state=seed)
    else:
        signal = pd.read_csv('sequi/signal_sequi.dat',
                                delim_whitespace=True, header=None).sample(frac=1)
        background = pd.read_csv('sequi/background_sequi.dat', 
                                delim_whitespace=True, header=None).sample(frac=1)
    #------------------------------------DTL intervention-1-------------------------------
    if DTL: 
        signal = signal[14000:]
        background = background[14000:]
    #------------------------------------------------------------------------------------
    #
    # (4) Last col (0-1) tagging and
    #     Optional Cuadrinorm Column
    t_ = 8
    if cuadrinorm:
        signal[t_] = spit_cuadrinorm(signal)
        background[t_] = spit_cuadrinorm(background)
        t_ += 1
    signal[t_] = 1
    background[t_] = 0
    # (5) Mix signal and background under "p_train", "p_val",
    #     "p_test", "kwargs" specs
    #------------------------------------DTL intervention-2-------------------------------
    if DTL: pass
    else:
        training_set_raw = [signal[:14000], background[:14000]]
        training_set = mix_p_concentrated(training_set_raw, 'train', p_train,**kwargs)
    #------------------------------------------------------------------------------------
    validation_set_raw = [signal[-6000:-3000], background[-6000:-3000]]
    testing_set_raw = [signal[-3000:], background[-3000:]]
    validation_set = mix_p_concentrated(validation_set_raw, 'valid', p_val,**kwargs) 
    testing_set = mix_p_concentrated(testing_set_raw, 'test', p_test,**kwargs)

    #------------------------------------DTL intervention-3-------------------------------
    # (6) CORRECTNESS TESTS:
    if DTL:
        V = [validation_set, testing_set] 
        uniqueness_test_DTL(*V)    
        tolerance_test_DTL(*V,p_val,p_test) 
        length_test_DTL(*V) 
    else:
        V = [training_set, validation_set, testing_set] 
        #(6.1): Uniqueness Test
        uniqueness_test(*V)     
        # (6.2): Tolerance Test
        tolerance_test(*V,p_train,p_val,p_test) 
        # (6.3): Length Test
        length_test(*V) 
    #------------------------------------------------------------------------------------

    # (7) SAVE
    #------------------------------------DTL intervention-4-------------------------------
    if not DTL: saver(training_set,f'train_p{int(100*p_train)}_c{int(cuadrinorm)}_s1_',dire)
    #------------------------------------------------------------------------------------
    saver(validation_set,f'val_p{int(100*p_val)}_c{int(cuadrinorm)}_s1_',dire)
    saver(testing_set,f'test_p{int(100*p_test)}_c{int(cuadrinorm)}_s1_',dire)    
    print(f'STATUS: SUCCESS w params p {p_train} {p_val} {p_test}, seed = {seed} , cuadrinorm = {cuadrinorm}')
    return 

def uniqueness_test(a,b,c):
    """
    Check that test,train,val have
    different datapoints under a 
    certain treshold (1%?)
    """
    treshold = 0.01
    for X in combinations([a,b,c],2):
        if sum(pd.concat([X[0],X[1]]).duplicated())/(len(X[0])+len(X[1]))>treshold:
            raise Exception('\n\nBuildingError: uniqueness test failed\n\n')
        else: pass
    print('STATUS: Uniqueness Test Passed!')
    return

def tolerance_test(a,b,c,p1,p2,p3):
    """
    Check that the tagged_proba
    has the correct signal_noise_ratio
    under a certain treshold (1%?)
    """
    # treshold understood as.....
    # deviation from the fraction
    # e.g. 1% means 0.01*0.99 signal 
    # will be tolerated, NOT 0.01+0.01
    treshold = 0.01
    x = [a,b,c]
    y = [p1,p2,p3]
    for i in range(3):
        if (y[i]*(1+treshold) > sum(x[i].iloc[:,-1])/len(x[i]) and
            sum(x[i].iloc[:,-1])/len(x[i]) > y[i]*(1-treshold) ): pass
        else: raise Exception('\n\nBuildingError: tolerance test failed\n\n')         
    print('STATUS: Tolerance Test Passed!')
    return

def length_test(a,b,c):
    """
    Check that train,test,val
    have all the required length 
    i.e. 
    TRAIN = 28k
    VAL = 3k
    TEST = 3k
    """
    if (len(a)==28000 and
        len(b)==3000  and
        len(c)==3000): pass
    else: raise Exception("\n\nBuildingError: length test failed\n\n")
    print('STATUS: Length Test Passed!')
    return

def uniqueness_test_DTL(b,c):
    """
    Check that test,train,val have
    different datapoints under a 
    certain treshold (1%?)
    """
    treshold = 0.01
    for X in combinations([b,c],2):
        if sum(pd.concat([X[0],X[1]]).duplicated())/(len(X[0])+len(X[1]))>treshold:
            raise Exception('\n\nBuildingError: uniqueness test failed\n\n')
        else: pass
    print('STATUS: Uniqueness Test Passed!')
    return

def tolerance_test_DTL(b,c,p2,p3):
    """
    Check that the tagged_proba
    has the correct signal_noise_ratio
    under a certain treshold (1%?)
    """
    # treshold understood as.....
    # deviation from the fraction
    # e.g. 1% means 0.01*0.99 signal 
    # will be tolerated, NOT 0.01+0.01
    treshold = 0.01
    x = [b,c]
    y = [p2,p3]
    for i in range(2):
        if (y[i]*(1+treshold) > sum(x[i].iloc[:,-1])/len(x[i]) and
            sum(x[i].iloc[:,-1])/len(x[i]) > y[i]*(1-treshold) ): pass
        else: raise Exception('\n\nBuildingError: tolerance test failed\n\n')         
    print('STATUS: Tolerance Test Passed!')
    return

def length_test_DTL(b,c):
    """
    Check that train,test,val
    have all the required length 
    i.e. 
    TRAIN = 28k
    VAL = 3k
    TEST = 3k
    """
    if (len(b)<3005  and len(b) >2995 and
        len(c)<3005 and len(c)>2995): pass
    else: raise Exception("\n\nBuildingError: length test failed\n\n")
    print('STATUS: Length Test Passed!')
    return

#listas = [x/100 for x in range(5,100,5)]
#listas = [0.01,0.02,0.03,0.04] + listas
#for tp_ in listas:
#    for vp_ in listas:
#        make_dataset(0.5,vp_,tp_,True,dire='sequi_con_cuadrinorm')
make_dataset(0.5,0.2,0.2,True,dire='sequi_sin_cuadrinorm')










