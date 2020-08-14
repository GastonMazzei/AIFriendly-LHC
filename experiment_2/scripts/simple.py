import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def column_convert(pand,eje=1,optbool=False):
  ticktranslator = {'test_acc':'Filtro 2', 
                    'validation_accuracy_mean':'Filtro 1',
  'epochs':'Epochs','neurons':'Neurons',
                'signal_noise_ratio_test':'Filtro 2 signal',
                 'signal_noise_ratio_val':'Filtro 1 signal',
                'batch':'Batch size', 'activation':'Activation',
                  'scaler':'Scaler','test_acc':'Filtro 2', 
                 'validation_accuracy_mean':'Filtro 1',
                'training_accuracy_mean':'Train Acc'}
  if optbool:
    return [ticktranslator[x] for x in pand]
  else: return pand.rename(ticktranslator,axis=eje)

def corr(d,ax,f=False,optbool=True,**kwargs):
  names = [
       #'epochs', 
       #'batch', 
       #'neurons', 
       #'activation', 
       'signal_noise_ratio_train',
       #'signal_noise_ratio_val', 
       #'signal_noise_ratio_test', 
       'X_train_len',
       'X_val_len', 
       'X_test_len', 
       'training_accuracy_mean', 
       #'validation_accuracy_mean',
       'training_accuracy_std', 
       'validation_accuracy_std',
       'training_loss_mean', 
       'validation_loss_mean', 
       'training_loss_std',
       'validation_loss_std', 
       #'scaler',
       'paradigm',
       'auc', 
       'soft_auc', 
       'f1', 
       #'test_acc',
               ]
  d = d.drop(names,axis=1)	
  q = d.corr()
  want = kwargs.get('want',['test_acc','validation_accuracy_mean'])
  xticktranslator = {'test_acc':'Filtro 2', 
                    'validation_accuracy_mean':'Filtro 1'}
  yticktranslator = {'epochs':'Epochs','neurons':'Neurons',
                'signal_noise_ratio_test':'Filtro 2 signal',
                 'signal_noise_ratio_val':'Filtro 1 signal',
                'batch':'Batch size', 'activation':'Activation',
                  'scaler':'Scaler','test_acc':'Filtro 2', 
                 'validation_accuracy_mean':'Filtro 1'}
  q = q[want].drop(index=want)
  if kwargs.get('optdrop',False): q = q.drop(index=kwargs.get('optdrop'))
  df = q
  q = df.assign(m=df.mean(axis=1)).sort_values('m').drop('m', axis=1)
  out_name = kwargs.get('output_name','temp')
  cmap = sns.diverging_palette(145, 280, s=85, l=25, n=7)
  if kwargs.get('bins',False): q = (5*q).round(0)/5
  if kwargs.get('mask',None): 
    mask = pd.DataFrame(columns = q.columns, index=q.index)
    mask.loc['signal_noise_ratio_test','validation_accuracy_mean'] = True
    mask.loc['signal_noise_ratio_val','test_acc'] = True
  else: mask=None
  sns.heatmap(q,vmin=-1,vmax=1,
              xticklabels=[xticktranslator[x] for x in want],
              yticklabels=[yticktranslator[y] for y in q.index],
                     cmap=cmap,center=0,ax=ax,annot=True,mask=mask,
              cbar = kwargs.get('cbar',True))
  ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=10)
  ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=15)
  if optbool: f.savefig(out_name+'.png')
  return

def select(d1,d2):
  l0 = len(d1)+len(d2)
  q = pd.concat([d1,d2])[pd.concat([d1,d2]).duplicated()]
  print(f'Final Selector crunched models from {l0} to {len(q)}')
  return q


def filter_1(d,treshold1,treshold2):
  l0 = len(d)
  d = d[(d['training_accuracy_mean']-d['training_accuracy_std']).between(treshold1,1) & 
               ((d['validation_accuracy_mean']-d['validation_accuracy_std'])/
                  (d['training_accuracy_mean']+d['training_accuracy_std'])).between(1-treshold2,1+treshold2)]
  print('Applying filter 1 (valid) contracted the database from ',l0,' to ',len(d))
  return d

def anti_filter_1(d,treshold1,treshold2):
  l0 = len(d)
  d = d[~((d['training_accuracy_mean']-d['training_accuracy_std']).between(treshold1,1) & 
               ((d['validation_accuracy_mean']-d['validation_accuracy_std'])/
                  (d['training_accuracy_mean']+d['training_accuracy_std'])).between(1-treshold2,1+treshold2))]
  print('Applying anti-filter 1 (valid) contracted the database from ',l0,' to ',len(d))
  return d



def filter_2(d,treshold1,treshold2):
  l0 = len(d)
  d = d[(d['training_accuracy_mean']-d['training_accuracy_std']).between(treshold1,1) & 
               (d['test_acc']/
                  (d['training_accuracy_mean']+d['training_accuracy_std'])).between(1-treshold2,1+treshold2)]
  print('Applying filter 2 (test) contracted the database from ',l0,' to ',len(d))
  return d


def anti_filter_2(d,treshold1,treshold2):
  l0 = len(d)
  d = d[~((d['training_accuracy_mean']-d['training_accuracy_std']).between(treshold1,1) & 
               (d['test_acc']/
                  (d['training_accuracy_mean']+d['training_accuracy_std'])).between(1-treshold2,1+treshold2))]
  print('Applying anti-filter 2 (test) contracted the database from ',l0,' to ',len(d))
  return d



def run_zero(t1=0.9,t2=0.05,name='results_sequi_con_cuadrinorm.csv',outname=False):
  d = pd.read_csv(name)
  treshold1 = t1
  treshold2 = t2
  d1 = filter_1(d,treshold1,treshold2)
  d2 = filter_2(d,treshold1,treshold2)
  d3 = select(d1,d2)
  plt.close('all')
  fig = plt.figure()
  ax1 = plt.subplot(221)
  ax2 = plt.subplot(223)
  ax3 = plt.subplot(122)
  corr(d3,ax3,False,False,want=['validation_accuracy_mean','test_acc'],bins=True,mask=True)
  corr(d1,ax1,False,False,want=['validation_accuracy_mean'],
           cbar=False,optdrop=['test_acc','signal_noise_ratio_test'],bins=True)
  corr(d2,ax2,False,False,want=['test_acc'],
           cbar=False,optdrop=['validation_accuracy_mean','signal_noise_ratio_val'],bins=True)
  plt.tight_layout()
  if outname: plt.savefig(outname+'.png')
  else: plt.savefig('algo_0.png')
  return

if __name__=="__main__":

  # first one
  def first(t1=False,t2=False):  
    if t1 and t2:
      run_zero(t1,t2,globalname,f'zero_{int(100*t1)}_id{namekeys[globalname]}')
    else:
      for i in [0.85,]:
        run_zero(i,0.07,globalname,f'zero_{int(100*i)}_id{namekeys[globalname]}')

  # second one
  # FOR (t1,t2,globalname) = (0.9,0.05,i=1)
  def second(t1=0.85,t2=0.05):
    A = 'signal_noise_ratio_val'
    B = 'signal_noise_ratio_test'
    C = 'Signal'
    #t1 = 0.9
    #t2 = 0.05
    #t1 = 0.85
    #t2 = 0.07
    d = pd.read_csv(globalname)
    d1 = filter_1(d,t1,t2)
    d2 = filter_2(d,t1,t2)
    d3 = select(d1,d2)
    d4 = anti_filter_1(d,t1,t2)
    d5 = anti_filter_2(d,t1,t2)
    t = pd.DataFrame({C:d1[A],
                    'Accuracy':d1['validation_accuracy_mean'],
                     'Type': 'Filter 1'
                      })
    t = pd.concat([t, pd.DataFrame({C:d2[B],
                    'Accuracy':d2['test_acc'],
                    'Type': 'Filter 2'
                      })])  
    t = pd.concat([t, pd.DataFrame({C:pd.concat([d3[B],
                                              d3[A]]),
                    'Accuracy':pd.concat([d3['test_acc'],
                                          d3['validation_accuracy_mean']]),
                    'Type': 'Filter 1&2',})])
    t = pd.concat([t, pd.DataFrame({C:pd.concat([d5[B],
                                              d4[A]]),
                    'Accuracy':pd.concat([d5['test_acc'],
                                          d4['validation_accuracy_mean']]),
                    'Type': 'Excluded',
                      })])  
    f, ax = plt.subplots()  
    if False:
      # reversed
      sns.scatterplot(x='Accuracy',y=C,data=t,ax=ax,hue='Type')
      ax.set_xlim(0.7,1)
      ax.set_ylim(0,1.1) 
    else:
      sns.scatterplot(x=C,y='Accuracy',data=t,ax=ax,hue='Type',
                     hue_order = ['Excluded','Filter 1','Filter 2', 'Filter 1&2'])
      ax.set_xlim(0,1)
      ax.plot([x/10 for x in range(1)],[1 for x in range(1)],c='r',lw='1')
      ax.plot([x/10 for x in range(1)],[0.8 for x in range(1)],c='r',lw='1')
      ax.set_xlim(-0.1,1.1)
      ax.set_ylim(0.2,1.4)
    f.savefig(f'signal_id{namekeys[globalname]}.png')
    return 
  
  #Third one
  def third(t1,t2,A,localname,plotname,optbool=False,optstring=False,ax=False):
    #A = 'activation'
    #B = 'signal_noise_ratio_test'
    C = localname
    #t1 = 0.9
    #t2 = 0.05
    #t1 = 0.85
    #t2 = 0.07
    d = pd.read_csv(globalname)
    d1 = filter_1(d,t1,t2)
    d2 = filter_2(d,t1,t2)
    d3 = select(d1,d2)
    d4 = anti_filter_1(d,t1,t2)
    d5 = anti_filter_2(d,t1,t2)
    t = pd.DataFrame({C:d1[A],
                    'Accuracy':d1['validation_accuracy_mean'],
                     'Type': 'Filter 1'
                      })
    t = pd.concat([t, pd.DataFrame({C:d2[A],
                    'Accuracy':d2['test_acc'],
                    'Type': 'Filter 2'
                      })])  
    t = pd.concat([t, pd.DataFrame({C:pd.concat([d3[A],
                                              d3[A]]),
                    'Accuracy':pd.concat([d3['test_acc'],
                                          d3['validation_accuracy_mean']]),
                    'Type': 'Filter 1&2'})])
    t = pd.concat([t, pd.DataFrame({C:pd.concat([d5[A],
                                              d4[A]]),
                    'Accuracy':pd.concat([d5['test_acc'],
                                          d4['validation_accuracy_mean']]),
                    'Type': 'Excluded'
                      })])  
    if ax: saver=False
    else: 
      f, ax = plt.subplots()
      saver = True
    if optbool: sns.scatterplot(x=C,y='Accuracy',data=t,ax=ax)
    else: sns.scatterplot(x=C,y='Accuracy',data=t,ax=ax,hue='Type',
                     hue_order = ['Excluded','Filter 1','Filter 2', 'Filter 1&2'],size='Type')
    #ax.set_xlim(0,1)
    #ax.set_ylim(0,1)
    if not optstring: ax.set_title(f'{fancy_names[globalname]}: \nvalAccuracy VS {C}')
    else: ax.set_title(f'{fancy_names[globalname]}: \nvalAccuracy VS {C} '+optstring)
    if saver: f.savefig(plotname+'.png')
    return

  #Fourth one
  def fourth(t1,t2,A,localname,B,localnameB,plotname,ax=False,):
    #A = 'activation'
    #B = 'signal_noise_ratio_test'
    C = localname
    D = localnameB
    #t1 = 0.9
    #t2 = 0.05
    #t1 = 0.85
    #t2 = 0.07
    d = pd.read_csv(globalname)
    d1 = filter_1(d,t1,t2)
    d2 = filter_2(d,t1,t2)
    d3 = select(d1,d2)
    d4 = anti_filter_1(d,t1,t2)
    d5 = anti_filter_2(d,t1,t2)
    t = pd.DataFrame({C:d1[A],
                    D:d1[B],
                     'Type': 'Filter 1',
                      })
    t = pd.concat([t, pd.DataFrame({C:d2[A],
                    D:d2[B],
                    'Type': 'Filter 2',
                      })])  
    t = pd.concat([t, pd.DataFrame({C:d3[A],
                    D:d3[B],
                    'Type': 'Filter 1&2',})])
    t = pd.concat([t, pd.DataFrame({C:pd.concat([d4[A],d5[A]]),
                    D:pd.concat([d4[B],d5[B]]),
                    'Type': 'Excluded',
                      })])  
    if not ax: 
      f, ax = plt.subplots()
      saver = True
    else: saver=False
    alf = {'results_manu_sin_cuadrinorm.csv':0.3,
         'results_sequi_sin_cuadrinorm.csv':0.7,
        'results_sequi_con_cuadrinorm.csv':0.7}
    sns.scatterplot(x=D,y=C,data=t,ax=ax,hue='Type',
                     hue_order = ['Excluded','Filter 1','Filter 2', 'Filter 1&2'],alpha='auto',size='Type')
    #ax.set_xlim(0,1)
    #ax.set_ylim(0,1)
    ax.set_title(f'{fancy_names[globalname]}: \n{C} VS {D} ')
    if saver: f.savefig(plotname+'.png')
    return

  def capture_null_model():
    def doit(t):
      d = pd.read_csv(globalname)
      # filter good trained only
      d = d[d['training_accuracy_mean'].between(t,1)]
      q_aux1 = d[(d['test_acc']+d['signal_noise_ratio_test']).abs().between(0.97,1.03)]    
      q_aux2 = d[(d['validation_accuracy_mean']+d['signal_noise_ratio_val']).abs().between(0.97,1.03)] 

      q = pd.concat([ pd.DataFrame({'Signal':q_aux1['signal_noise_ratio_test'],  
                            'Accuracy':q_aux1['test_acc'], 'tag': f'Filter 2 ({len(q_aux1)} samples)'}) ,
                   pd.DataFrame({'Signal':q_aux2['signal_noise_ratio_val'],
             'Accuracy':q_aux2['validation_accuracy_mean'], 'tag':f'Filter 1 ({len(q_aux1)} samples)' }) ])   

    def exit():
      import sys
      sys.exit(1)

    # Abortado al descubrir que training set accuracy filtraba todo
    #transf(q_aux1,ax[0],'Filter 2')
    #transf(q_aux2,ax[2],'Filter 1')  
    def transf(q0,axy,label):
      wanted = ['epochs','neurons','signal_noise_ratio_val','signal_noise_ratio_test','test_acc','validation_accuracy_mean',
                'training_accuracy_mean','scaler','activation']
      q2 = q0[wanted]
      q2 = column_convert(q2)
      wanted = column_convert(wanted,1,True)
      q2 = q2.corr()
      cmap = sns.diverging_palette(145, 280, s=85, l=25, n=7) #palette
      for x in q2.columns: q2.loc[x,x] = 0
      sns.set_style('darkgrid')
      if True:
        q2 = (2*q2).round(0)/2
        q2 = q2.loc[~(q2==0).all(axis=1)]
        q2 = q2.loc[:, (q2 != 0).any(axis=0)]
        q2 = q2.assign(m=q2.mean(axis=1)).sort_values('m').drop('m', axis=1) #order
        mask = q2.isin([0])
        #mask.loc['Filtro 1','Filtro 1 signal'] = True
        #mask.loc['Filtro 1 signal','Filtro 1'] = True
        #mask.loc['Filtro 2','Filtro 2 signal'] = True
        #mask.loc['Filtro 2 signal','Filtro 2'] = True
        sns.heatmap(q2,ax=axy,cmap=cmap,center=0,vmin=-1,vmax=1,annot=True,mask=mask,
               xticklabels=q2.columns,
               yticklabels=q2.index,linecolor='black',linewidths=1)
      axy.set_xticklabels(axy.xaxis.get_majorticklabels(), rotation=25)
      axy.set_yticklabels(axy.yaxis.get_majorticklabels(), rotation=35)   
      axy.set_title(label) 	
      return 

    f, ax = plt.subplots(1,2,figsize=(14,10))    
    sns.scatterplot(x='Signal',y='Accuracy',data=q,ax=ax[1],hue='tag')
    plt.tight_layout()
    f.savefig(f'null_caputre_id{namekeys[globalname]}')
    return 

  # define STUFF
  treshold1 = 0.87
  treshold2 = 0.07
  fancy_names = {'results_manu_sin_cuadrinorm.csv':'LHCO data',
         'results_sequi_sin_cuadrinorm.csv':'PP->h->yy',
        'results_sequi_con_cuadrinorm.csv':'PP->h->yy con M^2',}
  globalnames = ['results_manu_sin_cuadrinorm.csv',
         'results_sequi_sin_cuadrinorm.csv',
        'results_sequi_con_cuadrinorm.csv',
       #'results_sequi_sin_cuadrinorm_base.csv',
       #'results_sequi_con_cuadrinorm_base.csv',
            ]
  namekeys = {}
  i = 1
  for x in globalnames: 
    namekeys[x] = i
    i += 1
  #-------------start script
  foo, axy = plt.subplots(1,3,figsize=(17,7))
  j = 0
  if True:
    for globalname in globalnames:
      loopdata = [('epochs','Epochs',f'epochs_id{namekeys[globalname]}'),
              ('neurons','Neurons',f'neurons_id{namekeys[globalname]}'),
               ('batch','Batch Size',f'batch_id{namekeys[globalname]}'),
               ('activation','Activation',f'activation_id{namekeys[globalname]}'),
               ('scaler','Scaler',f'scaler_id{namekeys[globalname]}'),]
      #first(treshold1,treshold2) #
      #second(treshold1,treshold2) #
      #for i_ in loopdata:
      #  third(treshold1,treshold2,i_[0],i_[1],i_[2])
      #fourth(treshold1,treshold2, 'epochs','Epochs','neurons','Neurons',f'neurons_vs_epochs_id{namekeys[globalname]}',axy[j])
      #first(treshold1,treshold2)
      #capture_null_model()
      third(0.9,0.05,'neurons','Neurons',f'neurons_id{namekeys[globalname]}',False,False,axy[j])
      j += 1
    plt.tight_layout()
    foo.savefig('temp.png')
    print('\nSUCCESS!\n')

  





#------------------THE-END--------------------------------------------------
#
#......basuritas...
#
#f.suptitle(f'Heatmap for {name}"s correlation',fontsize=11,)
#
#sns.relplot(x='signal_noise_ratio_test',y='auc',data=d).savefig('temp.png')
#
# ---custom 
#f, ax = plt.subplots(1,3,)#figsize=(25,15))
#corr(d3,f,ax[1],False,want=['validation_accuracy_mean','test_acc'])
#corr(d1,f,ax[0],False,want=['validation_accuracy_mean'],cbar=False)
#corr(d2,f,ax[2],False,want=['test_acc'],cbar=False)
#for _ in ax:
#  _.set_xlabel(xlabel=_.get_xlabel(),fontsize=30,wrap=True,backgroundcolor='k',color='white')
#  _.set_ylabel(ylabel=_.get_ylabel(),fontsize=20,wrap=True,backgroundcolor='k',color='white')
#  _.tick_params(labelsize=5)
#plt.tight_layout()
#f.savefig('temp.png')
#--end of custom 
#
#----------------------------------------------------------------------------
