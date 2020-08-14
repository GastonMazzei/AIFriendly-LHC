import contextlib
import io
import sys
import pandas as pd
from simple import filter_1
from numpy import mean,std
import seaborn as sns
import matplotlib.pyplot as plt

def filter_sh(d,treshold1,treshold2):
  l0 = len(d)
  d = d[(d['training_accuracy_mean']-d['training_accuracy_std']).between(treshold1,1) & 
               ((d['validation_accuracy_mean']-d['validation_accuracy_std'])/
                  (d['training_accuracy_mean']+d['training_accuracy_std'])).between(1-treshold2,1+treshold2)]
  #print('Applying filter 1 (valid) contracted the database from ',l0,' to ',len(d))
  return d

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

new = 'signal_noise_ratio_train'
v = [x/100 for x in range(50,100)]
#v=[0.3,0.7]
results = pd.DataFrame(columns=['treshold1','treshold2','corr(auc,train_signal)',
'test_signal','test_signalmax','test_signalmin','aucmean','aucmin','aucmax',
                                                  new,new+'max',new+'min',
                                            'signal_noise_ratio_val','signal_noise_ratio_valmin','signal_noise_ratio_valmax']
                                                            )
d = pd.read_csv('results_sequi_sin_cuadrinorm_base.csv')
#d = d[(d['batch']<=100)]
print('dataset len is ',len(d))

# CUSTOMIZE ME---------------------
d = d[(d['scaler']!=1)]
optmessage = ' (MinMaxScaler)'
#-----------------------------------

LIMIT = 1
#d = d[d['activation']!=2]
v2 = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,]
v2 = [x/10 for x in v2]
print('now dataset len is ',len(d))
for treshold1 in v:
  for treshold2 in v2:
    #with nostdout():
    #  t = filter_1(d,treshold1,treshold2)
    #print('case t1 t2: ',treshold1, treshold2)
    t = filter_sh(d,treshold1,treshold2)
    #if len(t)<7: print('case t1 t2: ',treshold1, treshold2,'PASSING')
    if False: pass
    else:
      if len(t)<15 and treshold2>0.05: 
        if treshold1<LIMIT: 
          LIMIT=treshold1
          print(treshold1,' IS NEW', len(t))
      try: 
        tempdict = {'treshold1':treshold1,'treshold2':treshold2,
         #'corr(auc,train_signal)':t.corr().loc['signal_noise_ratio_train']['soft_auc'],
         #'test_signal':mean(t['signal_noise_ratio_test']),
         #'test_signalmin':min(t['signal_noise_ratio_test']),
         #'test_signalmax':max(t['signal_noise_ratio_test']),
                                  #'aucmean':mean(t['soft_auc']),
                                  #'aucmin':min(t['soft_auc']),
                                  #'aucmax':max(t['soft_auc']),
                                                new:mean(t[new]),
                                                new+'min':min(t[new]),
                                                new+'max':max(t[new]),
                                   #'signal_noise_ratio_val':mean(t['signal_noise_ratio_val']),
                                   #'signal_noise_ratio_valmax':max(t['signal_noise_ratio_val']),
                                   #'signal_noise_ratio_valmin':min(t['signal_noise_ratio_val']),
                                                                               }
        for x in tempdict.keys(): tempdict[x]=[tempdict[x]]
        results = pd.concat([results,pd.DataFrame(tempdict)],axis=0)
      except Exception as ins:
        print(ins.args) 
        print('fail!')

print('len of results is .. ',len(results))
print('ended')

#r = results.dropna()
#c = r.corr()
#f, ax = plt.subplots(figsize=(12,12))
#sns.heatmap(c['train_signal_mean'].sort_values().to_frame(),ax=ax,vmin=-1,vmax=1,cmap=sns.color_palette("coolwarm", 7))
#ax.set_yticklabels(ax.get_yticklabels(),rotation=20)
#f.savefig('temp.png')


#new = 'signal_noise_ratio_train'
r = results
f, ax = plt.subplots(figsize=(12,12))
alpha = 0.05
A = r[r['treshold2'].between(alpha+0.001,1)]

B = r[r['treshold2'].between(0,alpha)]
B['type']='correct'
A['type']='overfitted'
q = pd.concat([A,B])
H = B
if False:
  old = new
  new = 'cool_quotient'
  for q in ['']:
    H[new+q] = H['signal_noise_ratio_val'+q]/H[old+q] 
  for q in ['max','min']:
    H[new+q] = H['signal_noise_ratio_val_'+q]/H[old+q] 
H = pd.concat(
                  [pd.DataFrame({'treshold1': H['treshold1'] , new:H[new], 'type':'mean'}),
                  pd.DataFrame({'treshold1': H['treshold1'] , new:H[new+'max'], 'type':'max'}),
                  pd.DataFrame({'treshold1': H['treshold1'] , new:H[new+'min'], 'type':'min'}),]
                                 )
H.treshold1 = H.treshold1.astype(float)
H[new] = H[new].astype(float)
sns.lineplot(x='treshold1',y=new,data=H,hue='type',ax=ax,)#estimator=None, lw=1,)
ax.set_xlabel('Filter Accuracy Treshold')
namer = {'signal_noise_ratio_train':'Training Signal %'}
try: nama = namer[new]
except: nama = new

ax.set_ylabel(nama)
if False:
  ax.set_yscale('log')
  ax.set_ylim(0.4,1)
  ax.set_yticks([0.5,0.6,0.7,0.8,0.9,1])
  ax.set_yticklabels(['50%','60%','70%','80%','90%','100%'])

try:
  ax.hlines(mean(d[new]),0.5,1,label='\nmin, max \n& mean\n del dataset',color='k',linestyle='dashed')
  ax.hlines(max(d[new]),0.5,1,color='k',linestyle='dashed')#,label='maximo muestral'
  ax.hlines(min(d[new]),0.5,1,color='k',linestyle='dashed')#,label='minimo muestral'
except KeyError: pass

hm = max(H[new])
hmi = min(H[new])
h = abs(hm-hmi)
ax.set_ylim(min([0,hmi-0.05*h]),hm+0.05*h)
ax.fill_betweenx((min([0,hmi-0.05*h]),hm+0.05*h), LIMIT,1,color='red',alpha=0.06,label='\nregion con\n#puntos < 15') 
ax.legend()
try: ax.set_title(f'Promedio (y rango) de {nama} en funcion de la selectividad del filtro (accuracy)\npara los modelos sin-overfitting'+optmessage)
except NameError: ax.set_title(f'Promedio (y rango) de {nama}\nen funcion de la selectividad del filtro (accuracy)\npara los modelos sin-overfitting')

ax.set_xlim(0.5,1)
f.savefig('temp.png')


