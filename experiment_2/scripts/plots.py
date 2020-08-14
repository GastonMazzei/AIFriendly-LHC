import matplotlib.pyplot as plt 
import pandas as pd
import re
import sys
from simple import filter_1
from random import choice
GLOBALNAME = 'sequi_sin'

def initialize():
  f = open(f'vectors_{GLOBALNAME}_cuadrinorm.txt','r')
  texto = ''
  for L in f:
    texto += L+'\n'
  f.close()
  casos = texto.split('ID')
  casos = [casos[2*i+2] for i in range(int(len(casos)/2)-1)]   
  casos_dict = {}
  for x in range(len(casos)):
    casos_dict[x] = casos[x]
  return casos_dict

def rocme(a):
  print('pend!')
  a = re.findall('\[([0-9\,\s\.]*)\]',a)
  a = [[float(x) for x in a[h].split(', ')] for h in range(len(a))]
  return a	
 

def process(casos_dict,y_vec=[4]):
  for y in y_vec:
    casos_dict[y] = casos_dict[y].split('\n')  
    casos_dict[y] = [casos_dict[y][2*i] for i in range(int(len(casos_dict[y])/2))]
    casos_dict[y] = {'id': casos_dict[y][0],
                 'acc': casos_dict[y][3],
                 'accv':casos_dict[y][4],
                 'loss': casos_dict[y][5],
                 'lossv': casos_dict[y][6],
                 'x': list(range(len(casos_dict[y][6]))),
                 'ROC':rocme(casos_dict[y][1]),
                                         }
  
  print('you should know that...')
  print(f'Len ROC: {len(casos_dict[y]["ROC"])}'\
          f'and inner lens.. TPR {len(casos_dict[y]["ROC"][0])}'\
          f' , FPR {len(casos_dict[y]["ROC"][1])}' )
          #f'tresholds {len(casos_dict[y]["ROC"][2])}')
  print(f'Len acc: ',len(casos_dict[y]['acc']))
  print(f'Len accv: ',len(casos_dict[y]['accv']))
  print(f'Len loss: ',len(casos_dict[y]['loss']))
  print(f'Len lossv: ',len(casos_dict[y]['lossv']))

  for s in ['acc','accv','loss','lossv']:
    casos_dict[y][s] = extract_vect(casos_dict[y][s])
  return casos_dict

def extract_vect(v):
  return [float(x) for x in v[1:-1].split(',')]

def pick(one,two,three,four,opt=False):
  d = pd.read_csv(f'idmap_{GLOBALNAME}_cuadrinorm.csv')
  print('contraction from ',len(d))
  d = d[((d['signal_noise_ratio_train']==float(one))&(d['signal_noise_ratio_val']==float(two)))]
  print('to ',len(d),' with constraints train,val signals: ',one,two)
  d = filter_1(d,float(three),float(four))
  q = choice(d.index.to_list())
  print(d.loc[q])
  if not(opt): return q
  else: return d

def aux(df,a,b):
  return df[((df['signal_noise_ratio_train']==a)&(df['signal_noise_ratio_val']==b))]

if __name__=='__main__':
  k = sys.argv[1]
  if k=='-1': 
    try: 
      k = pick(*sys.argv[2:])
      print('picked ',k)
    except Exception as ins:
      print('ERROR ',ins.args,' EXITING')
      sys.exit(1)
  else: k = int(k)
  data = process(initialize(),[k])[k]
  order_plot = {(0,0):'acc',
                (0,1):'accv',
                (1,0):'loss',
                (1,1):'lossv',
                (2,0):'ROC',                
                             }
  f, ax = plt.subplots(3,2,figsize=(20,16))
  for k in order_plot.keys():
    if order_plot[k]=='ROC':
      ax[k].plot(data[order_plot[k]][0],data[order_plot[k]][1])
      ax[k].set_title(order_plot[k])      
    else: 
      ax[k].plot(data[order_plot[k]])
      ax[k].set_title(order_plot[k])
      ax[k].set_ylim(0,1)
  ax[2,1].axis('off')
  f.savefig('temp.png')
