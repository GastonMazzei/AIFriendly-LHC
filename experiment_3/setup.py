from multiprocessing import cpu_count
from sys import exit
from os import remove

def exiter():
  print('INVALID ANSWER...')
  exit(1)
  return

def intro():
  mssg_0 = """

--------------------------------------------------------------------------------------------------------------------
    ...Hi, welcome to the Tensorflow Intel-Optimized dockerization with hardware-parameters configurator...        |
-------------------------------------------------------------------------------------------------------------------|                                                                                                                   
                 TLDR: You'll have to answer 4 questions before the script begins                                  |
                                                                                                                   |  
--------------------------------------------------------------------------------------------------------------------
LONG TEXT:                                                                                                         |
If you have an Intel core (post 2000-smth ?2012? e.g. i7) AND a non-NVIDIA GPU then the Intel-Optimized-Tensorflow |
is the best way to accelerate your calculations. A modern implementation that does not require messy installations |
is available (and encouraged by Intel themselves) via an OS-Independent method call 'Dockerization'.               |
A non memory-&-CPU constrained out-of-the-box Dockerization may lead to a 100% core-utilization which makes running|
other tasks with the script in the background plainly-impossible.                                                  |
The aim of this script is to (1) allow users manually-set Core & Memory limits, and (2) run pythonic scripts that  |
use Tensorflow making use of the Intel Architecture Optimization                                                   |
--------------------------------------------------------------------------------------------------------------------
* Author isnt related to Google, Nvidia, Intel, Tensorflow or Python. github.com/GastonMazzei for comments or Bugs!
          """

  Q_1 = '\nQuestion 1 of 4: do you want to set limit values for RAM memory? (y/n)'
  m_1 = '(format: gigas)               e.g. "answer: 4.5"'
  Q_2 = '\nQuestion 2 of 4: do you want to set limit values for SWAP memory? (y/n)'
  m_2 = '(format: gigas)               e.g. "answer: 4.5"'
  Q_3 = '\nQuestion 3 of 4: do you want to set limit values for Number Of Cores? (y/n)'
  m_3 = '(format: number)               e.g. "answer: 2"'
  Q_4 = '\nQuestion 4 of 4: do you want to set limit values for Power per Core (%)? (y/n)'
  m_4 = '(format: percentage)               e.g. "answer: 60"'
  Q_v = [Q_1, Q_2, Q_3, Q_4]
  m_v = [m_1, m_2, m_3, m_4]
  info_names = ['ram', 'swap', 'coresn', 'coresp']
  info = {}

  def request_1(q):
    print(q)
    answ = input("Your answer: ")    
    #print(answ)
    return answ
  
  def request_2(m):
    print(m)
    answ = input("Your answer: ")    
    #print(answ)
    return answ

  def protector(answ):
    if answ.lower() in ['y','1','yes','true']:
      return True
    elif answ.lower() in ['n','0','no','false']:  
      return False
    else: 
      exiter()
      return
  
  print(mssg_0)
  for x in range(4):
    if protector(request_1(Q_v[x])):
      info[info_names[x]] = request_2(m_v[x])  
  return info

def set_manually(**kwargs):
  txt_0 = """
version: '2.4'
services:
  lhc_pipeline:
    image: intel/intel-optimized-tensorflow
    build:
      context: .
      dockerfile: Dockerfile-lhc_pipeline
    volumes:
      - "./:/workdir"
    #mem_limit: 4000m #RAM
    #memswap_limit: 4000m #RAM+SWAP
    #cpu_percent: 50 #WINDOWS
    #cpu_count: 4 #WINDOWS
    #cpus: 2 #LINUX
    #cpuset: 0,1,2,3 #LINUX
    restart: always
volumes:
    uploads:
  """
  txt_1 = txt_0.split('\n')
  if 'ram' in kwargs.keys():
    txt_1[10] = f'    mem_limit: {int(1000*float(kwargs["ram"]))}m'
    if 'swap' in kwargs.keys():
      txt_1[11] = f'    memswap_limit: {int(1000*float(kwargs["ram"])+1000*float(kwargs["swap"]))}m'
  else:
    if 'swap' in kwargs.keys():
      print('cannot limit swap without limiting RAM... swap restriction will be ignored')
  if 'coresn' in kwargs.keys():
    txt_1[13] = f'    cpu_count: {int(kwargs["coresn"])}'
    txt_1[15] = f'    cpuset: {str(list(range(int(kwargs["coresn"]))))[1:-1].replace(" ","")}' 
    if 'coresp' in kwargs.keys(): 
      txt_1[12] = f'    cpu_percent: {int(kwargs["coresp"])}'
      txt_1[14] = f'    cpus: {round(float(kwargs["coresp"])/100*float(kwargs["coresn"]),2)}'
  elif 'coresp' in kwargs.keys():
    txt_1[12] = f'    cpu_percent: {int(kwargs["coresp"])}'
    txt_1[14] = f'    cpus: {round(float(kwargs["coresp"])/100*cpu_count(),2)}'
  try: remove('docker-compose.yml')
  except FileNotFoundError: pass
  f = open('docker-compose.yml','w')
  for line in txt_1:
    f.write(line)
    f.write('\n')
  f.close()
     



set_manually(**intro())
print('\n\nSUCCESS: DEFINED THE PARAMETERS!')
print('\nLast But Not Least:\n___how to run your Python Program?__\nInstead of running "python script.py" in thecommand line you must:\n'\
      '----\n(1) rename the script as "main.py" \n(2) run "bash run.sh"\n----\n...and VIOILA... the script '\
      'will be running with Intel-Tensorflow Hardware Optimizations under the selected memory & CPU constraints!')
print('\n\nPD: only bug known:: keras.backend.clear_session() is not releasing RAM')
# CHECK MKL IS ON
#docker-compose run IOP bash task.sh --rm
#task.sh is 
#python -c "import tensorflow; print(tensorflow.pywrap_tensorflow.IsMklEnabled())"






























