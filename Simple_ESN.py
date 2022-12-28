import numpy as np
from scipy.stats import uniform
import scipy.sparse as s


class ESN():
  # VETTORI STATO E INPUT COLONNA
  def __init__(self, rho =0.9, Nr=100, Nu=1, r_density =0.2, i_density =0.1, Ny=1):
    #iperparametri rete
    self.rho = rho
    self.Nr = Nr
    self.Nu = Nu
    self.Ny = Ny
    self.r_density = r_density
    self.i_density = i_density

    #parametri rete
    self.W = self.build_recurrent_matrix()
    self.W_in = self.build_input_matrix()

    # maschera dropout
    self.D = np.zeros(self.W.shape)      ; self.D[:,:] = self.W[:,:] != 0 
    self.D_in = np.zeros(self.W_in.shape); self.D_in[:,:] = self.W_in[:,:] != 0 

    self.x = np.zeros((Nr,1)) # stato corrente

  def build_recurrent_matrix(self):
    wrandom = s.random(self.Nr,self.Nr,density = self.r_density, data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    w = wrandom * ( self.rho / max(np.abs(np.linalg.eigvals(wrandom))) )
    return np.array(w)

  def build_input_matrix(self):
    w_in = s.random( self.Nr , self.Nu+1 , density = self.i_density , data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    w_in = w_in/np.linalg.norm(w_in) # normalizzazione
    return np.array(w_in)

  def compute_state(self, u):
    u = np.vstack( (u,1) )
    z = np.dot( self.W_in, u ) + np.dot( self.W, self.x )
    output = np.tanh( z )
    self.x = output
    return np.copy( output ) # lo restituisco se serve a qualcuno da fuori

  def compute_output(self):
    return np.dot( self.Wout, self.x )

  def compute_output(self,u):
    return np.dot( self.Wout, self.compute_state(u) )

  def train(self,train_x,train_y,wash_seq):
    for d in wash_seq:
      self.compute_state(d) # washout
    c_state = self.compute_state
    s = np.array( list( map( c_state, train_x ) ) ) # shape(len(data),Nr,1)
    s = s.reshape( np.size(train_x), self.Nr ) # shape(len(data),Nr)
    d = train_y.T #shape(len(data),Ny)
    self.Wout = np.transpose( np.dot( np.linalg.pinv(s) , d ) )

  def score(self, X, y, washout=True):
    c_out = self.compute_output
    out = np.array( list( map( c_out, X ) ) ) #shape (len(data),Ny,1)
    out = out.reshape(np.size(X)) # solo output monodimensionale
    wash_len = min(int(len(X)/3),500)
    return np.mean( np.square( y[wash_len:] - out[wash_len:] ) ) 

################################# ESP INDEX #################################

def ESP_Index(esn,data,P,T):
  esn.x = np.zeros( ( esn.Nr,1 ) )
  c_state= esn.compute_state # funzione che calcola lo stato della rete
  s0 = np.array( list( map( c_state, data ) ) ) # orbita x0
  D = np.zeros( P )
  for i in range( P ):
    esn.x = np.random.rand( esn.Nr, 1 )
    si = np.array( list( map( c_state, data ) ) ) # orbita xi
    d = np.zeros( np.size(data,axis=0) - T )
    for t in range( T, np.size(data,axis=0) ):
      d[t-T] = np.linalg.norm( s0[t] - si[t] )
    D[i] = np.mean( d )
  return np.mean( D )


################################# DIMENSIONE SPAZIO STATI #################################
# calcola dimensione spazio stati 
def DSS(esn,data):
  esn.x=np.zeros((esn.Nr,1)) # resetto lo stato iniziale della rete
  c_state= esn.compute_state # funzione che calcola lo stato della rete
  l = np.array( list( map( c_state, data ) ) ) # ogni elemento della lista contiene lo stato della rete al tempo t, t=1:len(data)
  m = l.reshape(np.size(data,axis=0) , esn.Nr).T #shape(100,2000) ogni colonna corrisponde allo stato al tempo t, t=1:len(data)
  cov_m=np.cov(m) # matrice di covarianza per gli stati attraversati dalla rete: ogni unita' ricorrente vista come variabile casuale
  eigs= np.linalg.eigvalsh(cov_m)  # funzione specifica per calcolo autovalori matrici simmetriche, numericamente stabile
  dim = np.sum(eigs)**2/np.sum(np.square(eigs)) # calcolo dimensione effettiva spazio degli stati, come indicato in paper
  return dim

################################# CAPACITA DI MEMORIA #################################

# calcolo capacita' di memoria .
def MC(esn,data):
  c_state = esn.compute_state # funzione che calcola output
  list( map( c_state, data[:1000] ) )

  m1 = np.hstack( list( map(c_state,data[1000:]) ) )
  m = np.dot( esn.Wout, m1).T

  v1 = np.var(data[1000-2*esn.Ny:])
  MC =(np.cov(m[:,0] , data[1000:])[0,0])**2 / (v1 * np.var(m[:,0])) + sum( (np.cov( m[:,k] , data[1000-k:-k])[0,0])**2 / ( v1 * np.var(m[:,k])) for k in range(1,esn.Ny) )
  return MC


###################################################################################################
################################# ALLENAMENTO HEBBIANO ###########################################
###################################################################################################

# calcola modifiche da effettuare a matrice dei pesi w
# x: preattivazione (nel caso di Win input), y: relativa attivazione
def compute_weights(w,x,y,step):
  m = np.diag(np.square(y.reshape(-1)))
  xt = x.reshape(1,-1)
  d = step * ( np.dot(y, xt ) - np.dot(m,w) )
  return d

# esegue epoca di apprendimento hebbiano sulla matrice dei pesi di input. Mantiene densita' matrice allenata
# esn: rete da allenare, train_seq: dati per allenamento, step: learning rate
def train_input(esn,train_seq,step):
  for el in train_seq:
    preact = np.vstack((el,1))
    act = esn.compute_state(el)
    esn.W_in += np.multiply(compute_weights(esn.W_in,preact,act,step) , esn.D_in ) # applico modifiche, maschera mantiene costante la densita' 

# esegue epoca di apprendimento hebbiano sulla matrice dei pesi ricorrenti. Mantiene densita' matrice allenata
# esn: rete da allenare, train_seq: dati per allenamento, step: learning rate
def train_rec(esn,train_seq,step):
  for el in train_seq:
    preact= np.copy(esn.x)
    act = esn.compute_state(el)
    esn.W += np.multiply( compute_weights(esn.W,preact,act,step) , esn.D ) 

# esegue epoca di apprendimento hebbiano su entrambe le matrici contemporaneamente. Mantiene densita' matrici allenate
# esn: rete da allenare, train_seq: dati per allenamento, stepa: learning rates [in_step, rec_step]
def train_both(esn,train_seq,steps):
  in_step = steps[0]
  rec_step = steps[1]
  for el in train_seq:
    pin = np.vstack((el,1))
    preact= np.copy(esn.x)
    act = esn.compute_state(el)
    esn.W_in += np.multiply(compute_weights(esn.W_in,pin,act,in_step) , esn.D_in )
    esn.W += np.multiply( compute_weights(esn.W,preact,act,rec_step) , esn.D ) 

    

###################################################################################################
################################# ESPERIMENTI  ###########################################
###################################################################################################
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def eval_shallow_one(train_data, test_data, tresh=0.0001 , max_epochs = 1000 , step=1e-05 , train_fun=train_input , r_density=0.2 , i_density=0.1 , Nr=100, Nu=1 , mesure_interval=50, rho=0.9):
  test_MCs = [] ; train_MCs = [] 
  train_dims = [] ; test_dims = []
  rhos = []

  esn = ESN( Nu=Nu, Nr=Nr, Ny=2*Nr , rho=rho , r_density=r_density , i_density=i_density )
  # rhos = np.array( max(np.abs(np.linalg.eigvals(esn.W))) )

  train_x = train_data[1000:5000]
  train_y = np.vstack( list( train_data[1000-k:5000-k] for k in range(esn.Ny) ) )

  esn.train(train_x, train_y, train_data[:1000])

  test_MCs.append( MC(esn,test_data) ) ; train_MCs.append( MC(esn,train_data[:2000]) )
  train_dims.append( DSS(esn,train_data) ) ; test_dims.append( DSS(esn,test_data) )
  rhos.append( max(np.abs(np.linalg.eigvals(esn.W))) ) 

  for epoch in range(max_epochs):

    train_fun(esn,train_data,step) 

    if epoch % mesure_interval == 0:
      clear_output(wait=True)
      esn.train(train_x,train_y,train_data[:1000]) 

      test_MCs.append( MC(esn,test_data) ) ; train_MCs.append( MC(esn,train_data[:2000]) )
      train_dims.append( DSS(esn,train_data) ) ; test_dims.append( DSS(esn,test_data) )
      rhos.append( max(np.abs(np.linalg.eigvals(esn.W))) ) 
      print('epoch: ',epoch)
      
    # end if

  # end for
  
  
  # save results
  title=''
  now = datetime.now()
  if type(step) == list:
    title = f'max_epochs_{max_epochs}_{step[0]}_{step[1]}_rdensity_{r_density}_idensity_{i_density}_Nr_{Nr}_Nu_{Nu}_mesInterval_{mesure_interval}_init_rho_{rho}_{now.strftime("%-d-%b-%H:%M:%S")}'
  else:
    string_train_fun = f'{train_fun}'.split(' ')[1]
    title = f'max_epochs_{max_epochs}_{step}_{string_train_fun}_rdensity_{r_density}_idensity_{i_density}_Nr_{Nr}_Nu_{Nu}_mesInterval_{mesure_interval}_init_rho_{rho}_{now.strftime("%-d-%b-%H:%M:%S")}'
  
  pd.DataFrame.from_dict({
      'train_MCs':test_MCs,
      'test_MCs':train_MCs,
      'train_dims':test_dims,
      'test_dims':train_dims,
      'rhos':rhos
  }).to_csv(title)

  fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7,9))
  ax1.set_ylabel('MC') ; ax1.set_xlabel('epoche x ' + str(mesure_interval)) ; ax1.title.set_text('Memory capacity'); 
  ax2.set_ylabel('DSS') ; ax2.set_xlabel('epoche x ' + str(mesure_interval)) ; ax2.title.set_text('Dimensione spazio stati')
  ax1.plot(train_MCs,'b');ax1.plot(test_MCs,'r') ; ax1.legend(['MC train', 'MC val'], loc=4)
  ax2.plot(train_dims,'b');ax2.plot(test_dims,'r') ; ax2.legend(['dim train', 'dim val'], loc=4)
  ax3.set_ylabel('rho') ; ax3.title.set_text('raggio spettrale primo reservoir')
  ax3.plot( rhos,'-r')
  fig.tight_layout()

  print("step: ",step,' mesure_interval: ',mesure_interval, ' stop-treshold: ',tresh, '\n'  )
  print("r_density: ",esn.r_density, 'i_density:',esn.i_density, ' Nr: ',esn.Nr, ' rho: ',esn.rho, '\n'  )
  print("norma pesi in alla fine: ", np.linalg.norm(esn.W_in))
  print("norma pesi rec alla fine: ", np.linalg.norm(esn.W))
  print("raggio spettrale matrice ricorrente alla fine: ",max(np.abs(np.linalg.eigvals(esn.W))) ,'\n' )

