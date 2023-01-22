import os
import pickle
import numpy as np
import scipy.sparse as s
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import uniform
from IPython.display import display, clear_output

class Reservoir():
  def __init__(self, rho=0.9 , Nu=10, Nr=10,r_density=0.5, i_density=1):
    self.rho = rho
    self.Nr = Nr
    self.Nu = Nu
    self.r_density = r_density
    self.i_density = i_density

    self.W = self.build_recurrent_matrix()
    self.W_in = self.build_input_matrix()  
    
    self.D = np.zeros(self.W.shape)      ; self.D[:,:] = self.W[:,:] != 0 
    self.D_in = np.zeros(self.W_in.shape); self.D_in[:,:] = self.W_in[:,:] != 0 

    self.x = np.zeros((Nr,1)) # stato corrente
  
  def build_recurrent_matrix(self):
    wrandom = s.random(self.Nr,self.Nr,density = self.r_density, data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    w = wrandom * ( self.rho / max(np.abs(np.linalg.eigvals(wrandom))) )
    return np.array(w)
  # end ESN.build_recurrent_matrix

  def build_input_matrix(self):
    w_in = s.random( self.Nr , self.Nu+1 , density = self.i_density , data_rvs=uniform(loc=-1,scale=2).rvs ).todense() # matrice sparsa con valori in distribuzione uniforme tra -1 e 1
    w_in = w_in/np.linalg.norm(w_in) # normalizzazione
    return np.array(w_in)

  def compute_state(self, u):
    u = np.vstack((u,1))
    z = np.dot( self.W_in, u ) + np.dot( self.W, self.x )
    output = np.tanh(z)
    self.x = output
    return np.copy(output) # lo restituisco se serve a qualcuno da fuori
    

class DeepESN():
  # VETTORI STATO E INPUT COLONNA
  def __init__(self, rho =0.9, N=10, Nr=10, Nu=1, Ny=1, r_density=0.5, i_density=1):
    #iperparametri rete
    self.rho = rho
    self.N = N
    self.Nu = Nu
    self.Nr = Nr
    self.Ny = Ny

    self.ress = [None]*N
    self.ress[0] = Reservoir(Nu=Nu, Nr=Nr, r_density=r_density, i_density=i_density, rho=rho)
    for i in range(1,N):
      self.ress[i] = Reservoir(Nu=Nr,Nr=Nr, r_density=r_density, i_density=i_density, rho=rho) 
    
  def compute_state(self,u):
    cu = self.ress[0].compute_state(u) 
    for i in range(1,self.N):
      cu = self.ress[i].compute_state(cu)
    return np.array( list( np.copy(res.x) for res in self.ress ) ) # shape (N,Nr,1)
    
  def compute_output(self): 
    x_c = np.vstack( list( self.ress[i].x for i in range( self.N) ))
    return np.dot( self.Wout, x_c )
  
  def compute_output(self,u): 
    x_c = self.compute_state(u).reshape(-1,1)
    return np.dot( self.Wout, x_c )
  	
  def compute_output_i(self,u,i):
    x_c = self.compute_state(u)[i]
    return np.dot( self.ress[i].Wout, x_c )
  
  # allenamento readout tutta la rete
  def train(self,train_x,train_y,wash_seq):
    for d in wash_seq:
      self.compute_state(d) # washout
    c_state = self.compute_state
    s = np.array( list( map( c_state, train_x ) ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
    s = s.reshape( np.size(train_x) , self.Nr*self.N )
    d = train_y.T #shape(len(data),Nr)
    self.Wout = np.transpose( np.dot( np.linalg.pinv(s) , d ) )

  # allenamento i-esimo readout
  def train(self,train_x,train_y,wash_seq,i): # allena il readout dell'i-esimo reservoir
    for d in wash_seq:
      self.compute_state(d) # washout
    s = np.array( list( self.compute_state(d)[i] for d in train_x ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
    s = s.reshape( np.size(train_x) , self.Nr )
    d = train_y.T #shape(len(data),Nr)
    self.ress[i].Wout = np.transpose( np.dot( np.linalg.pinv(s) , d ) )
      
  def score(self, X, y, washout=True):
    c_out = self.compute_output
    out = np.array( list( map( c_out, X ) ) ) #shape (len(data),Ny,1)
    out = out.reshape(np.size(X)) # solo output monodimensionale
    wash_len = min(int(len(X)/3),500)
    return np.mean( np.square( y[wash_len:] - out[wash_len:] ) ) 
  
  def reset_states(self):
    for res in self.ress:
      res.x = np.zeros((self.Nr,1))

################################# ESP INDEX #################################

# orbita: serie di stati attrvarsati dalla rete durante la sottoposizione di una sequenza di input.
# per reti DeepESN, si prendono gli stati di tutti i reservoir e si concatenano in un unico vettore colonna.
def ESP_Index(esn,data,P,T):
  # orbita x0. shape( len(data), (esn.N*esn.Nr), 1), ovvero un vettore di vettori colonna, ognuno dei quali rappresenta l'i-esimo stato (come concatenzione stati reservoir)
  s0 = np.array( list( esn.compute_state(d).reshape(-1,1) for d in data ) )  
  D = np.zeros( P ) # inizializzo vettore D
  for i in range( P ):
    esn.x = np.random.rand( esn.Nr, 1 ) # setto stato iniziale a vettore randomico
    si = np.array( list( esn.compute_state(d).reshape(-1,1) for d in data ) ) # orbita xi.
    d = np.zeros( np.size(data,axis=0) - T ) # per contenere distanze euclidee tra stati al t-esimo passo
    for t in range( T, np.size(data,axis=0) ):
      d[t-T] = np.linalg.norm( s0[t] - si[t] ) # d[t] contiene distanza tra stati al t-esimo passo, calcolati da stati iniziali diversi
    D[i] = np.mean( d ) # media distanze stati per due stati iniziali diversi
  return np.mean( D ) # a grandi linee media distanze stati per tanti stati iniziali diversi


################################# DIMENSIONE SPAZIO STATI #################################
# calcolo dimensioni spazio stati di ogni reservoir, le mette in un vettore e ritorna quel vettore
def DSS(esn,data): 
  esn.reset_states()
  c_state = esn.compute_state
  l = np.array( list( map( c_state, data ) ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
  dims=[None]*esn.N
  for i in range( esn.N ):
    m_i = l[:,i,:,:].reshape( np.size(data) , esn.Nr ).T # matrice degli stati assunti dall'i-esimo reservoir
    cov_m=np.cov(m_i)
    eigs= np.linalg.eigvalsh(cov_m)
    dims[i] = np.sum(eigs)**2/np.sum(np.square(eigs)) 
  return dims

# calcola dimensione spazio stati di un reservoir
def DSSi(esn,data,i):  
  esn.reset_states()
  l = np.array( list( esn.compute_state(d)[i] for d in data ) ) #shape ( len(data) , esn.Nr , 1 )
  m = l.reshape(np.size(data,axis=0) , esn.Nr).T
  cov_m = np.cov(m) 
  eigs = np.linalg.eigvalsh(cov_m)  
  dim = np.sum(eigs)**2/np.sum(np.square(eigs))
  return dim

def DSSi_all(esn,data):  
  dims = np.zeros(esn.N)
  esn.reset_states()
  c_state = esn.compute_state
  l = np.array( list( map( c_state, data ) ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
  for j in range( esn.N ):
    m_j = l[:,j,:,:].reshape( np.size(data) , esn.Nr ).T # matrice degli stati assunti dall'i-esimo reservoir
    cov_m=np.cov(m_j)
    eigs= np.linalg.eigvalsh(cov_m)
    dims[j] = np.sum(eigs)**2/np.sum(np.square(eigs)) 
  return dims

# dimensione spazio stati globale. Si usa concatenazione stati di ogni reservoir
def glob_DSS(esn,data):  
  esn.reset_states()
  c_state = esn.compute_state
  l = np.array( list( map( c_state, data ) ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
  m = l.reshape( np.size(data) , esn.Nr*esn.N ).T
  cov_m = np.cov( m )
  eigs= np.linalg.eigvalsh( cov_m )
  dim = np.sum( eigs )**2 / np.sum( np.square( eigs ) ) 
  return dim


################################# CAPACITA DI MEMORIA #################################
# allena il readout dell'i-esimo reservoir
def train(self,train_x,train_y,i): 
  s = np.array( list( self.compute_state(d)[i] for d in train_x ) ) #shape ( len(data) , esn.N , esn.Nr , 1 )
  s = s.reshape( np.size(train_x) , self.Nr )
  d = train_y.T #shape(len(data),Nr)
  self.ress[i].Wout = np.transpose( np.dot( np.linalg.pinv(s) , d ) )

# capacita di memoria i-esimo reservoir
def MCi(esn,i,data):
  for d in data[:1000]:
    esn.compute_state(d) # washout

  m = np.array( list( esn.compute_output_i(d,i) for d in data[1000:] ) ).reshape(1000,esn.Ny) # matrice degli yk: uno per colonna

  v1 = np.var(data[1000-esn.Ny:])
  MC =(np.cov(m[:,0] , data[1000:])[0,0])**2 / (v1 * np.var(m[:,0])) + sum( (np.cov( m[:,k] , data[1000-k:-k])[0,0])**2 / ( v1 * np.var(m[:,k])) for k in range(1,esn.Ny) )

  if MC > 100 : MC=0
  return MC

# capacita' di memoria globale, si usa concatenazione degli stati dei reservoir
def glob_MC(esn,data):
  for d in data[:1000]:
    esn.compute_state(d) # washout

  m = np.array( list( esn.compute_output(d) for d in data[1000:]) ).reshape(1000,esn.Ny) # matrice degli yk: uno per colonna

  v1 = np.var(data[1000-esn.Ny:])
  MC =(np.cov(m[:,0] , data[1000:])[0,0])**2 / (v1 * np.var(m[:,0])) + sum( (np.cov( m[:,k] , data[1000-k:-k])[0,0])**2 / ( v1 * np.var(m[:,k])) for k in range(1,esn.Ny) )
  if MC>100:MC=0
  return MC


###################################################################################################
################################# ALLENAMENTO HEBBIANO ###########################################
###################################################################################################

# calcola modifiche da effettuare a matrice dei pesi w secondo regola oja/anti-oja
# x: preattivazione (nel caso di Win input), y: relativa attivazione
def compute_weights(w,x,y,step): 
  m = np.diag(np.square(y.reshape(-1)))
  xt = x.reshape(1,-1)
  d = step * ( np.dot(y, xt ) - np.dot(m,w) )
  return d

################################# ALLENAMENTO SINGOLI RESERVOIR #################################

def train_both(esn,i,train_seq,steps):
  esn.reset_states()
  in_step = steps[0]
  rec_step = steps[1]
  res = esn.ress[i]
  W_in = res.W_in ; D_in = res.D_in
  W = res.W       ; D = res.D
  if i>0:  
    for el in train_seq:
      p = esn.ress[i-1].x # preattivazione per matrice input e' lo stato del reservoir precedente al tempo t
      pin = np.vstack((p,1)) # completa di bias
      preact = res.x # invece la preattivazione per matrice ricorrente e' lo stato di questo reservoir al tempo t.
      act= esn.compute_state(el)[i] # l'attivazione post-sinaptica invece e' data dallo stato di questo reservoir al tempo t+1
      W_in += np.multiply( compute_weights( W_in, pin , act , in_step) , D_in )
      W += np.multiply( compute_weights( W , preact , act , rec_step ) , D ) 
  else:
    for el in train_seq:
      pin = np.vstack((el,1)) # per quanto riguarda il primo reservoir la preattivazione consiste nell'input corrente
      preact = res.x # invece la preattivazione per matrice ricorrente e' lo stato di questo reservoir al tempo t.
      act = res.compute_state(el) # in questo caso non ho bisogno di far passare il flusso per tutta la rete!
      W_in += np.multiply( compute_weights( W_in, pin , act , in_step) , D_in )
      W += np.multiply( compute_weights( W , preact , act , rec_step ) , D ) 

def train_input(esn,i,train_seq,in_step):
  esn.reset_states()
  res = esn.ress[i]
  W_in = res.W_in ; D_in = res.D_in
  if i>0:  
    for el in train_seq:
      p = esn.ress[i-1].x # preattivazione per matrice input e' lo stato del reservoir precedente al tempo t
      pin = np.vstack((p,1)) # completa di bias
      act= esn.compute_state(el)[i] # l'attivazione post-sinaptica invece e' data dallo stato di questo reservoir al tempo t+1
      W_in += np.multiply( compute_weights( W_in, pin , act , in_step) , D_in )
  else:
    for el in train_seq:
      pin = np.vstack((el,1)) # per quanto riguarda il primo reservoir la preattivazione consiste nell'input corrente
      act = res.compute_state(el) # in questo caso non ho bisogno di far passare il flusso per tutta la rete!
      W_in += np.multiply( compute_weights( W_in, pin , act , in_step) , D_in )


def train_rec(esn,i,train_seq,rec_step):
  esn.reset_states()
  res = esn.ress[i]
  W = res.W       ; D = res.D
  if i>0:  
    for el in train_seq:
      preact = res.x # invece la preattivazione per matrice ricorrente e' lo stato di questo reservoir al tempo t.
      act= esn.compute_state(el)[i] # l'attivazione post-sinaptica invece e' data dallo stato di questo reservoir al tempo t+1
      W += np.multiply( compute_weights( W , preact , act , rec_step ) , D ) 
  else:
    for el in train_seq:
      preact = res.x # invece la preattivazione per matrice ricorrente e' lo stato di questo reservoir al tempo t.
      act = res.compute_state(el) # in questo caso non ho bisogno di far passare il flusso per tutta la rete!
      W += np.multiply( compute_weights( W , preact , act , rec_step ) , D ) 

################################# ALLENAMENTO TUTTI I RESERVOIR CONTEMPORANEAMENTE #################################

def glob_train_both(esn,train_seq,steps):
  esn.reset_states()
  in_step = steps[0]
  rec_step = steps[1]
  pins=[None]*esn.N; precs=[None]*esn.N; 
  
  for el in train_seq:
    pins[0] = np.vstack((el,1)) # per quanto riguarda il primo reservoir la preattivazione consiste nell'input corrente
    pins[1:] = list( np.vstack((esn.ress[i-1].x , 1)) for i in range(1,esn.N) ) # invece la preattivazione per matrice ricorrente e' lo stato di questo reservoir al tempo t.
    precs[:] = list( np.copy(esn.ress[i].x) for i in range(esn.N) )
    acts = esn.compute_state(el) 
    for i in range( np.size(esn.ress) ):
      esn.ress[i].W_in += np.multiply( compute_weights( esn.ress[i].W_in, pins[i] , acts[i] , in_step) , esn.ress[i].D_in )
      esn.ress[i].W += np.multiply( compute_weights( esn.ress[i].W , precs[i] , acts[i] , rec_step ) , esn.ress[i].D ) 
 

################################# ALLENAMENTO UN RESERVOIR ALLA VOLTA #################################

def eval_one_reservoir_at_a_time(train_data, test_data, N = 5, Nr=100, Nu=1, rho=0.9, r_density=0.1, i_density=1, max_epochs=10, step=-2e-05 , train_fun=train_rec, mesure_interval=1):
  train_x = train_data[1000:5000]
  train_y = np.vstack( list( train_data[1000-k:5000-k] for k in range(200) ) )

  esn = DeepESN( N=N, Nr=Nr, Ny=2*Nr , rho=rho , r_density=r_density , i_density=i_density )

  for i in range(esn.N):
    for d in train_data[:1000]:
      esn.compute_state(d) # washout
    train(esn,train_x,train_y,i)  #alleno i-esimo readout

  test_dims = np.array( DSSi_all(esn,test_data ) )
  test_MCs = np.array( list( MCi(esn,i,test_data ) for i in range(esn.N) ) )

  train_dims = np.array( DSSi_all(esn,train_data ) )
  train_MCs = np.array( list( MCi(esn,i,train_data[:2000] ) for i in range(esn.N) ) )

  rhos = np.array( max(np.abs(np.linalg.eigvals(esn.ress[0].W))) )
  
  best_Ws, best_W_ins = [None]*esn.N, [None]*esn.N
  best_MCs = [0]*esn.N

  print(test_MCs)

  for i in range( 1 ):
    
    print('res:', i )
    for epch in range(max_epochs):
      clear_output(wait=True)
      print(epch)
      train_fun(esn,i,train_data,step)
      print('hebb train done')
      
      if epch % mesure_interval == 0:
        
        # train a readout for each resevoir
        for j in range(esn.N):
          for d in train_data[:1000]:
            esn.compute_state(d) # washout
          train(esn,train_x,train_y,j)  
        print('readout train done')

        new_MCs = np.array( list( MCi(esn,i,test_data ) for i in range(esn.N) ) ) 
        test_MCs = np.vstack( ( test_MCs, new_MCs ) )
        test_dims = np.vstack( ( test_dims, DSSi_all(esn,test_data) ) )
        print('res test done')

        train_MCs = np.vstack( ( train_MCs, np.array( list( MCi(esn,i,train_data[:2000] ) for i in range(esn.N) ) ) ) )
        train_dims = np.vstack( ( train_dims, DSSi_all(esn,train_data) ) )
        print('res train done')

        rhos = np.vstack( ( rhos, max(np.abs(np.linalg.eigvals(esn.ress[0].W))) ) )
        
        if new_MCs[i] > best_MCs[i]:
          best_Ws[i], best_W_ins[i] = np.copy(esn.ress[i].W), np.copy(esn.ress[i].W_in)

    clear_output()

  if not os.path.exists(f"results_deep/"):
    os.makedirs(f"results_deep/")
    
  if not os.path.exists(f"models_deep/"):
    os.makedirs(f"models_deep/")
    
  data = {
    'test_MCs': test_MCs,
    'train_MCs': train_MCs,
    'test_dims': test_dims,
    'train_dims': train_dims,
  }
  title=''
  now = datetime.now()
  if type(step) == list:
    title = f'deep_one_at_time_max_epochs_{max_epochs}_{step[0]}_{step[1]}_rdensity_{r_density}_idensity_{i_density}_Nr_{Nr}_Nu_{Nu}_mesInterval_{mesure_interval}_init_rho_{rho}_{now.strftime("%-d-%b-%H:%M:%S")}'
  else:
    string_train_fun = f'{train_fun}'.split(' ')[1]
    title = f'deep_one_at_time_max_epochs_{max_epochs}_{step}_{string_train_fun}_rdensity_{r_density}_idensity_{i_density}_Nr_{Nr}_Nu_{Nu}_mesInterval_{mesure_interval}_init_rho_{rho}_{now.strftime("%-d-%b-%H:%M:%S")}'
  with open(f'results_deep/{title}.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
  data_model = {
    'best_Ws': best_Ws,
    'best_W_ins': best_W_ins,
  }
  with open(f'models_deep/{title}.pickle', 'wb') as handle:
    pickle.dump(data_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

  fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,9))

  ax1.set_ylabel('MC') ; ax1.title.set_text('Memory capacity'); 
  ax2.set_ylabel('DSS') ; ax2.title.set_text('Dimensione spazio stati')
  ax3.set_ylabel('rho') ; ax3.title.set_text('raggio spettrale primo reservoir')

  ax1.plot(test_MCs[:,0],'-r') ;ax1.plot(test_MCs[:,1],'-c') ;ax1.plot(test_MCs[:,2],'-m') 
  ax1.plot(test_MCs[:,3],'-y') ;ax1.plot(test_MCs[:,4],'-b') 

  ax1.plot(train_MCs[:,0],'--r') ;ax1.plot(train_MCs[:,1],'--c') ;ax1.plot(train_MCs[:,2],'--m') 
  ax1.plot(train_MCs[:,3],'--y') ;ax1.plot(train_MCs[:,4],'--b') 

  ax1.legend(['reservoir 1 validation','reservoir 2 validation','reservoir 3 validation','reservoir 4 validation','reservoir 5 validation','reservoir 1 train','reservoir 2 train','reservoir 3 train','reservoir 4 train','reservoir 5 train',],loc='center left', bbox_to_anchor=(1, 0.5))

  ax2.plot(test_dims[:,0],'-r') ;ax2.plot(test_dims[:,1],'-c') ;ax2.plot(test_dims[:,2],'-m') ;
  ax2.plot(test_dims[:,3],'-y') ;ax2.plot(test_dims[:,4],'-b') 

  ax2.plot(train_dims[:,0],'--r') ;ax2.plot(train_dims[:,1],'--c') ;ax2.plot(train_dims[:,2],'--m') ;
  ax2.plot(train_dims[:,3],'--y') ;ax2.plot(train_dims[:,4],'--b') 

  ax2.legend(['reservoir 1 validation','reservoir 2 validation','reservoir 3 validation','reservoir 4 validation','reservoir 5 validation', 'reservoir 1 train','reservoir 2 train','reservoir 3 train','reservoir 4 train','reservoir 5 train',],loc='center left', bbox_to_anchor=(1, 0.5))

  ax3.plot( rhos[:,0],'-r')


  fig.tight_layout()
