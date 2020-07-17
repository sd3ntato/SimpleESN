import numpy as np
from scipy.stats import uniform
import scipy.sparse as s

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
  def __init__(self, rho =0.9, N=10, Nr=10, Nu=1,r_density=0.5, i_density=1):
    #iperparametri rete
    self.rho = rho
    self.N = N
    self.Nu = Nu
    self.Nr = Nr
    self.Ny = 1

    self.ress = [None]*N
    self.ress[0] = Reservoir(Nu=Nu, Nr=Nr, r_density=r_density, i_density=i_density)
    for i in range(1,N):
      self.ress[i] = Reservoir(Nu=Nr,Nr=Nr, r_density=r_density, i_density=i_density) 
    
  def compute_state(self,u):
    cu = self.ress[0].compute_state(u) 
    for i in range(1,self.N):
      cu = self.ress[i].compute_state(cu)
    return np.array( list( np.copy(res.x) for res in self.ress ) ) # shape (N,Nr,1)
  
  def reset_states(self):
    for res in self.ress:
      res.x = np.zeros((self.Nr,1))

def compute_dim_state_space(esn,data):
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

def compute_weights(w,x,y,step):
  m = np.diag(np.square(y.reshape(-1)))
  xt = x.reshape(1,-1)
  d = step * ( np.dot(y, xt ) - np.dot(m,w) )
  return d