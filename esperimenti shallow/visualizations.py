import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def plot_given_conditions(what, conditions, ax, filenames, lim=None):
  for_the_mean_MC = None
  for_the_mean_DSS = None
  for f in filenames:
    if( np.all( list( map( lambda c: c in f, conditions ) ) ) ):
      df = pd.read_csv(f)
      df[df>60] = 0
      ax.plot( df[f'test_{what}s'], alpha=0.1 )
      try:
        for_the_mean_MC =  np.vstack(( for_the_mean_MC, df[f'test_MCs'].to_numpy() ))
        for_the_mean_DSS =  np.vstack(( for_the_mean_DSS, df[f'test_dims'].to_numpy() ))
      except:
        for_the_mean_MC = df[f'test_MCs'].to_numpy()
        for_the_mean_DSS = df[f'test_dims'].to_numpy()

  try:
    means_MC = np.mean(for_the_mean_MC,axis=0)
    variances_MC = np.var(for_the_mean_MC, axis=0)
    
    means_DSS = np.mean(for_the_mean_DSS,axis=0)
    variances_DSS = np.var(for_the_mean_DSS, axis=0)
    
    ax.plot( means_MC if what=='MC' else means_DSS )
    ax.set_title( conditions[0] )
    ax.set_ylim(lim)
    print( for_the_mean_MC.shape )
    return means_MC, variances_MC, means_DSS, variances_DSS, for_the_mean_MC.shape[0]
  except:
    print(f'warning: this thing is like this: {for_the_mean_MC}')
    return None

def get_filenames(directory):
  filenames = []
  for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f'{f}'!= f'{directory}.DS_Store' :
      filenames.append(f)

  return filenames

def visualize( what='MC', i_densities=[ '0.2', '0.5', '0.8', '1'],   train_fun='train_rec',   r_density = 0.1,   max_epochs = 2000,  lr = -1e-05, lim=None ):
  filenames = get_filenames(directory='./results/' )

  fig, axs = plt.subplots(2, 2, sharey=True)
  fig.tight_layout(h_pad=3.5)

  means_MC = {}
  variances_MC = {}
  means_DSS = {}
  variances_DSS = {}
  n_samples = {}

  for i, el in enumerate(i_densities):
    conditions = [ f'idensity_{el}', train_fun, f'rdensity_{r_density}', f'max_epochs_{max_epochs}', f'{lr}' ]
    if (res:= plot_given_conditions(what, conditions, axs[ i//2 , i%2 ], filenames, lim=lim )) is  not None: 
      means_MC[el], variances_MC[el], means_DSS[el], variances_DSS[el], n_samples[el] = res

  def compute_stats(means_MC, variances_MC, means_DSS, variances_DSS):
    return f"{ np.round( means_MC[0], 2) } ± { np.round( np.sqrt( variances_MC[0]), 2) }", \
           f"{ np.round(means_MC[np.argmax(means_MC)], 2 ) } ± { np.round( np.sqrt( variances_MC[np.argmax(means_MC)]), 2) }  ", \
               np.argmax(means_MC), \
           f"{ np.round((means_MC[ np.argmax(means_MC) ] - means_MC[0] ) / means_MC[0] * 100 , 2)} ", \
           f"{ np.round( means_DSS[0] ,2) } ± {np.round( np.sqrt(variances_DSS[0]), 2)} ", \
           f"{ np.round( means_DSS[np.argmax(means_MC)] ,2) } ± {np.round( np.sqrt(variances_DSS[np.argmax(means_MC)]), 2)} ", \
           f"{ np.round((means_DSS[ np.argmax(means_MC) ] - means_DSS[0] ) / means_DSS[0] * 100 , 2)} ", \

  df = pd.DataFrame( np.vstack( [ np.hstack( ( compute_stats( means_MC[el], variances_MC[el], means_DSS[el], variances_DSS[el] ), n_samples[el] ) ) if el in means_MC else [math.nan]*5  for el in i_densities ] ), index=i_densities, columns=[f'init_MC', f'final_MC', 'idx_max', 'IPM MC', 'init_DSS', 'final_DSS', 'IPM DSS', 'n_samples'] ) 
  
  return df