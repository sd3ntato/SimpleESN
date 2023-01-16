import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def plot_given_conditions(what, conditions, ax, filenames, lim=None):
  for_the_mean = None
  for f in filenames:
    if( np.all( list( map( lambda c: c in f, conditions ) ) ) ):
      df = pd.read_csv(f)
      df[df>60] = 0
      ax.plot( df[f'test_{what}s'], alpha=0.1 )
      try:
        for_the_mean =  np.vstack(( for_the_mean, df[f'test_{what}s'].to_numpy() ))
      except:
        for_the_mean = df[f'test_{what}s'].to_numpy()

  try:
    means = np.mean(for_the_mean,axis=0)
    variances = np.var(for_the_mean, axis=0)
    ax.plot( means )
    ax.set_title( conditions[0] )
    ax.set_ylim(lim)
    print( for_the_mean.shape )
    return means, variances, for_the_mean.shape[0]
  except:
    print(f'warning: this thing is like this: {for_the_mean}')
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

  means = {}
  variances = {}
  n_samples = {}

  for i, el in enumerate(i_densities):
    conditions = [ f'idensity_{el}', train_fun, f'rdensity_{r_density}', f'max_epochs_{max_epochs}', f'{lr}' ]
    if (res:= plot_given_conditions(what, conditions, axs[ i//2 , i%2 ], filenames, lim=lim )) is  not None: 
      means[el], variances[el], n_samples[el] = res

  def compute_stats(means, variances):
    return f"{ np.round( means[0], 2) }±{ np.round( np.sqrt( variances[0]), 2) }", \
           f"{ np.round(means[np.argmax(means)], 2 ) }±{ np.round( np.sqrt( variances[np.argmax(means)]), 2) }  ", \
               np.argmax(means), \
           f"{ np.round((means[ np.argmax(means) ] - means[0] ) / means[0] * 100 , 2)} "

  df = pd.DataFrame( np.vstack( [ np.hstack( ( compute_stats( means[el], variances[el] ), n_samples[el] ) ) if el in means else [math.nan]*5  for el in i_densities ] ), index=i_densities, columns=[f'init_{what}', f'final_{what}', 'idx_max', 'IPM', 'n_samples'] ) 
  
  return df