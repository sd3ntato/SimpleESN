import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def plot_given_conditions(conditions, ax, filenames):
  for_the_mean = None
  for f in filenames:
    if( np.all( list( map( lambda c: c in f, conditions ) ) ) ):
      df = pd.read_csv(f)
      df[df>60] = 0
      ax.plot( df['test_MCs'], alpha=0.1 )
      try:
        for_the_mean =  np.vstack(( for_the_mean, df['test_MCs'].to_numpy() ))
      except:
        for_the_mean = df['test_MCs'].to_numpy()

  try:
    means = np.mean(for_the_mean,axis=0)
    ax.plot( means )
    ax.set_title( conditions[0] )
    print( for_the_mean.shape )
    return means, for_the_mean.shape[0]
  except:
    print(f'waring: this thing is like this: {for_the_mean}')
    return None

def get_filenames(directory):
  filenames = []
  for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f'{f}'!= f'{directory}.DS_Store' :
      filenames.append(f)
  
  return filenames

def visualize_MC( i_densities=[ '0.2', '0.5', '0.8', '1'],   train_fun='train_rec',   r_density = 0.1,   max_epochs = 2000,  lr = -1e-05 ):
  filenames = get_filenames(directory='./results/' )

  fig, axs = plt.subplots(2, 2, sharey=True)
  fig.tight_layout(h_pad=3.5)

  means = {}
  n_samples = {}

  for i, el in enumerate(i_densities):
    conditions = [ f'idensity_{el}', train_fun, f'rdensity_{r_density}', f'max_epochs_{max_epochs}', f'{lr}' ]
    if (res:= plot_given_conditions( conditions, axs[ i//2 , i%2 ], filenames )) is  not None: 
      means[el], n_samples[el] = res

  def compute_stats(means):
    return means[0], means[np.argmax(means)], (means[ np.argmax(means) ] - means[0] ) / means[0] * 100

  df = pd.DataFrame( np.vstack( [ np.hstack( ( compute_stats( means[el] ), n_samples[el] ) ) if el in means else [math.nan]*4  for el in i_densities ] ), index=i_densities, columns=['init_MC', 'final_MC', 'IPM', 'n_samples'] ) 
  
  return df