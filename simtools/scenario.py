#!/usr/bin/env python

import simtools.stat
import simtools.tree

import random
import numpy as np
import pandas as pd

def pick_prevalence_unbiased(n):
  prev = np.random.dirichlet([1]*n)
  return prev

def get_tumor_setup(num_clones, num_samples, tumor_type, seed=0):
  '''Generate configuration for a tumor sequencing dataset.
  
  A setup comprises multiple clones and sampling regions and
  returns a clone tree and prevalence matrix for the specified
  input parameters.
  
  Parameters
  ---
  :num_clones:  Number of clones present in the tumor.
  :num_samples: Number of tumor regions sampled.
  :tumor_type:  One of "hs", "ms", "us" (highly, moderately or unstructured).
  :seed:        Random seed.
  
  Output
  ---
  1. Clone tree in Newick notation. (string)
  2. Prevalence matrix (sample X clone). (DataFrame)
  3. Sampling scheme. (DataFrame)
  4. Prior used for selection of shared clones per sample. (DataFrame)
  '''

  # make reproducible random choices
  random.seed(seed)
  np.random.seed(seed)

  # define labels for clones and samples
  lbl_clones = ['C{}'.format(i+1) for i in range(num_clones)]
  lbl_regions = ['R{}'.format(i+1) for i in range(num_samples)]

  # init random clone tree topology
  tree_nwk = simtools.tree.get_random_topo(num_clones, lbl_clones)
  tree_dist = simtools.tree.get_leaf_dist(tree_nwk)

  # get prior probabilities for number of clones per sample
  assert tumor_type in ['hs', 'ms','us'], 'Tumor type must be "hs", "ms" or "us".'
  if tumor_type == 'hs':
    p_nclones = simtools.stat.get_beta_discrete(num_clones, a=1, b=5)
  elif tumor_type == 'ms':
    p_nclones = simtools.stat.get_beta_discrete(num_clones, a=1, b=1)
  elif tumor_type == 'us':
    p_nclones = simtools.stat.get_beta_discrete(num_clones, a=5, b=1)

  # choose number of clones for each sample:
  nclones = np.random.choice(
    range(1, num_clones+1), num_samples, p=p_nclones)
  df_sampling = pd.DataFrame(data=nclones,
                             index=lbl_regions,
                             columns=['nclones'])

  # choose clones to include in each sample
  all_clones_sampled = False
  while not all_clones_sampled:
    lst_clones = []
    set_clones = set()
    for idx, row in df_sampling.iterrows():
      # get number of clones selected before
      nclones = row['nclones']
      # choose random initial clone
      c_init = np.random.choice(lbl_clones)
      sel_clones = [c_init]
      c_add = simtools.tree.pick_related_nodes(nclones-1, tree_dist, c_init)
      sel_clones += list(c_add)
      lst_clones.append(sel_clones)
      set_clones.update(sel_clones)
  all_clones_sampled = (len(set_clones) == num_clones)
  df_sampling['sel_clones'] = pd.Series(lst_clones, index=df_sampling.index)

  # assign prevalence for clones in each sample
  df_prev = pd.DataFrame(0, index=lbl_regions,
                         columns=lbl_clones, dtype=float)
  for idx, row in df_sampling.iterrows():
    p = np.random.dirichlet([1]*row['nclones'])
    df_prev.loc[idx, row['sel_clones']] = p

  # return result
  return (tree_nwk, df_prev, df_sampling, p_nclones)
