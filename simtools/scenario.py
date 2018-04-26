#!/usr/bin/env python
from __future__ import print_function

from . import stat
from . import tree

import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
import uuid

class TumorSetup:
  '''Encapsulates configuration parameters for a tumor simulation.'''
  def __init__(self, tree, prev, samp, p, ttype):
    self.id = str(uuid.uuid1())

    self.nclones = len(prev.columns)
    self.nsamples = len(prev.index)
    self.ttype = ttype

    self.tree_nwk = tree
    self.df_prev = prev
    self.df_sampling = samp
    self.p_nclones = p

def pick_prevalence_unbiased(n):
  prev = np.random.dirichlet([1]*n)
  return prev

def get_tumor_setup(num_clones, num_samples, tumor_type, seed=0, retries=100):
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
  tree_nwk = tree.get_random_topo(num_clones, lbl_clones)
  tree_dist = tree.get_leaf_dist(tree_nwk)

  # get prior probabilities for number of clones per sample
  assert tumor_type in ['hs', 'ms','us'], 'Tumor type must be "hs", "ms" or "us".'
  if tumor_type == 'hs':
    p_nclones = stat.get_beta_discrete(num_clones, a=1, b=5)
  elif tumor_type == 'ms':
    p_nclones = stat.get_beta_discrete(num_clones, a=1, b=1)
  elif tumor_type == 'us':
    p_nclones = stat.get_beta_discrete(num_clones, a=5, b=1)

  # choose number of clones for each sample:
  nclones = np.random.choice(
    range(1, num_clones+1), num_samples, p=p_nclones)
  df_sampling = pd.DataFrame(data=nclones,
                             index=lbl_regions,
                             columns=['nclones'])

  # choose clones to include in each sample
  lst_clones = []
  all_clones_sampled = False
  while not all_clones_sampled and retries>0:
    retries -= 1
    lst_clones = []
    set_clones = set()
    for idx, row in df_sampling.iterrows():
      # get number of clones selected before
      nclones = row['nclones']
      # choose random initial clone
      c_init = np.random.choice(lbl_clones)
      sel_clones = [c_init]
      # choose additional clones according to tree distance
      c_add = tree.pick_related_nodes(nclones-1, tree_dist, c_init)
      sel_clones += list(c_add)
      lst_clones.append(sel_clones)
      set_clones.update(sel_clones)
    all_clones_sampled = (len(set_clones) == num_clones)
  
  # make sure that valid result was found (tries not exceeded)
  if retries == 0:
    print('''[ERROR] No complete sampling could be generated.
[ERROR] (num_clones: {}, num_samples: {}, tissue_type: {})
[ERROR] Increase number of samples or reduce number of clones.'''.format(
  num_clones, num_samples, tumor_type), file=sys.stderr)
    sys.exit(1)

  df_sampling['sel_clones'] = pd.Series(lst_clones, index=df_sampling.index)

  # assign prevalence for clones in each sample
  df_prev = pd.DataFrame(0, index=lbl_regions,
                         columns=lbl_clones, dtype=float)
  for idx, row in df_sampling.iterrows():
    p = np.random.dirichlet([1]*row['nclones'])
    df_prev.loc[idx, row['sel_clones']] = p

  # return result
  return TumorSetup(tree_nwk, df_prev, df_sampling, p_nclones, tumor_type)


def write_tumor_scenario(tumor_setup, dir_out_root, args):
  '''Export simulation scenario to file system.

  A directory will be created containing:
  1. clone tree as Newick file
  2. config as YAML file
  3. meta info as YAML file
  '''

  # check if config template is available
  fn_config = os.path.join(os.path.dirname(__file__), 'config.yml')
  assert os.path.exists(fn_config), '[ERROR] Missing template: {}'.format(fn_config)
  print('[INFO] Using config template: {}'.format(fn_config))

  # create new directory for replicate
  dir_out = os.path.join(dir_out_root, tumor_setup.id)
  os.makedirs(dir_out)

  # write tree to file
  fn_tree = 'clone_tree.nwk'
  with open(os.path.join(dir_out, fn_tree), 'wt') as f:
    f.write('{}\n'.format(tumor_setup.tree_nwk))

  # write prevalence matrix to file
  fn_prev = 'clone_prev.csv'
  path_prev = os.path.join(dir_out, fn_prev)
  tumor_setup.df_prev.to_csv(path_prev, float_format='%.2f')

  # create config file from template
  conf = yaml.load(open(fn_config, 'rt'))
  conf['tree'] = fn_tree
  conf['sampling'] = fn_prev
  if args:
    if args.seq_art_path:
      conf['seq-art-path'] = args.seq_art_path
    if args.seq_coverage:
      conf['seq-coverage'] = args.seq_coverage

  # export config to file
  fn_conf_out = os.path.join(dir_out, 'config.yml')
  with open(fn_conf_out, 'wt') as f:
    yaml.dump(conf, f)
  
  # compile meta info
  meta = {}
  meta['nclones'] = tumor_setup.nclones
  meta['nsamples'] = tumor_setup.nsamples
  meta['ttype'] = tumor_setup.ttype

  # write meta info to file
  fn_meta_out = os.path.join(dir_out, 'meta.yml')
  with open(fn_meta_out, 'wt') as f:
    yaml.dump(meta, f, default_flow_style=False)
