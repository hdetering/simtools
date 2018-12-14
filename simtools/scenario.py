#!/usr/bin/env python
from __future__ import print_function, division

from . import stat
from . import tree

import os
import sys
import random
import ete3
import itertools as it
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import expm
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

def powerset(iterable):
    'powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)'
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

def get_rate_matrix(disp, ext):
    '''Expand a list of dispersal and extinction rates to a evolutionary rate matrix.

    Row and column indices encode range compositions, e.g.:
      0 = 00 = ()
      1 = 01 = (R1)
      2 = 10 = (R2)
      3 = 11 = (R1,R2)
    '''

    n = len(ext) # number of areas
    Q = np.zeros((2**n, 2**n))
    for i in range(2**n):
      for j in range(2**n):
        if i==j: # diagonal will be filled later
          continue
        d = i ^ j # edit distance between ranges (XOR in binary notation)
        # if distance is a power of two, it is an instantaneous change
        if (d & (d-1)) == 0 and d != 0:
          if j-i > 0: # dispersal event
            # calculate index of target region
            t = int(np.log2(d))
            # add dispersal rate for each source region to target
            rate = sum([disp[s,t] for s in range(n) if (i & 2**s)])
            Q[i, j] = rate
          else: # extinction event (d<0)
            # calculate index of deleted region
            idx = int(np.log2(d))
            Q[i, j] = ext[idx]

    # set diagonal such that rows sum to zero
    Q[np.diag_indices_from(Q)] = -Q.sum(axis=1)

    return Q

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
  lbl_healthy = 'H' # healthy cells clone
  lbl_normal = 'RN' # normal sample
  lbl_clones = ['C{}'.format(i+1) for i in range(num_clones)]
  lbl_regions = ['R{}'.format(i+1) for i in range(num_samples)]

  # init random clone tree topology
  tree_nwk = tree.get_random_topo_nodes(num_clones, lbl_clones, lbl_healthy)
  tree_nwk_leaf = tree.reformat_int_to_leaf(tree_nwk, ignore_root=True)
  tree_dist = tree.get_leaf_dist(tree_nwk_leaf)

  # get prior probabilities for number of clones per sample
  assert tumor_type in ['hs', 'ms','us'], 'Tumor type must be "hs", "ms" or "us".'
  if tumor_type == 'hs':
    p_nclones = stat.get_beta_discrete(num_clones, a=1, b=3)
  elif tumor_type == 'ms':
    p_nclones = stat.get_beta_discrete(num_clones, a=1, b=1)
  elif tumor_type == 'us':
    p_nclones = stat.get_beta_discrete(num_clones, a=3, b=1)

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
    #p = np.random.dirichlet([1]*row['nclones'])
    p = pick_prevalence_unbiased(row['nclones'])
    df_prev.loc[idx, row['sel_clones']] = p

  # add additional row for "normal" (healthy) sample
  df_prev.loc[lbl_normal] = [0.0]*num_clones

  # return result
  return TumorSetup(tree_nwk, df_prev, df_sampling, p_nclones, tumor_type)

#-------------------------------------------------------------------------------
# DEC biogeographical model
#-------------------------------------------------------------------------------

def idx2range(idx):
  '''Returns the geographical range (a list of region indices) corresponding to
     a bitmask representing presence in the regions.
  '''
  lst_regions = []

  i = 0
  while idx >= 2**i: # stop when no further indices can be present
    if (idx & 2**i) == 2**i: # i-th bit is set
      lst_regions.append(i)
    i += 1

  return lst_regions

def range2idx(lst_regions):
  '''Returns the numeric representation (bitmask) corresponding to a
     geographical range (a list of region indices).
  '''
  idx = 0
  for i in sorted(lst_regions):
    idx += 2**i

  return idx

def get_prev_random(df_pres):
  '''Select prevalences for a presence matrix.'''
  df_prev = pd.DataFrame().reindex_like(df_pres).fillna(0)
  for idx, row in df_pres.iterrows():
    p = pick_prevalence_unbiased(len(row[row>0]))
    df_prev.loc[idx, row>0] = p

  print(df_prev)
  return df_prev

def get_tumor_setup_DEC(num_clones, num_samples, disp, ext, seed=0):
  '''Generate configuration for a tumor sequencing dataset.

  A setup comprises multiple clones and sampling regions and
  returns a clone tree and prevalence matrix for the specified
  input parameters.

  Parameters
  ---
  :num_clones:  Number of clones present in the tumor.
  :num_samples: Number of tumor regions sampled.
  :disp:        Dispersal rates between regions. (numpy.array)
  :ext:         Extinction rates within each region.
  :seed:        Random seed.

  Output
  ---
  1. Clone tree in Newick notation. (string)
  2. Prevalence matrix (sample X clone). (DataFrame)
  3. Sampling scheme. (DataFrame)
  4. Prior used for selection of shared clones per sample. (DataFrame)
  '''

  # read/generate dispersal rates
  if not disp: # assume symmetrically uniform dispersal rates between regions
    disp = np.ones((num_samples, num_samples))
    disp[np.diag_indices_from(disp)] = 0.0
    disp /= disp.sum(axis=1)
  elif os.path.exists(disp):
    disp = np.genfromtxt(disp, delimiter=',')
  else:
    print('''[ERROR] file does not exist: {}.'''.format(disp), file=sys.stderr)
    sys.exit(1)
  # read/generate extinction rates
  if not ext: # default: choose 0.1 for all regions
    print('''[INFO] no extinction rates supplied for DEC model (default: 0.1)''')
    ext = np.array([0.1]*num_samples)
  elif os.path.exists(ext):
    ext = np.genfromtxt(ext, delimiter=',')
  else:
    print('''[ERROR] file does not exist: {}.'''.format(disp), file=sys.stderr)
    sys.exit(1)

  # perform sanity checks on dispersal / extinction rates
  if not (type(disp) == np.ndarray and disp.ndim == 2):
    print('''[ERROR] get_tumor_setup_DEC() expects parameter 'disp' to be 2D numpy.ndarray.
[ERROR]   got: {}, dims: {}'''.format(type(disp), disp.ndim), file=sys.stderr)
    sys.exit(1)
  if not (type(ext) == np.ndarray and ext.ndim == 1):
    print('''[ERROR] get_tumor_setup_DEC() expects parameter 'ext' to be 1D numpy.ndarray.
[ERROR]   got: {}, dims: {}'''.format(type(ext), ext.ndim), file=sys.stderr)
    sys.exit(1)
  #if ( not np.issubclass_(disp.dtype.type, np.float) ):
  #  print('''[ERROR] get_tumor_setup_DEC() expects parameter 'dist' to be of float type.
  # [ERROR]  got: {}'''.format(disp.dtype.type)), file=sys.stderr
  #  sys.exit(1)
  num_regions = ext.shape[0]
  if not ( disp.shape == (num_regions, num_regions) ):
    print('''[ERROR] get_tumor_setup_DEC() dimensions of 'disp' and 'ext' do not match
[ERROR]   {} vs. {}'''.format(disp.shape, ext.shape), file=sys.stderr)
    sys.exit(1)

  # make reproducible random choices
  random.seed(seed)
  np.random.seed(seed)

  # define labels for clones and samples
  lbl_healthy = 'H' # healthy cells clone
  lbl_normal = 'RN' # normal sample
  lbl_clones = ['C{}'.format(i+1) for i in range(num_clones)]
  lbl_regions = ['R{}'.format(i+1) for i in range(num_regions)]

  # init random clone tree topology
  tree_nwk = tree.get_random_topo_nodes(num_clones, lbl_clones, lbl_healthy)

  # construct instantaneous rate matrix
  Q = get_rate_matrix(disp, ext)

  # simulate range evolution
  clone_range = {}
  tree_ete = ete3.Tree(tree_nwk, format=1)
  # pick random region for root clone
  clone_range[tree_ete.name] = [random.randrange(num_regions)]
  # pick ranges for remaining tree nodes, dependent on parent range
  for node in tree_ete.traverse('preorder'): # preorder: root, left, right
    for child in node.children:
      # calculate transition probabilities
      t = child.dist
      P = expm(Q*t)
      # select child's range index from probabilities in P
      probs = P[range2idx(clone_range[node.name])]
      child_range_idx = np.random.choice(len(probs), p=probs)
      clone_range[child.name] = idx2range(child_range_idx)

  # construct presence matrix from ranges on tree nodes
  df_pres = pd.DataFrame(0, index=lbl_regions, columns=lbl_clones, dtype=int)
  for idx_clone in range(num_clones):
    clone = lbl_clones[idx_clone]
    r = clone_range[clone]
    df_pres.iloc[r, idx_clone] = 1

  # assign prevalence values for each clone and region
  df_prev = get_prev_random(df_pres)
  # add additional row for "normal" (healthy) sample
  df_prev.loc[lbl_normal] = [0.0]*num_clones

  # return result
  return TumorSetup(tree_nwk, df_prev, None, None, "DEC")

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
    if args.seed is not None:
      conf['seed'] = args.seed
    if args.seq_read_gen:
      conf['seq-read-gen'] = args.seq_read_gen
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
