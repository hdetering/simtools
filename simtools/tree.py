#!/usr/bin/env python

import numpy as np
import pandas as pd
import dendropy
import ete3

def get_random_topo(num_leaves, lbl_leaves):
  '''Generate random tree topoplogy for a given set of leaves.'''

  tree_ete = ete3.Tree()
  tree_ete.populate(num_leaves, names_library=lbl_leaves)
  tree_nwk = tree_ete.write()

  return tree_nwk


def get_leaf_dist(tree_nwk):
  '''Get pairwise distances between leafs of a tree.'''
  
  # read tree from Newick
  try:
    tree_dp = dendropy.Tree.get(data=tree_nwk, schema='newick')
  except:
    print('[ERROR] Cannot load tree from Newick:\n{}'.format(tree_nwk))

  # get leaf labels
  lbl = [x.label for x in tree_dp.taxon_namespace]
  # prepare distance matrix
  df = pd.DataFrame(0.0, index=lbl, columns=lbl)
  # determine distances
  m = tree_dp.phylogenetic_distance_matrix()
  for i in range(len(lbl)):
    for j in range(i+1, len(lbl)):
      tax1 = tree_dp.taxon_namespace[i]
      tax2 = tree_dp.taxon_namespace[j]
      d = m.patristic_distance(tax1, tax2)
      df.loc[tax1.label, tax2.label] = d
      df.loc[tax2.label, tax1.label] = d
    
  return df

def pick_related_nodes(n, dist, e_init):
  '''Choose a set of nodes related to an initial node.'''
  
  # make sure input is of the expected type
  assert type(dist) is pd.DataFrame, 'Expected type of parameter "dist": pandas.DataFrame'
  
  # get distances for initial element
  d = dist.loc[e_init, dist.columns != e_init]
  # transform distances to similarities
  s = np.exp(-d)
  # transform similarities to probabilities (normalize)
  p = s / sum(s)
  # choose elements according to probabilities
  res = np.random.choice(p.index, n, replace=False, p=p)
  
  return res