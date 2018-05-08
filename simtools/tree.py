#!/usr/bin/env python

import random
import numpy as np
import pandas as pd
import dendropy
import ete3

def get_random_topo_leaves(num_leaves, lbl_leaves):
  '''Generate random tree topoplogy for a given set of leaves.'''

  tree_ete = ete3.Tree()
  tree_ete.populate(num_leaves, names_library=lbl_leaves)
  tree_nwk = tree_ete.write()

  return tree_nwk

def get_random_topo_nodes(num_nodes, lbl_nodes):
  '''Generate random tree topology for a given set of nodes.'''

  tree_ete = ete3.Tree(name=lbl_nodes[0], dist=1.0)
  nodes = [tree_ete]
  for i in range(1, num_nodes):
    parent = random.choice(nodes)
    child = parent.add_child(name=lbl_nodes[i], dist=1.0)
    nodes.append(child)

  tree_nwk = tree_ete.write(format=1, format_root_node=True)
  return tree_nwk

def reformat_int_to_leaf(tree_nwk):
  '''Reformat tree: pull out internal nodes as leafs.'''

  tree_ete = ete3.Tree(tree_nwk, format=1)
  for node in tree_ete.traverse():
    if not node.is_leaf():
      # create new leaf node
      leaf = ete3.TreeNode(name=node.name, dist=0)
      # mark populated internal node as unpopulated
      node.name = ''
      # attach leaf to internal node
      node.add_child(leaf)

  return tree_ete.write()

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
