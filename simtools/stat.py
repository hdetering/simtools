#!/usr/bin/env python

import re
import sys
import numpy as np
from scipy.stats import beta, uniform

class Distribution:
  '''Enables and sampling of values from a given statistical distribution.
  
  Nomenclature to define distributions:
  - F:x   := always return same fixed value x
  - U:l,h := uniform distribution with support [l,h)
  '''

  _f_rand = None

  def __init__(self, str_dist):
    '''Attempt to initialize the distribution from a give string.'''
    if not str_dist:
      raise ValueError('Invalid distribution: "{}"'.format(str_dist))
    
    # regex: distro spec (up to 3 params allowed)
    r = '^(?P<d>[A-Z]+):(?P<p1>[0-9.]+)(,(?P<p2>[0-9.]+))?(,(?P<p3>[0-9.]+))?$'
    m = re.match(r, str_dist, re.IGNORECASE)
    if not m or not m.group('d'):
      raise ValueError('Invalid distribution: "{}"'.format(str_dist))
    
    # check for known distro types
    if m.group('d') in 'Ff': # fixed ditribution
      if not m.group('p1'):
        raise ValueError('Fixed distribution requires one param. (got "{}")'.format(str_dist))
      v = float(m.group('p1'))
      self._f_rand = lambda n: np.full(n, v)
    elif m.group('d') in 'Bb': # Beta distribution
      if not m.group('p1') or not m.group('p2'):
        raise ValueError('Beta distribution requires two params. (got "{}")'.format(str_dist))
      a = float(m.group('p1'))
      b = float(m.group('p2'))
      self._f_rand = lambda n: beta.rvs(a, b, size=n)
    elif m.group('d') in 'Uu': # Uniform distribution
      if not m.group('p1') or not m.group('p2'):
        raise ValueError('Uniform distribution requires two params. (got "{}")'.format(str_dist))
      l = float(m.group('p1'))
      h = float(m.group('p2'))
      self._f_rand = lambda n: uniform.rvs(l, h-l, size=n)
    else:
      raise ValueError('Unknown distribution: "{}"'.format(str_dist))
  
  def sample(self, n):
    '''Sample random values from the underlying distribution.
  
    Returns: numpy array
    Parameters:
    ---
    :n:  Number of values to sample.
    '''
    return list(self._f_rand(n))

def is_distribution(str_dist):
  #regex = '^[FU]:[0-9.]+(,[0-9.]+)*$'
  rx_uni = '^F:[0-9.]+$'
  rx_bin = '^(B|U):[0-9.]+,[0-9.]+$'
  regex = '|'.join([rx_uni, rx_bin])
  if re.match(regex, str_dist, re.IGNORECASE):
    return True
  else:
    return False


def get_beta_discrete(num_bins, a, b):
  # boundaries of bins
  limits = np.cumsum([0.0]+[1/num_bins]*num_bins)
  # get cumulative density for bin limits
  cdf = beta.cdf(limits, a, b)
  # get bin-wise probabilities
  bin_probs = np.diff(cdf)

  return bin_probs

def get_random_number(distro, n=1):
  '''Returns a random number sampled from the given distribution.

  Supported distributions:
  distro      | class   | params
  ------------+---------+-------------------------------------------
  F:val       | Fixed   | val: value to return
  U:min,max   | Uniform | min: minimum (incl.), max: maximum (excl.)

  TODO: expand using NumPy distributions:
  https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.random.html
  '''

  # parse distribution specification
  d_p = distro.split(':')
  if len(d_p) != 2:
    raise ValueError('Malformed distribution specification: {}'.format(distro))
  d, p = d_p
  params = p.split(',')

  if d == 'F':
    if len(params) == 1:
      res = [float(params[0])] * n
    else:
      raise ValueError('Malformed distribution specification: {}'.format(distro))
  elif d == 'U':
    if len(params) == 2:
      return np.random.uniform(float(params[0]), float(params[1]), n)
    else:
      raise ValueError('Malformed distribution specification: {}'.format(distro))

  if n == 1:
    return res[0]
  else:
    return res
