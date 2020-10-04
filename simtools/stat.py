#!/usr/bin/env python
# vim: syntax=python tabstop=2 shiftwidth=2 expandtab
# coding: utf-8

import re
import sys
import numpy as np
from pydoc import locate  # dynamic type casts
from scipy.stats import beta, binom, gamma, nbinom, poisson, uniform

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
    rx_num = r'[0-9]*(\.[0-9]+)?(e[+\-]?[0-9]+)?' # scientific number format
    r = r'^(?P<d>[A-Z]+):(?P<p1>%s)(,(?P<p2>%s))?(,(?P<p3>%s))?$' % tuple([rx_num]*3)
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
    elif m.group('d').lower() == 'binom': # Binomial distribution
      if not m.group('p1') or not m.group('p2'):
        raise ValueError('Binomial distribution requires two params: Binom:n,p. (got "{}")'.format(str_dist))
      n = int(m.group('p1'))
      p = float(m.group('p2'))
      self._f_rand = lambda m: binom.rvs(n, p, size=m)
    elif m.group('d').lower() == 'nb': # Negative Binomial distribution
      # sample from Poisson with Gamma-distributed means
      if not m.group('p1') or not m.group('p2'):
        raise ValueError('Negative Binomial distribution requires two params: NB:mu,disp. (got "{}")'.format(str_dist))
      mu   = float(m.group('p1'))
      disp = float(m.group('p2'))
      gamma_shape = disp
      gamma_scale = (disp+mu)/disp-1
      f_pois = lambda l: poisson.rvs(l)
      self._f_rand = lambda n: map(f_pois, gamma.rvs(gamma_shape, scale=gamma_scale, size=n))
    elif m.group('d').lower() == 'bp': # Bounded Pareto distribution
      # formulation from: https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution
      if not m.group('p1') or not m.group('p2') or not m.group('p3'):
        raise ValueError('Bounded Pareto distribution requires three params: BP:L,H,a. (got "{}")'.format(str_dist))
      l = float(m.group('p1'))
      h = float(m.group('p2'))
      a = float(m.group('p3'))
      if not (l > 0.0 and l < h):
        raise ValueError('Limits for Bounded Pareto must satisfy 0<L<H.')
      f_bp = lambda u: (-1*(u*h**a - u*l**a - h**a)/(h**a * l**a))**(-1/a)
      self._f_rand = lambda n: map(f_bp, uniform.rvs(0, 1, size=n))  
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
  rx_num = r'[0-9]*(\.[0-9]+)?(e[+\-]?[0-9]+)?' # scientific number format
  rx_uni = r'^F:%s$' % rx_num
  rx_bin = r'^(B|U):%s,%s$' % (rx_num, rx_num)
  regex = '|'.join([rx_uni, rx_bin])
  if re.match(regex, str_dist, re.IGNORECASE):
    return True
  else:
    return False

def set_values_for_params(params, n):
  '''Sample n values from the distribution given for each parameter.

  Expected keys in 'params':
    'user' : Value as specified by the user (can be distribution or fixed value)
    'type' : Data type to return (specified as string)
  
  Output: Adds a key 'rep_val' which contains sampled values for n replicates
  '''

  for p in params:
    v = params[p]['user']
    t = params[p]['type']
    #print('[DEBUG] {} : {}'.format(p, v))
    # check if user param specifies a distribution
    if is_distribution(str(v)): # sample replicate values from distribution
      print('{}: using distribution "{}"'.format(p, v))
      d = Distribution(v)
      l = d.sample(n)
      params[p]['rep_val'] = [locate(t)(x) for x in l]
    else: # same value for all replicates
      nv = locate(t)(v) if not (v is None) else None
      params[p]['rep_val'] = [nv] * n

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
