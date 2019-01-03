#!/usr/bin/env python

import numpy as np
from scipy.stats import beta

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
