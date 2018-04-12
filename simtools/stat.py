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