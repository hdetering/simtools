#!/usr/bin/env python
from __future__ import print_function, division

from glob import glob
import os
import uuid
import vcf

class VariantSet:
  '''Bundles multiregional tumor variants and meta data.'''

  def __init__(self, parent_dir, seed):
    self.id = str(uuid.uuid1())
    self.parent_dir = parent_dir
    self.seed = seed

    # identify parent files
    self.lst_vcf_gl = glob(os.path.join(parent_dir, 'germline.vcf*'))
    assert len(self.lst_vcf_gl) > 0
    self.fn_vcf_gl = self.lst_vcf_gl[0]
    self.lst_vcf_som = glob(os.path.join(parent_dir, 'somatic.vcf*'))
    assert len(self.lst_vcf_som) > 0
    self.lst_vcf_rg = glob(os.path.join(parent_dir, 'bam', '*.vcf*'))
    assert len(self.lst_vcf_rg) > 0

    # initialize properties used later
    self.vars_gl = []
    self.vars_som = []
    self.n_vars_gl = 0
    self.n_vars_gl_true = 0
    self.n_vars_som = 0
    self.n_vars_som_true = 0


def setup_dir(varset):
  '''Create subdir, add symlinks to necessary files.'''
  assert os.path.exists(varset.parent_dir)

  # create a directory to keep all varsets, if not present
  varset_dir = os.path.join(varset.parent_dir, 'varsets')
  if not os.path.exists(varset_dir):
    os.mkdir(varset_dir)
  
  # create directory for this replicate
  workdir = os.path.join(varset_dir, varset.id)
  assert not os.path.exists(workdir)
  os.mkdir(workdir)

  # identify necessary files
  lst_ref = glob(os.path.join(varset.parent_dir, 'ref.fa*'))

  # create symlinks
  for path in lst_ref: #+ lst_vcf:
    os.symlink(path, os.path.join(workdir, os.path.basename(path)))

def select_vars(varset, frac_vars_gl, frac_vars_som, rc_min):
  '''Sample subset of variants from a simulated dataset.

  1. reads VCF files with germline and somatic VCFs, select those having 
     minimum ALT allele read count.
  2. choose specified fraction of germline SNVs.
  3. choose specified fraction of somatic SNVs. 

  Parameters
  ---
  :frac_vars_gl:   Fraction of germline variants to output.
  :frac_vars_som:  Fraction of somatic variants to output.
  :rc_min:         Minimum ALT allele read count to accept a variant.

  Output
  ---
  1. Selected germline vars. (list of vcf.Record)
  2. Selected somatic vars. (list of vcf.Record)
  '''
  
  # somatic vars
  vcf_rdr = vcf.Reader(filename=varset.fn_vcf_gl)
  for rec in vcf_rdr:
    # check if read count is at least minimum for any ALT allele
    if any([x >= rc_min for x in rec.INFO['AC']]):
      pass