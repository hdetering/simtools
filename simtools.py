#!/usr/bin/env python
from __future__ import print_function

from simtools import scenario, spikein, stat, subsample

import os
import re
import sys
import random
import argparse
import numpy as np
import pandas as pd

# globals
MAX_SEED = 2**31-1

def run_gen_scenario(args):
  print('''
Generating tumor configurations (input for CloniPhy simulator).
-------------------------------------------------------------------------------
Parameters:
  number of clones:   {}
  number of samples:  {}
  tumor type:         {}
  replicates:         {}
  output dir:         {}
  seed:               {}
-------------------------------------------------------------------------------
'''.format(
    args.tree_nclones,
    args.smp_nsamples,
    args.smp_ttype,
    args.nrep,
    os.path.abspath(args.out),
    args.seed),
  file=sys.stderr)

  # sanity checks
  if not bool(args.mut_gl_rate) ^ bool(args.mut_gl_num):
    print('[ERROR] Specify either --mut-gl-rate or --mut-gl-num')
  if not bool(args.mut_som_rate) ^ bool(args.mut_som_num):
    print('[ERROR] Specify either --mut-som-rate or --mut-som-num')

  # store simulation params for each replicate
  params = {
    'tree_nclones'      : {'user': args.tree_nclones,      'type': 'int'},
    'smp_nsamples'      : {'user': args.smp_nsamples,      'type': 'int'},
    'smp_cont'          : {'user': args.smp_cont,          'type': 'float'},
    'smp_dec_disp'      : {'user': args.smp_dec_disp,      'type': 'float'},
    'smp_dec_ext'       : {'user': args.smp_dec_ext,       'type': 'float'},
    'mut_gl_rate'       : {'user': args.mut_gl_rate,       'type': 'float'},
    'mut_gl_num'        : {'user': args.mut_gl_num,        'type': 'int'},
    'mut_som_rate'      : {'user': args.mut_som_rate,      'type': 'float'},
    'mut_som_num'       : {'user': args.mut_som_num,       'type': 'int'},
    'mut_som_trunk'     : {'user': args.mut_som_trunk,     'type': 'float'},
    'mut_som_cnv_ratio' : {'user': args.mut_som_cnv_ratio, 'type': 'float'},
    'seq_rc_error'      : {'user': args.seq_rc_error,      'type': 'float'}
  }

  # check input params that may be spedified as distributions
  stat.set_values_for_params(params, args.nrep)
  #-----------------------------------------------------------------------------
  # for p in params:
  #   v = params[p]['user']
  #   t = params[p]['type']
  #   #print('[DEBUG] {} : {}'.format(p, v))
  #   # check if user param specifies a distribution
  #   if stat.is_distribution(v): # sample replicate values from distribution
  #     print('{}: using distribution "{}"'.format(p, v))
  #     d = stat.Distribution(v)
  #     l = d.sample(args.nrep)
  #     params[p]['rep_val'] = [locate(t)(x) for x in l]
  #   else: # same value for all replicates
  #     nv = locate(t)(v)
  #     params[p]['rep_val'] = [nv] * args.nrep
  #-----------------------------------------------------------------------------
  
  # generate replicates
  for i in range(args.nrep):
    rep_seed = random.randrange(MAX_SEED)
    np.random.seed(seed=rep_seed)
    #import pdb; pdb.set_trace()
    if args.smp_ttype == 'DEC':
      # distribution to sample dispersal rates
      dist_disp = stat.Distribution(params['smp_dec_disp']['user'])

      ts = None
      # scenario generation can fail if dispersal rate is too low.
      # repeat on failure with resampled dispersal rate.
      while ts is None:
        disp = dist_disp.sample(1)[0]
        ts = scenario.get_tumor_setup_DEC(
          params['tree_nclones']['rep_val'][i],
          args.tree_lbl_pfx,
          args.tree_lbl_og,
          args.smp_lbl_pfx,
          args.smp_lbl_nrm,
          params['smp_nsamples']['rep_val'][i],
          disp,
          params['smp_dec_ext']['rep_val'][i],
          args.smp_retries,
          params['smp_cont']['rep_val'][i],
          rep_seed
        )
    else:
      ts = scenario.get_tumor_setup(
        params['tree_nclones']['rep_val'][i],
        params['smp_nsamples']['rep_val'][i],
        args.smp_ttype,
        rep_seed,
        args.smp_retries)
    
    # set non-phylogeny attributes
    ts.mut_gl_rate       = params['mut_gl_rate']['rep_val'][i]
    ts.mut_gl_num        = params['mut_gl_num']['rep_val'][i]
    ts.mut_som_rate      = params['mut_som_rate']['rep_val'][i]
    ts.mut_som_num       = params['mut_som_num']['rep_val'][i]
    ts.mut_som_trunk     = params['mut_som_trunk']['rep_val'][i]
    ts.mut_som_cnv_ratio = params['mut_som_cnv_ratio']['rep_val'][i]
    ts.seq_rc_error      = params['seq_rc_error']['rep_val'][i]

    scenario.write_tumor_scenario(ts, args.out, args)
    print('{}:\t{}'.format(i+1, os.path.join(os.path.abspath(args.out), ts.id)))

def run_spikein(args):
  np.random.seed(seed=args.seed)
  workdir = os.path.abspath(args.dir)
  print('''
Adding variants to existing VCF.
-------------------------------------------------------------------------------
Parameters:
  workdir:          {}
  prevalences:      {}
  reference:        {}
  mutation load:    {}
  VAF distribution: {}
  target depth:     {}
  infinite sites:   {}
  seed:             {}
-------------------------------------------------------------------------------
'''.format(
    workdir,
    args.prev.name,
    args.ref.name,
    args.mut_load,
    args.vaf_dist,
    args.depth,
    'no' if args.no_ism else 'yes',
    args.seed),
  file=sys.stderr)

  # parse prevalence matrix
  df_prev = pd.read_csv(args.prev, index_col=0)
  # locate input VCF files
  fns = [os.path.join(workdir, fn) for fn in sorted(os.listdir(workdir)) if re.match('^R\w+.rc.vcf.gz$', fn)]

  # sanity checks
  assert len(fns) == len(df_prev.index) # all regions have a VCF file

  dist_load = stat.Distribution(args.mut_load)
  mut_load = dist_load.sample(1)[0]
  dist_vaf = stat.Distribution(args.vaf_dist)
  # sample depth from negative binomial distro
  mu = args.depth
  dispersion = 30
  dist_dp = stat.Distribution('NB:{},{}'.format(mu, dispersion))

  spikein.main(
    fns,
    df_prev,
    args.ref,
    mut_load,
    dist_dp,
    dist_vaf,
    not args.no_ism
  )

def run_subsample(args):
  print('''
Subsampling variants from tumor dataset (output of CloniPhy simulator).
-------------------------------------------------------------------------------
Parameters:
  parent directory:   {}
  no. germline vars:  {}
  no. somatic vars:   {}
  replicates:         {}
  seed:               {}
-------------------------------------------------------------------------------
'''.format(
    args.dir_parent,
    args.num_gl_mut,
    args.num_som_mut,
    args.nrep,
    #os.path.abspath(args.out),
    args.seed),
  file=sys.stderr)

  # sanity checks
  assert os.path.exists(args.dir_parent)

  # store simulation params for each replicate
  params = {
    'snv_gl_frac'  : {'user': args.snv_gl_frac,  'type': 'float'},
    'snv_som_frac' : {'user': args.snv_som_frac, 'type': 'float'},
    'snv_som_fpr'  : {'user': args.snv_som_frac, 'type': 'float'},
    'snv_rc_min'   : {'user': args.snv_rc_min,   'type': 'int'},
  }

  # check input params that may be spedified as distributions
  stat.set_values_for_params(params, args.nrep)

  for i in range(args.nrep):
    rep_seed = random.randrange(MAX_SEED)
    np.random.seed(seed=rep_seed)

    varset = subsample.VariantSet(args.dir_parent, rep_seed)

    # set up directory structure for replicate
    subsample.setup_dir(varset)
    
    # select vars
    subsample.select_vars(
      varset,
      params['snv_gl_frac']['rep_val'][i],
      params['snv_som_frac']['rep_val'][i],
      params['snv_rc_min']['rep_val'][i]
    )
    # write selected variants to files


def main(args):
  # generate random random seed if none was provided
  if args.seed is None:
    args.seed = random.randrange(MAX_SEED)
  random.seed(int(args.seed))

  args.func(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate and convert files for use with CloniPhy.')
  subparsers = parser.add_subparsers(title="mode")

  #-----------------------------------------------------------------------------
  parser_scenario = subparsers.add_parser('scenario')
  parser_scenario.set_defaults(func=run_gen_scenario)
  parser_scenario.add_argument('--nrep', type=int, default=1, help='Number of replicates [1].')
  parser_scenario.add_argument('--out', default='sims', help='Output directory ["sims"]')
  parser_scenario.add_argument('--seed', type=int, help='Master random seed.')

  group_tree = parser_scenario.add_argument_group('tree')
  group_tree.add_argument('--tree-nclones', default='U:2,10', help='Number of clones.')
  group_tree.add_argument('--tree-lbl-pfx', default='clone_', help='Prefix for labelling tree nodes.')
  group_tree.add_argument('--tree-lbl-og', default='healthy', help='Label of (healthy) outgroup.')
  group_tree.add_argument('--tree-anc-rate', type=float, default=1.0,  help='Probability of internal nodes to be populated.')

  group_sampling = parser_scenario.add_argument_group('sampling')
  group_sampling.add_argument('--smp-nsamples', default='U:3,8', help='Number of sampled regions.')
  group_sampling.add_argument('--smp-lbl-pfx', default='R', help='Prefix for labelling regions.')
  group_sampling.add_argument('--smp-lbl-nrm', default='RN', help='Label for normal region.')
  group_sampling.add_argument('--smp-cont', default='F:0', help='Contamination with normal cells (distribution) [F:0]')
  group_sampling.add_argument('--smp-ttype', required=True, help='Tissue structural type; one of "DEC", "us", "ms", "hs".')
  group_sampling.add_argument('--smp-dec-disp', help='DEC only: dispersal success rate matrix; distribution or CSV file (nsamples X nsamples), no header.')
  group_sampling.add_argument('--smp-dec-ext', default='F:0.0', help='DEC only: extinction rates (distribution, single value or nsamples); single line CSV file.')
  group_sampling.add_argument('--smp-retries', type=int, default=100, help='Number of retries in sampling phase [100].')

  group_ref = parser_scenario.add_argument_group('reference')
  group_ref.add_argument('--ref-seq-num', type=int, default=10, help='Number of reference sequences (e.g. chromosomes) to generate [1].')
  group_ref.add_argument('--ref-seq-len-mean', type=int, default=1000000, help='Mean sequence bp length [10,000,000].')
  group_ref.add_argument('--ref-seq-len-sd', type=int, default=0, help='Standard deviation of sequence length [0].')
  
  group_mut = parser_scenario.add_argument_group('mutations')
  group_mut.add_argument('--mut-gl-rate', help='Germline mutation rate.')
  group_mut.add_argument('--mut-gl-num', help='Number of germline mutations.')
  group_mut.add_argument('--mut-som-rate', help='Somatic mutation rate.')
  group_mut.add_argument('--mut-som-num', help='Number of somatic mutations.')
  group_mut.add_argument('--mut-som-trunk', default='F:0.1', help='Fraction of trunk (i.e. clonal) mutations.')
  group_mut.add_argument('--mut-som-cnv-ratio', default='F:0.0', help='Expected fraction of somatic CNVs.')

  group_seq = parser_scenario.add_argument_group('sequencing')
  group_seq.add_argument('--seq-read-gen', type=bool, default=False, help='Simulate reads? (false: simulate read counts)')
  group_seq.add_argument('--seq-art-path', help='Path to ART executable.')
  group_seq.add_argument('--seq-coverage', type=int, help='Sequencing depth.')
  group_seq.add_argument('--seq-rc-error', default='F:0.0', help='Sequencing error rate (per bp).')
  #-----------------------------------------------------------------------------
  
  # spike in mutations following a VAF distribution into existing VCF
  #-----------------------------------------------------------------------------
  parser_spikein = subparsers.add_parser('spikein')
  parser_spikein.set_defaults(func=run_spikein)
  parser_spikein.add_argument('--dir', required=True, help='Working directory (for input/output files).')
  parser_spikein.add_argument('--prev', type = argparse.FileType('rt'), required=True, help='Input clone prevalence file.')
  parser_spikein.add_argument('--ref', type = argparse.FileType('rt'), required=True, help='Reference genome.')
  parser_spikein.add_argument('--mut-load', default='F:0.5', help='Mutational load of neutral muts (fraction of total vars).')
  parser_spikein.add_argument('--vaf-dist', default='U:0.0,0.5', help='Variant allele frequency distribution.')
  parser_spikein.add_argument('--depth', required=True, help='Mean sequencing depth.')
  parser_spikein.add_argument('--no-ism', action='store_true', help='Deactivates infinite sites model.')
  parser_spikein.add_argument('--seed', type=int, help='Random seed.')
  #-----------------------------------------------------------------------------
  
  # select a subset of variants from a tumor dataset
  #-----------------------------------------------------------------------------
  parser_subsample = subparsers.add_parser('subsample')
  parser_subsample.set_defaults(func=run_subsample)
  parser_subsample.add_argument('--dir-parent', required=True, help='Directory with simulated tumor data.')
  parser_subsample.add_argument('--nrep', type=int, default=1, help='Number of replicates [1].')
  parser_subsample.add_argument('--seed', type=int, help='Master random seed.')
  parser_subsample.add_argument('--snv-gl-frac', default='F:1.0', help='Fraction of true germline SNVs.')
  parser_subsample.add_argument('--snv-som-frac', default='U:0.1,1.0', help='Fraction of true somatic SNVs.')
  parser_subsample.add_argument('--snv-som-fpr', default='U:0.0,0.5', help='False positive rate among somatic SNVs.')
  #parser_subsample.add_argument('--snv-rc-min', default='U:1,5', help='Minimum number of reads to detect variant.')
  #-----------------------------------------------------------------------------
  
  if len(sys.argv) == 1:
    parser.print_usage(sys.stderr)
    sys.exit(1)

  args = parser.parse_args()
  main(args)
