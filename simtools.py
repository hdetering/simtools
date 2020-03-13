#!/usr/bin/env python
from __future__ import print_function

from simtools import scenario, stat

import os
import sys
import random
import argparse
import numpy as np
from pydoc import locate  # dynamic type casts

# globals
MAX_SEED = 2**32

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

  # store simulation params for each replicate
  params = {
    'tree_nclones'      : {'user': args.tree_nclones,      'type': 'int'},
    'smp_nsamples'      : {'user': args.smp_nsamples,      'type': 'int'},
    'smp_cont'          : {'user': args.smp_cont,          'type': 'float'},
    'smp_dec_disp_rg'   : {'user': args.dec_disp_rg,       'type': 'float'},
    'smp_dec_ext'       : {'user': args.dec_ext,           'type': 'float'},
    'mut_gl_rate'       : {'user': args.mut_gl_rate,       'type': 'float'},
    'mut_som_rate'      : {'user': args.mut_som_rate,      'type': 'float'},
    'mut_som_trunk'     : {'user': args.mut_som_trunk,     'type': 'float'},
    'mut_som_cnv_ratio' : {'user': args.mut_som_cnv_ratio, 'type': 'float'},
    'seq_rc_error'      : {'user': args.seq_rc_error,      'type': 'float'}
  }

  # check input params that may be spedified as distributions
  #-----------------------------------------------------------------------------
  for p in params:
    v = params[p]['user']
    t = params[p]['type']
    #print('[DEBUG] {} : {}'.format(p, v))
    # check if user param specifies a distribution
    if stat.is_distribution(v): # sample replicate values from distribution
      print('{}: using distribution "{}"'.format(p, v))
      d = stat.Distribution(v)
      l = d.sample(args.nrep)
      params[p]['rep_val'] = [locate(t)(x) for x in l]
    else: # same value for all replicates
      nv = locate(t)(v)
      params[p]['rep_val'] = [nv] * args.nrep
  #-----------------------------------------------------------------------------
  
  # generate replicates
  for i in range(args.nrep):
    rep_seed = random.randrange(MAX_SEED)
    np.random.seed(seed=rep_seed)

    if args.smp_ttype == 'DEC':
      ts = scenario.get_tumor_setup_DEC(
        params['tree_nclones']['rep_val'][i],
        args.tree_lbl_pfx,
        args.tree_lbl_og,
        args.smp_lbl_pfx,
        args.smp_lbl_nrm,
        params['smp_nsamples']['rep_val'][i],
        params['smp_dec_disp_rg']['rep_val'][i],
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
    ts.mut_gl_num        = round(params['mut_gl_raet']['rep_val'][i])
    ts.mut_som_num       = round(params['mut_som_rate']['rep_val'][i])
    ts.mut_som_trunk     = params['mut_som_trunk']['rep_val'][i]
    ts.mut_som_cnv_ratio = params['mut_som_cnv_ratio']['rep_val'][i]
    ts.seq_rc_error      = params['seq_rc_error']['rep_val'][i]

    scenario.write_tumor_scenario(ts, args.out, args)
    print('{}:\t{}'.format(i+1, os.path.join(os.path.abspath(args.out), ts.id)))

def main(args):
  # generate random random seed if none was provided
  if args.seed is None:
    args.seed = random.randrange(MAX_SEED)
  random.seed(int(args.seed))

  args.func(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate and convert files for use with CloniPhy.')
  subparsers = parser.add_subparsers(title="mode")

  parser_scenario = subparsers.add_parser('scenario')
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
  group_sampling.add_argument('--dec-disp-rg', help='DEC only: dispersal rate matrix; distribution or CSV file (nsamples X nsamples), no header.')
  group_sampling.add_argument('--dec-ext', default='F:0.1', help='DEC only: extinction rates (distribution, single value or nsamples); single line CSV file.')
  group_sampling.add_argument('--smp-retries', type=int, default=100, help='Number of retries in sampling phase [100].')

  group_ref = parser_scenario.add_argument_group('reference')
  group_ref.add_argument('--ref-seq-num', type=int, default=10, help='Number of reference sequences (e.g. chromosomes) to generate [1].')
  group_ref.add_argument('--ref-seq-len-mean', type=int, default=1000000, help='Mean sequence bp length [10,000,000].')
  group_ref.add_argument('--ref-seq-len-sd', type=int, default=0, help='Standard deviation of sequence length [0].')
  
  group_mut = parser_scenario.add_argument_group('mutations')
  group_mut.add_argument('--mut-gl-rate', default='F:1e-3', help='Number of germline mutations.')
  group_mut.add_argument('--mut-som-rate', default='F:1e-4', help='Number of somatic mutations.')
  group_mut.add_argument('--mut-som-trunk', default='F:0.1', help='Fraction of trunk (i.e. clonal) mutations.')
  group_mut.add_argument('--mut-som-cnv-ratio', default='F:0.0', help='Expected fraction of somatic CNVs.')

  group_seq = parser_scenario.add_argument_group('sequencing')
  group_seq.add_argument('--seq-read-gen', type=bool, default=False, help='Simulate reads? (false: simulate read counts)')
  group_seq.add_argument('--seq-art-path', help='Path to ART executable.')
  group_seq.add_argument('--seq-coverage', type=int, help='Sequencing depth.')
  group_seq.add_argument('--seq-rc-error', default='F:0.0', help='Sequencing error rate (per bp).')
  parser_scenario.set_defaults(func=run_gen_scenario)

  if len(sys.argv) == 1:
    parser.print_usage(sys.stderr)
    sys.exit(1)

  args = parser.parse_args()
  main(args)
