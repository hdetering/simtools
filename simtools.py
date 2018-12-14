#!/usr/bin/env python
from __future__ import print_function

from simtools import scenario

import os
import sys
import random
import argparse

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
    args.nclones, 
    args.nsamples, 
    args.ttype, 
    args.nrep, 
    os.path.abspath(args.out), 
    args.seed), 
  file=sys.stderr)

  for i in range(args.nrep):
    rep_seed = random.randrange(MAX_SEED)
    tumor_setup = scenario.get_tumor_setup(
      args.nclones, 
      args.nsamples, 
      args.ttype, 
      rep_seed,
      args.retries)
    scenario.write_tumor_scenario(tumor_setup, args.out, args)
    print('{}:\t{}'.format(i+1, os.path.join(os.path.abspath(args.out), tumor_setup.id)))

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
  parser_scenario.add_argument('--nclones', type=int, required=True,  help='Number of clones.')
  parser_scenario.add_argument('--nsamples', type=int, required=True, help='Number of sampled regions.')
  parser_scenario.add_argument('--ttype', required=True, help='Tissue structural type; one of "us", "ms", "hs".')
  parser_scenario.add_argument('--nrep', type=int, default=1,  help='Number of replicates [1].') 
  parser_scenario.add_argument('--out', default='sims',  help='Output directory ["sims"]')
  parser_scenario.add_argument('--seed', type=int, help='Master random seed.') 
  parser_scenario.add_argument('--retries', type=int, default=100, help='Number of retries in sampling phase [100].')

  group_mut = parser_scenario.add_argument_group('mutations')
  group_mut.add_argument('--mut-som-trunk', default=0, help='Number of trunk (i.e. clonal) mutations [0]')  

  group_seq = parser_scenario.add_argument_group('sequencing')
  group_seq.add_argument('--seq-read-gen', type=bool, default=False, help='Simulate reads? (false: simulate read counts)')
  group_seq.add_argument('--seq-art-path', help='Path to ART executable.')
  group_seq.add_argument('--seq-coverage', type=int, help='Sequencing depth.')
  parser_scenario.set_defaults(func=run_gen_scenario)  

  if len(sys.argv) == 1:
    parser.print_usage(sys.stderr)
    sys.exit(1)

  args = parser.parse_args()
  main(args)
