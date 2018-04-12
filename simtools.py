#!/usr/bin/env python

import os
import random
import argparse

def run_gen_scenario(args):
  pass

def main(args):
  random.seed(int(args.seed))
  args.func(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate and convert files for use with CloniPhy.')
  subparsers = parser.add_subparsers(title="mode")

  parser_indel = subparsers.add_parser('scenario')
  parser_indel.add_argument('--nclones', default=3,  help='Number of clones (default: 3).')
  parser_indel.add_argument('--nsamples', default=5, help='Number of sampled regions (default: 5).')
  parser_indel.add_argument('--ttype', default='us', help='Tissue structural type; one of "us", "ms", "hs" (default: "us").')
  parser_indel.add_argument('--nrep', default=1,  help='Number of replicates (default: 1).')
  parser_indel.add_argument('--seed', default=0, help='Random seed (default: 0).') 
  parser_indel.set_defaults(func=run_gen_scenario)

  args = parser.parse_args()
  main(args)
