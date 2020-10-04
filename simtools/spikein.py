#!/usr/bin/env python

from simtools import stat

import os
import vcf
import numpy as np
import pandas as pd
from Bio import SeqIO
from typing import List
from collections import OrderedDict

class Variant:
  '''Stores information about mutated loci.'''
  def __init__(self, chrom, pos, id, ref, alt):
    self.chrom = chrom
    self.pos = pos
    self.id = id
    self.ref = ref
    self.alt = alt # list of ALT alleles
    self.qual = '60'
    self.formats = [] 
    self.smp_data = OrderedDict()
  
  def add_format(self, id_sample, tag, value):
    if tag not in self.formats:
      self.formats.append(tag)
    if id_sample not in self.smp_data:
      self.smp_data[id_sample] = OrderedDict()
    assert tag not in self.smp_data[id_sample], "Tag '%s' already present for sample '%s'" % (tag, id_sample)
    self.smp_data[id_sample][tag] = value
  
  def to_vcf_record(self, lst_samples) -> vcf.model._Record:
    assert set(lst_samples).issubset(set(list(self.smp_data.keys())))
    rec = vcf.model._Record(
      CHROM = self.chrom,
      POS = self.pos+1,
      ID = self.id,
      REF = self.ref,
      ALT = [vcf.model._Substitution(x) for x in self.alt],
      QUAL = self.qual,
      FILTER = None,
      INFO = None,
      FORMAT = ':'.join(self.formats),
      sample_indexes = {s: i for i, s in enumerate(lst_samples)}
    )
    samp_fmt = vcf.model.make_calldata_tuple(self.formats)
    samples = [vcf.model._Call(rec, smp, samp_fmt(*(list(vals.values())))) 
               for smp, vals in self.smp_data.items() if smp in lst_samples]
    rec.samples = samples
    return rec

def parse_ref_genome(fn_fasta):
  dict_chrom_seq = OrderedDict()
  dict_chrom_len = OrderedDict()

  for rec in SeqIO.parse(fn_fasta, format = 'fasta'):
    dict_chrom_seq[rec.id] = rec.seq
    dict_chrom_len[rec.id] = len(rec.seq)
  
  return dict_chrom_seq, dict_chrom_len

def parse_vcf(lst_fn_vcf, lst_id_sample, dict_chrom_pos):
  '''
  Parse input VCF file.

  Identifies different types of mutations by the ID field:
    - somatic:    ID ~ /^s.+/
    - seq. error: ID ~ /\\./

  Returns:
    - number of somatic variants
    - dict of somatic variants
  
  Updates:
    - dict_chrom_pos : dict of somatic variant loci
  '''
  
  dict_var = { smp : {} for smp in lst_id_sample}
  dict_chrom_pos = {} # variant loci that are not errors (for ISM)
  dict_som = {} # somatic variant loci (to calc total number)
  num_som_vars = 0

  # sanity checks
  assert len(lst_fn_vcf) == len(lst_id_sample)

  for i, smp in enumerate(lst_id_sample):
    rdr = vcf.Reader(filename=lst_fn_vcf[i])

    # sanity checks
    assert 'DP' in rdr.formats
    assert 'AD' in rdr.formats
    assert len(rdr.samples) == 1
    assert rdr.samples[0] == smp
    
    # parse VCF records
    for rec in rdr:
      chrom = rec.CHROM
      pos = rec.POS
      id = rec.ID
      ref = rec.REF
      alt = [str(x) for x in rec.ALT]
      cdata = rec.samples[0]
      dp = cdata['DP']
      ad = cdata['AD']
      
      var = Variant(chrom, pos, id, ref, alt)
      #var.add_format('DP', dp)
      #var.add_format('AD', ad)

      if chrom in dict_var[smp]:
        # NOTE: this allows only for one variant per position
        dict_var[smp][chrom][pos] = var
      else:
        dict_var[smp][chrom] = {pos: var}
        
      if not var.id is None: # real variant (not a seq error)
        if chrom in dict_chrom_pos:
          dict_chrom_pos[chrom].add(pos)
        else:
          dict_chrom_pos[chrom] = set([pos])
        if var.id.startswith('s'): # somatic variant (not germline)
          if chrom in dict_som:
            dict_som[chrom].add(pos)
          else:
            dict_som[chrom] = set([pos])
    
    # calculate number of somatic vars
    num_som_vars = sum([len(s) for s in dict_som.values()])

  return num_som_vars, dict_var

def write_vcfs(
  lst_fn_vcf,      # input VCF files
  #df_prev,         # prevalence matrix
  dict_smp_loc_var # new variants to be spiked into input
):
  # generate new filenames
  lst_fn_out = [fn.replace('.vcf', '.spikein.vcf') for fn in lst_fn_vcf]

  # loop over input VCFs
  for idx_vcf, fn_vcf in enumerate(lst_fn_vcf):
    smp = os.path.basename(fn_vcf).split('.')[0]
    rdr = vcf.Reader(filename=fn_vcf)
    fh_out = open(lst_fn_out[idx_vcf], 'w')
    wtr = vcf.Writer(fh_out, rdr)
    for chrom, pos_var in dict_smp_loc_var[smp].items():
      for pos, vars in pos_var.items():
        for v in vars:
          rec = v.to_vcf_record([smp])
          wtr.write_record(rec)
    fh_out.close()
  

def simulate_muts(
  num_new_vars,
  dict_chrom_seq,
  dict_chrom_len,
  smp_chrom_var,
  chrom_pos,
  is_inf_sites
):
  ''' Generate new mutations'''
  lst_new_vars = []
  dict_new_vars = {} # { id_sample: { { chrom: { pos: [var1,...] } } } }
  lst_chrom = list(dict_chrom_len.keys())

  for i in range(num_new_vars):
    is_loc_ok = False
    while not is_loc_ok:
      # pick random chromosome
      chrom = np.random.choice(lst_chrom)
      # pick random position
      pos = np.random.choice(dict_chrom_len[chrom])
      if is_inf_sites: # enforce infinite sites
        is_loc_ok = (pos not in chrom_pos[chrom])
      else: # don't enforce infinite sites
        is_loc_ok = True
      
      if not is_loc_ok: # start over if locus can't be used
        continue

      # get REF allele
      ref = dict_chrom_seq[chrom][pos]
      # pick ALT allele
      alt = np.random.choice(list(set('ACGT').difference(ref)))

      # create new variant
      id = 'n{}'.format(i)
      v = Variant(chrom, pos, id, ref, alt)
      lst_new_vars.append(v)
      # TODO: check if ISM, append to list on pos collision
      #if chrom in dict_new_vars:
      #  dict_new_vars[chrom][pos] = [v]
      #else:
      #  dict_new_vars[chrom] = {pos: [v]}
  
  return lst_new_vars

def simulate_read_counts(
  list_vars,
  df_prev,
  dist_dp,
  dist_vaf,
  fn_var_clone
):
  '''Assign each variant to a clone, include it in the samples where clone is present.'''
  # remove healthy clone from prevalence matrix (should not receive new variants)
  df_prev_min = df_prev.drop(columns=['healthy'])
  # get samples and clones from prevalence matrix
  lst_smp   = df_prev_min.index
  lst_clone = df_prev_min.columns
  # normalize matrix so that rows sum to 1
  #df_prev_norm = df_prev_min.div(df_prev_min.sum(axis=1), axis=0)
  # add row names as column
  df_prev_min.loc[:,'id_sample'] = df_prev_min.index
  # convert to long format
  df_pres = pd.melt(df_prev_min, id_vars=['id_sample'], var_name='id_clone', value_name='prev')
  df_pres['present'] = df_pres['prev'].apply(lambda p: True if p>0 else False)

  # assign new variants to clones (divide in equal proportions)
  #lst_ass = np.random.choice(lst_clone, size=num_var, replace=True)
  
  # write variant-to-clone assignment to file
  f_var_clone = open(fn_var_clone, 'wt')
  f_var_clone.write('chrom_pos,id_clone,id_mut\n')

  # compile list of vars for each regional sample
  smp_loc_var = {k : {} for k in df_prev.index}
    
  print(df_prev)
  print(df_prev_min)
  for v in list_vars:
    id_clone = np.random.choice(lst_clone)
    f_var_clone.write('{}_{},{},{}\n'.format(v.chrom, v.pos, id_clone, v.id))
    list_smp = df_pres[(df_pres.id_clone == id_clone) & df_pres.present]['id_sample']

    # sample read counts for variants in samples
    cnt = 0 # counts number of retries
    is_sampled = False # indicates if variant has rc_alt>0 in any sample
    while not is_sampled:
      #import pdb; pdb.set_trace()
      list_dp = []
      list_rc_alt = []
      print("%s samples for clone %s" % (len(list_smp), id_clone))
      for smp in list_smp:
        # sample depth from statistical distribution
        dp = dist_dp.sample(1)[0]
        list_dp.append(dp)
        # sample VAFs from statistical distribution
        vaf = dist_vaf.sample(1)[0]
        # sample ALT depth
        dist_rc_alt = stat.Distribution('Binom:{},{}'.format(dp, vaf))
        rc_alt = dist_rc_alt.sample(1)[0]
        print("vaf: %.4f, rc_alt: %d" % (vaf, rc_alt))
        list_rc_alt.append(rc_alt)
        if rc_alt > 0:
          is_sampled = True
      if not is_sampled:
        cnt += 1
        print("Retry sampling (%d)" % cnt)
      else:
        print("Success!")
    
    assert is_sampled, "Variant has ALT read count of zero in all samples."
    for i, smp in enumerate(list_smp):
      dp = list_dp[i]
      rc_alt = list_rc_alt[i]
      v.add_format(smp, 'DP', dp)
      v.add_format(smp, 'AD', [dp-rc_alt, rc_alt])

      # store variant for samples including clone
      if v.chrom not in smp_loc_var[smp]:
        smp_loc_var[smp][v.chrom] = {}
      if v.pos not in smp_loc_var[smp][v.chrom]:
        smp_loc_var[smp][v.chrom][v.pos] = [v]
      else:
        smp_loc_var[smp][v.chrom][v.pos].append(v)
      
  f_var_clone.close()

  return smp_loc_var
    

def main(
  lst_fn_vcf   : List[str],
  df_prev      : pd.core.frame.DataFrame,
  fh_ref,
  mut_load,
  dist_dp      : stat.Distribution,
  dist_vaf     : stat.Distribution,
  is_inf_sites : bool
):
  lst_smp = df_prev.index
  # load reference genome (to make REF alleles available)
  dict_chrom_seq, dict_chrom_len = parse_ref_genome(fh_ref)
  # initialize locus list from reference
  chrom_pos = {chrom: [] for chrom in dict_chrom_len.keys()}

  # extract variants from VCF files
  num_som_vars, smp_chrom_var = parse_vcf(lst_fn_vcf, lst_smp, chrom_pos)

  # determine number of mutations to add
  num_new_vars = int(mut_load * num_som_vars)

  print('Found {} somatic mutations.'.format(num_som_vars))
  print('Generating {} new mutations.'.format(num_new_vars))

  list_new_vars = simulate_muts(
    num_new_vars,
    dict_chrom_seq,
    dict_chrom_len,
    smp_chrom_var,
    chrom_pos,
    is_inf_sites
  )

  # variant-to-clone assignments are written to this file
  fn_var_clone = os.path.join(os.path.dirname(lst_fn_vcf[0]), 'var_clone.csv')
  dict_new_vars = simulate_read_counts(
    list_new_vars,
    df_prev,
    dist_dp,
    dist_vaf,
    fn_var_clone
  )

  # output vars
  write_vcfs(lst_fn_vcf, dict_new_vars)