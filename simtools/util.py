#!/usr/bin/env python

def is_number(s):
  '''Check whether a given string can be parsed as a number.'''
  try:
    float(s)
    return True
  except ValueError:
    return False