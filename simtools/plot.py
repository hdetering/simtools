#!/usr/bin/env python

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# TODO: is this method still needed?
def plot_bar(x, y, title='Barplot', ylab='variable y', xlab='category'):
  x_pos = list(range(len(x)))
  fig, ax = plt.subplots()
  ax.bar(x_pos, y, align='center', alpha=0.5)
  ax.grid()
  
  # set axis labels and title
  ax.set_ylabel(ylab)
  ax.set_xlabel(xlab)
  ax.set_xticks(range(len(x)))
  ax.set_xticklabels(x)
  ax.set_title(title)
  
  # set height of the y-axis
  plt.ylim([0, (max(y)) * 1.1])
  return fig