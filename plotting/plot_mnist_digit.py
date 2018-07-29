#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('index', help='Index number of dataset to plot')
args = parser.parse_args()

n = int(args.index)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

imagepath = '../data/mnist/mnist_training_images.dat'
labelpath = '../data/mnist/mnist_training_labels.dat'

images = np.reshape(np.fromfile(imagepath, dtype='float32'), [50000, 784])
labels = np.fromfile(labelpath, dtype='float32')

digit = np.reshape(images[n,:], [28, 28])[-1::-1,:]
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, aspect='equal')
ax.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(range(0, 32, 4))
plt.yticks(range(0, 32, 4))
plt.pcolor(digit, cmap=cm.binary, vmin=0, vmax=1)
plt.colorbar(shrink=0.8, ticks=np.arange(0, 1.1, 0.1))
plt.title('MNIST training sample ' + str(n) + '; Label = ' + str(int(labels[n])), fontsize=16)
plt.savefig('digit_' + '%2.2i' % n + '.svg')
plt.close(fig)
