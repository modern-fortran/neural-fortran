#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path', help='path to file where output of example_mnist is stored')
args = parser.parse_args()

path = args.path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

lines = [line.strip() for line in open(path).readlines()]

accuracy = []
accuracy.append(float(lines[0].split()[2]))
for line in lines[1:]:
    accuracy.append(float(line.split()[4]))

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, xlim=(0, len(accuracy)-1), ylim=(0, 100))
ax.tick_params(axis='both', which='major', labelsize=16)
plt.plot(accuracy, 'k-', marker='.', ms=12, lw=2)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy [%]', fontsize=16)
plt.grid()
plt.title('MNIST accuracy [%]', fontsize=16)
fig.subplots_adjust(bottom=0.2)
plt.savefig('mnist_accuracy.svg')
plt.close(fig)
