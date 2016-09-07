#!/bin/python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import math
import itertools

#Python 2.7

# energy = np.loadtxt(open("test.csv","rb"),delimiter=",",usecols=(0,1))
# magnet = np.loadtxt(open("test.csv","rb"),delimiter=",",usecols=(0,2))

file = 'BT_52'

data = pd.read_csv(file+'.csv',delimiter=',',header=None,names=['beta','energy'])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(data['beta'],data['energy'],label='Output',marker='.');
##

data = pd.read_csv(file+'_test.csv',delimiter=',',header=None,names=['beta','energy'])
ax.plot(data['beta'],data['energy'],label='Test Data');

ax.set_xlabel('Beta')
ax.set_ylabel('Energy')

ax.legend()
plt.show()
# plt.savefig(file+'.png')


