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

data = pd.read_csv(file+'.csv',delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(data['beta'],data['MC Walltime'],label='MC',marker='.')
ax.plot(data['beta'],data['Redist Walltime'],label='Redist',marker='.')
ax.plot(data['beta'],data['Obs Walltime'],label='Observables',marker='.')
##

ax.legend()
plt.show()
# plt.savefig(file+'.png')


