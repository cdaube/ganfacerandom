import sys, os, socket
hostname = socket.gethostname()
if hostname=='chrisd1-pc.psy.gla.ac.uk':
    homeDir = '/analyse/cdhome/'
    proj0257Dir = '/analyse/Project0257/'
    localHome = '/home/chrisd/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    proj0257Dir = '/analyse/Project0257/'
import numpy as np
import numpy.matlib as npmtlb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

nBlocks = 10
nParticipants = 7

trainSubsets = range(5,205,5)

allSG2KT = np.zeros((nBlocks,nParticipants,len(trainSubsets)))
allShapeKT = np.zeros((nBlocks,nParticipants,len(trainSubsets)))

measure = 'KT'

for ss in range(nParticipants):

    # check performance with varying numbers of PCA components
    for cc,nTrlPerFold in enumerate(trainSubsets):
        tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/performancesSubset/F{ss+1:02d}_nTrlPerFold{nTrlPerFold:03d}_performances_observed.npz")
        allSG2KT[:,ss,cc] = tmp['test_'+measure+'_sg_observed']
        allShapeKT[:,ss,cc] = tmp['test_'+measure+'_shape_observed']


x = list(trainSubsets)

fig, ax = plt.subplots()

ySG2 = np.mean(np.mean(allSG2KT,axis=0),axis=0)
errorSG2 = np.std(np.mean(allSG2KT,axis=0),axis=0)
ax.plot(x, ySG2, 'k', color='#1f77b4',label='styleGAN2')
ax.fill_between(x, ySG2-errorSG2, ySG2+errorSG2,alpha=0.5, edgecolor='#1f77b4', facecolor='#1f77b4')

yShape = np.mean(np.mean(allShapeKT,axis=0),axis=0)
errorShape = np.std(np.mean(allShapeKT,axis=0),axis=0)
plt.plot(x, yShape, 'k', color='#ff7f0e',label='MICA')
plt.fill_between(x, yShape-errorShape, yShape+errorShape,alpha=0.5, edgecolor='#ff7f0e', facecolor='#ff7f0e')

#plt.xticks(x,nCompsToTry+['raw'])
if measure=='KT':
    ax.set_ylabel(r"Kendall's $\tau$")
elif measure=='R2':
    ax.set_ylabel('$R^2$')
ax.set_xlabel('Trials per fold used for training')
ax.legend(loc="upper left")

#plt.show()

# save figure as pdf
fig = ax.get_figure()
fig.set_size_inches(10,6)
fig.savefig(f"{localHome}/ownCloud/FiguresGANface/performance_{measure}_trainOnSubset.pdf",dpi=300)

