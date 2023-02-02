import sys, os, socket
hostname = socket.gethostname()
if hostname=='chrisd1-pc.psy.gla.ac.uk':
    homeDir = '/analyse/cdhome/'
    proj0257Dir = '/analyse/Project0257/'
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
nParticipants = 10

nCompsToTry = list(range(1,5))+list(range(5,50,5))+list(range(50,350,50))
nCompsToTryPlus = nCompsToTry + ['raw']

allSG2KT = np.zeros((nBlocks,nParticipants,len(nCompsToTry)+1))
allShapeKT = np.zeros((nBlocks,nParticipants,len(nCompsToTry)+1))

measure = 'R2'

for ss in range(nParticipants):

    # check performance with varying numbers of PCA components
    for cc,nComps in enumerate(nCompsToTry):
        tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/performancesWithNPCcomps/F{ss+1:02d}_ncomps{nComps:03d}_performances_observed.npz")
        allSG2KT[:,ss,cc] = tmp['test_'+measure+'_sgPC_observed']
        allShapeKT[:,ss,cc] = tmp['test_'+measure+'_shapePC_observed']

    tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    allSG2KT[:,ss,cc+1] = tmp['test_'+measure+'_sg_observed']
    allShapeKT[:,ss,cc+1] = tmp['test_'+measure+'_shape_observed']

nCompsToTryPlus = nCompsToTry + [350]
x = list(range(0,len(nCompsToTryPlus)))

fig, ax = plt.subplots()

ySG2 = np.mean(np.mean(allSG2KT,axis=0),axis=0)
errorSG2 = np.std(np.mean(allSG2KT,axis=0),axis=0)
ax.plot(x, ySG2, 'k', color='#1f77b4',label='styleGAN2')
ax.fill_between(x, ySG2-errorSG2, ySG2+errorSG2,alpha=0.5, edgecolor='#1f77b4', facecolor='#1f77b4')

yShape = np.mean(np.mean(allShapeKT,axis=0),axis=0)
errorShape = np.std(np.mean(allShapeKT,axis=0),axis=0)
plt.plot(x, yShape, 'k', color='#ff7f0e',label='MICA')
plt.fill_between(x, yShape-errorShape, yShape+errorShape,alpha=0.5, edgecolor='#ff7f0e', facecolor='#ff7f0e')

plt.xticks(x,nCompsToTry+['raw'])
if measure=='KT':
    ax.set_ylabel(r"Kendall's $\tau$")
elif measure=='R2':
    ax.set_ylabel('$R^2$')
ax.set_xlabel('number of PCs')
ax.legend(loc="upper left")

#plt.show()

# save figure as pdf
fig = ax.get_figure()
fig.set_size_inches(10,6)
fig.savefig(f"{proj0257Dir}studentProjects/florian/figures/performance_{measure}_WithNpcaComps.pdf",dpi=300)


# look at performance of joint feature space for different numbers of components
measure = 'KT'
nCompsToTryPlus = nCompsToTry + ['raw']
allJoint = np.zeros((nBlocks,nParticipants,len(nCompsToTryPlus),len(nCompsToTryPlus)))
outBaseDir = f"{proj0257Dir}/studentProjects/florian/analysisResults/"

for ss in range(10):

    # check performance with varying numbers of PCA components
    for c1,nCompsSG in enumerate(nCompsToTryPlus):
        for c2,nCompsSh in enumerate(nCompsToTryPlus):

            if nCompsSG!='raw':
                nCompsSgTxt = f"{nCompsSG:03d}"
            elif nCompsSG=='raw':
                nCompsSgTxt = nCompsSG

            if nCompsSh!='raw':    
                nCompsShTxt = f"{nCompsSh:03d}"
            elif nCompsSh=='raw':
                nCompsShTxt = nCompsSh

            tmp = np.load(f"{outBaseDir}/performancesWithNPCcomps/F{ss+1:02d}_ncompsSG{nCompsSgTxt}_ncompsSh{nCompsShTxt}_performances_observed.npz")
            allJoint[:,ss,c1,c2] = tmp['test_'+measure+'_joint_observed']


# make image plot
data = np.mean(allJoint[:,0,:,:],axis=0)
data = np.mean(np.mean(allJoint,axis=0),axis=0) # average across folds, then across participants
fig, ax = plt.subplots()
pos = ax.imshow(data, origin='lower',vmin=0)
ax.set_xticks(range(len(nCompsToTryPlus)))
ax.set_xticklabels(nCompsToTryPlus)
ax.set_yticks(range(len(nCompsToTryPlus)))
ax.set_yticklabels(nCompsToTryPlus)
ax.set_xlabel('number of MICA PCs')
ax.set_ylabel('number of styleGAN2 PCs')
fig.colorbar(pos, ax=ax)

# np.max(data)
# 0.3567110450723226
# vs mean performance of shape alone:
# 0.3574478167517171
# joint never better

#plt.show()

# save figure as pdf
fig.set_size_inches(9,9)
fig.savefig(f"{proj0257Dir}studentProjects/florian/figures/performance_joint_{measure}_WithNpcaComps.pdf",dpi=300)