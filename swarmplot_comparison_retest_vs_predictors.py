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
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KernelDensity

nBlocks = 10
nParticipants = 10
nPermutations = 1000

# for permutation distribution kde
kdeRescale = 4
queryPoints=np.linspace(-.4,.2,100)[:,None]

# load data
allRetestKT = []
allSG2KT = []
allShapeKT = []
allkdeRetest = np.zeros((queryPoints.size,nParticipants))
allkdeSG2 = np.zeros((queryPoints.size,nParticipants))
allkdeShape = np.zeros((queryPoints.size,nParticipants))
for ss in range(nParticipants):

    # load data
    tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_retestKT.npz")
    allRetestKT.append(tmp['retest_KT'])
    tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    allSG2KT.append(tmp['test_KT_sg_observed'])
    allShapeKT.append(tmp['test_KT_shape_observed'])

    # load permutations and do kernel density estimates on them
    tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_retestKT.npz")
    kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(tmp['retest_KT_perm'].reshape(-1,1))
    allkdeRetest[:,ss] = np.exp(kde.score_samples(queryPoints))

    tmpPerm = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_perm.npz")
    #kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(np.mean(tmpPerm['test_KT_sg_perm'],axis=1).reshape(-1,1))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(tmpPerm['test_KT_sg_perm'].reshape(-1,1))
    allkdeSG2[:,ss] = np.exp(kde.score_samples(queryPoints))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.02).fit(tmpPerm['test_KT_shape_perm'].reshape(-1,1))
    allkdeShape[:,ss] = np.exp(kde.score_samples(queryPoints))

# rescale KDEs between 0 and 1
allkdeRetest = allkdeRetest/np.max(allkdeRetest)/kdeRescale
allkdeSG2 = allkdeSG2/np.max(allkdeSG2)/kdeRescale
allkdeShape = allkdeShape/np.max(allkdeShape)/kdeRescale

# define participant and block indices
participantIdx = np.repeat(list(range(1,nParticipants+1)),nBlocks)
blockIdx = np.squeeze(npmtlb.repmat(list(range(nBlocks)),1,nParticipants))
df_x = pd.DataFrame({'Participants':participantIdx, 'Blocks': blockIdx})

# for df, create vector of retest KT for block 1 and NaNs for all other blocks
retestKTfordf = npmtlb.repmat(np.asarray(allRetestKT),nBlocks,1)
retestKTfordf[1:,:] = np.NaN
df_retestKT = pd.DataFrame({'retest': retestKTfordf.T.flatten()})
# create dataframe of performances of all predictors (stacked such that we have nParticipants blocks of data)
df_predictors = pd.DataFrame({'styleGAN2':np.reshape(np.asarray(allSG2KT),(nBlocks*nParticipants)),
                              'MICA':np.reshape(np.asarray(allShapeKT),(nBlocks*nParticipants))})
# assemble all into one dataframe
df_full = pd.concat([df_x, df_retestKT, df_predictors], axis=1) 
# convert to long format for seaborn swarmplot
df_long = df_full.melt(id_vars=['Participants','Blocks'], var_name='Predictor', value_name='tau')
# save data frame
df_long.to_csv(path_or_buf=f"{proj0257Dir}studentProjects/florian/df_long.csv")

# call seaborn swarmplot
ax = sns.swarmplot(data=df_long,x='Predictor',y='tau',hue='Participants',palette='Set3',
                    linewidth=1, alpha = 1)
sns.boxplot(x='Predictor', y='tau', data=df_long, 
                 showcaps=False,boxprops={'facecolor':'None'},
                 showfliers=False,whiskerprops={'linewidth':0}, ax=ax)
ax.set_ylabel(r"Kendall's $\tau$")
ax.set_ylim(-.2, .6)

cMap = sns.color_palette("Set3")

# add permutation distributions averaged across folds
for ss in range(nParticipants):
    ax.fill_between(allkdeRetest[:,ss],np.squeeze(queryPoints),alpha = .05,color=cMap[ss])
    ax.fill_between(allkdeSG2[:,ss]+1,np.squeeze(queryPoints),alpha = .05,color=cMap[ss])
    ax.fill_between(allkdeShape[:,ss]+2,np.squeeze(queryPoints),alpha = .05,color=cMap[ss])
    ax.plot(allkdeRetest[:,ss],np.squeeze(queryPoints),color=cMap[ss])
    ax.plot(allkdeSG2[:,ss]+1,np.squeeze(queryPoints),color=cMap[ss])
    ax.plot(allkdeShape[:,ss]+2,np.squeeze(queryPoints),color=cMap[ss])

plt.grid()
plt.show()

# save figure as pdf
fig = ax.get_figure()
fig.set_size_inches(8,6)
fig.savefig(f"{proj0257Dir}studentProjects/florian/figures/swarmPlot_KT.pdf",dpi=300)