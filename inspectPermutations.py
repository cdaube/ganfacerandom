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
from facePred_utils import getP

nBlocks = 10
nParticipants = 10
nPermutations = 1000

allP_sg_R2 = np.zeros(nParticipants)
allP_shape_R2 = np.zeros(nParticipants)
allP_sg_KT = np.zeros(nParticipants)
allP_shape_KT = np.zeros(nParticipants)


# create subplots for retest KT
fig, axs = plt.subplots(2, 5, sharex='all', sharey='all')
fig.set_size_inches(12,6)
allP_retest = np.zeros(nParticipants)
for ss, ax in enumerate(axs.reshape(-1)[0:10]): 
    tmp = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_retestKT.npz")

    ax.hist(tmp['retest_KT_perm'], bins=25, label = 'permutations')
    ax.axvline(x = tmp['retest_KT'], color = 'r', label = 'observed')

    ax.set_title('Participant '+str(ss+1))
    ax.set_xlabel(r"Kendall's $\tau$")

    if ss==0 | ss == 5:
        ax.set_ylabel('Count')
    if ss==0:
        ax.legend(loc="upper left")

    allP_retest[ss] = getP(tmp['retest_KT'],tmp['retest_KT_perm'])

fig.suptitle("Correlation of responses to repeated trials relative to permutations", fontsize=14)
plt.savefig(f"{proj0257Dir}studentProjects/florian/figures/F_all_retest_KT.pdf",dpi=300)


# do group level figures for prediction significance (averaged across folds)
# create subplots for R2, styleGAN2
fig, axs = plt.subplots(2, 5, sharex='all', sharey='all')
fig.set_size_inches(12,6)
for ss, ax in enumerate(axs.reshape(-1)[0:10]): 
    # load data
    tmpObs = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    tmpPerm = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_perm.npz")

    ax.hist(np.mean(tmpPerm['test_R2_sg_perm'],axis=1), bins=25, label = 'permutations')
    ax.axvline(x = np.mean(tmpObs['test_R2_sg_observed']), color = 'r', label = 'observed')

    ax.set_title('Participant '+str(ss+1))
    ax.set_xlabel('$R^2$')

    if ss==0 | ss == 5:
        ax.set_ylabel('Count')
    if ss==0:
        ax.legend(loc="upper left")

fig.suptitle("Performance of model averaged across folds relative to permutations", fontsize=14)
plt.savefig(f"{proj0257Dir}studentProjects/florian/figures/F_all_R2_sg.pdf",dpi=300)

# create subplots for R2, shape
fig, axs = plt.subplots(2, 5, sharex='all', sharey='all')
fig.set_size_inches(12,6)
for ss, ax in enumerate(axs.reshape(-1)[0:10]): 
    # load data
    tmpObs = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    tmpPerm = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_perm.npz")

    ax.hist(np.mean(tmpPerm['test_R2_shape_perm'],axis=1), bins=25, label = 'permutations')
    ax.axvline(x = np.mean(tmpObs['test_R2_shape_observed']), color = 'r', label = 'observed')

    ax.set_title('Participant '+str(ss+1))
    ax.set_xlabel('$R^2$')

    if ss==0 | ss == 5:
        ax.set_ylabel('Count')
    if ss==0:
        ax.legend(loc="upper left")

fig.suptitle("Performance of model averaged across folds relative to permutations", fontsize=14)
plt.savefig(f"{proj0257Dir}studentProjects/florian/figures/F_all_R2_shape.pdf",dpi=300)


# create subplots for KT, styleGAN2
fig, axs = plt.subplots(2, 5, sharex='all', sharey='all')
fig.set_size_inches(12,6)
for ss, ax in enumerate(axs.reshape(-1)[0:10]): 
    # load data
    tmpObs = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    tmpPerm = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_perm.npz")

    ax.hist(np.mean(tmpPerm['test_KT_sg_perm'],axis=1), bins=25, label = 'permutations')
    ax.axvline(x = np.mean(tmpObs['test_KT_sg_observed']), color = 'r', label = 'observed')

    ax.set_title('Participant '+str(ss+1))
    ax.set_xlabel(r"Kendall's $\tau$")

    if ss==0 | ss == 5:
        ax.set_ylabel('Count')
    if ss==0:
        ax.legend(loc="upper left")

fig.suptitle("Performance of model averaged across folds relative to permutations", fontsize=14)
plt.savefig(f"{proj0257Dir}studentProjects/florian/figures/F_all_KT_sg.pdf",dpi=300)


# create subplots for KT, shape
fig, axs = plt.subplots(2, 5, sharex='all', sharey='all')
fig.set_size_inches(12,6)
for ss, ax in enumerate(axs.reshape(-1)[0:10]): 
    # load data
    tmpObs = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    tmpPerm = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_perm.npz")

    ax.hist(np.mean(tmpPerm['test_KT_shape_perm'],axis=1), bins=25, label = 'permutations')
    ax.axvline(x = np.mean(tmpObs['test_KT_shape_observed']), color = 'r', label = 'observed')

    ax.set_title('Participant '+str(ss+1))
    ax.set_xlabel(r"Kendall's $\tau$")

    if ss==0 | ss == 5:
        ax.set_ylabel('Count')
    if ss==0:
        ax.legend(loc="upper left")

fig.suptitle("Performance of model averaged across folds relative to permutations", fontsize=14)
plt.savefig(f"{proj0257Dir}studentProjects/florian/figures/F_all_KT_shape.pdf",dpi=300)




# single subject figures
for ss in range(nParticipants):

    # load data
    tmpObs = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_observed.npz")
    tmpPerm = np.load(f"{proj0257Dir}/studentProjects/florian/analysisResults/F{ss+1:02d}_performances_perm.npz")

    # create subplots for R2, styleGAN2
    fig, axs = plt.subplots(4, 5, sharex='all', sharey='all')
    fig.set_size_inches(12,15)

    for pp, ax in enumerate(axs.reshape(-1)[0:10]): 
        ax.hist(tmpPerm['test_R2_sg_perm'][:,pp], bins=25, label = 'permutations')
        ax.axvline(x = tmpObs['test_R2_sg_observed'][pp], color = 'r', label = 'observed')

        ax.set_title('fold '+str(pp+1)+', styleGAN2')
        ax.set_xlabel('$R^2$')

        if pp==5 | pp == 9:
            ax.set_ylabel('Count')
        if pp==0:
            ax.legend(loc="upper left")

    allP_sg_R2[ss] = getP(np.mean(tmpObs['test_R2_sg_observed']),np.mean(tmpPerm['test_R2_sg_perm'],axis=1))

    # create subplots for R2, shape
    for pp, ax in enumerate(axs.reshape(-1)[10:20]): 
        ax.hist(tmpPerm['test_R2_shape_perm'][:,pp], bins=25, label = 'permutations')
        ax.axvline(x = tmpObs['test_R2_shape_observed'][pp], color = 'r', label = 'observed')

        ax.set_title('fold '+str(pp+1)+', MICA')
        ax.set_xlabel('$R^2$')

        if pp==5 | pp == 9:
            ax.set_ylabel('Count')

    allP_shape_R2[ss] = getP(np.mean(tmpObs['test_R2_shape_observed']),np.mean(tmpPerm['test_R2_shape_perm'],axis=1))

    fig.suptitle("$R^2$ in test sets relative to permutations", fontsize=14)
    plt.savefig(f"{proj0257Dir}studentProjects/florian/figures/F{ss+1:02d}_Permutations_R2.pdf",dpi=300)

    # create subplots for KT, styleGAN2
    fig, axs = plt.subplots(4, 5, sharex='all', sharey='all')
    fig.set_size_inches(12,15)

    for pp, ax in enumerate(axs.reshape(-1)[0:10]): 
        ax.hist(tmpPerm['test_KT_sg_perm'][:,pp], bins=25, label = 'permutations')
        ax.axvline(x = tmpObs['test_KT_sg_observed'][pp], color = 'r', label = 'observed')

        ax.set_title('fold '+str(pp+1)+', styleGAN2')
        ax.set_xlabel(r"Kendall's $\tau$")

        if pp==5 | pp == 9:
            ax.set_ylabel('Count')
        if pp==0:
            ax.legend(loc="upper left")

    allP_sg_KT[ss] = getP(np.mean(tmpObs['test_KT_sg_observed']),np.mean(tmpPerm['test_KT_sg_perm'],axis=1))

    # create subplots for KT, shape
    for pp, ax in enumerate(axs.reshape(-1)[10:20]): 
        ax.hist(tmpPerm['test_KT_shape_perm'][:,pp], bins=25, label = 'permutations')
        ax.axvline(x = tmpObs['test_KT_shape_observed'][pp], color = 'r', label = 'observed')

        ax.set_title('fold '+str(pp+1)+', MICA')
        ax.set_xlabel(r"Kendall's $\tau$")

        if pp==5 | pp == 9:
            ax.set_ylabel('Count')

    fig.suptitle("$R^2$ in test sets relative to permutations", fontsize=14)
    plt.savefig(f"{proj0257Dir}studentProjects/florian/figures/F{ss+1:02d}_Permutations_KT.pdf",dpi=300)

    allP_shape_KT[ss] = getP(np.mean(tmpObs['test_KT_shape_observed']),np.mean(tmpPerm['test_KT_shape_perm'],axis=1))

