import sys, socket
hostname = socket.gethostname()
if hostname=='chrisd1-pc.psy.gla.ac.uk':
    homeDir = '/analyse/cdhome/'
    proj0257Dir = '/analyse/Project0257/'
elif hostname[0:7]=='deepnet':
    homeDir = '/home/chrisd/'
    proj0257Dir = '/analyse/Project0257/'
elif hostname[0:8]=='compute-':
    homeDir = '/home/chrisd/'
    proj0257Dir = '/analyse/Project0257/'

import argparse
from scipy.io import loadmat
from scipy.stats import zscore
import numpy as np
from facePred_utils import *
from sklearn.decomposition import PCA

outBaseDir = f"{proj0257Dir}/studentProjects/florian/analysisResults/"

def main(args):

    args = parseArgs(args)

    nBlocks = 10
    nPermutations = 1000
    nTrials = 2000
    nRepeatedTrials = 200
    nTrialsPerBlock = int(nTrials/nBlocks)
    nDimsSG2 = 512
    nDimsShape = 300
    nCompsToTry = list(range(1,5))+list(range(5,50,5))+list(range(50,350,50))
    nCompsToTryPlus = nCompsToTry + ['raw']
    trainSubsets = range(5,205,5)

    # load styleGAN2 and shape latents in original order (does not correspond to experimental order of any participant)
    XsgOrig, XshapeOrig = loadOrigOrderLatents(nTrials,nDimsSG2,nDimsShape,nBlocks,nTrialsPerBlock,proj0257Dir)

    # main loop through participants
    for ss in args.ss:

        # load data aligned for this participant
        tmp = loadmat(f"{proj0257Dir}/studentProjects/florian/expData/F{ss+1:02d}fulldata.mat")
        Xsg = tmp['X']
        Y = tmp['Y']
        Xshape = reorderXShape(nTrials,nDimsShape,XsgOrig,XshapeOrig,Xsg)
        
        if args.noiseceiling:
            # estimate noise ceiling
            retest_KT, retest_KT_p, retest_KT_perm = measureNoiseCeiling(Xsg,Y,nRepeatedTrials,nPermutations)

            # save this participant's retest KT results
            np.savez(f"{outBaseDir}/F{ss+1:02d}_retestKT.npz",
                    retest_KT=retest_KT, retest_KT_p=retest_KT_p, retest_KT_perm=retest_KT_perm)

        if args.observedperformance:
            # assess performance of styleGAN2 latents
            test_R2_sg_observed, train_R2_sg_observed, test_KT_sg_observed, coef_sg = \
                trainModelExtractPerformances(Xsg, Y, nBlocks)
            # assess performance of shape latents (MICA)
            test_R2_shape_observed, train_R2_shape_observed, test_KT_shape_observed, coef_shape = \
                trainModelExtractPerformances(Xshape, Y, nBlocks)
            # also assess performance of joint feature space
            Xjoint = zscore(np.hstack((Xsg,Xshape)),axis=0)
            test_R2_joint_observed, train_R2_joint_observed, test_KT_joint_observed, coef_joint = \
                trainModelExtractPerformances(Xjoint, Y, nBlocks)

            # save this participant's performances on observed data
            np.savez(f"{outBaseDir}/F{ss+1:02d}_performances_observed.npz", 
                    test_R2_sg_observed=test_R2_sg_observed,
                    test_KT_sg_observed=test_KT_sg_observed,
                    train_R2_sg_observed=train_R2_sg_observed,
                    coef_sg=coef_sg,
                    test_R2_shape_observed=test_R2_shape_observed,
                    test_KT_shape_observed=test_KT_shape_observed,
                    train_R2_shape_observed=train_R2_shape_observed,
                    coef_shape=coef_shape,
                    test_R2_joint_observed=test_R2_joint_observed,
                    test_KT_joint_observed=test_KT_joint_observed,
                    train_R2_joint_observed=train_R2_joint_observed,
                    coef_joint=coef_joint)

        if args.permutations:
            # measure performance of styleGAN2 when trials are shuffled
            test_R2_sg_perm, test_KT_sg_perm, train_R2_sg_perm = perm_trainModelExtractPerformances(Xsg, Y, nBlocks, nPermutations, ss)
            # measure performance of styleGAN2 when trials are shuffled
            test_R2_shape_perm, test_KT_shape_perm, train_R2_shape_perm = perm_trainModelExtractPerformances(Xsg, Y, nBlocks, nPermutations, ss)

            # save this participant's permutation results
            np.savez(f"{outBaseDir}/F{ss+1:02d}_performances_perm.npz", 
                    test_R2_sg_perm=test_R2_sg_perm, 
                    test_KT_sg_perm=test_KT_sg_perm,
                    train_R2_sg_perm=train_R2_sg_perm,
                    test_R2_shape_perm=test_R2_shape_perm, 
                    test_KT_shape_perm=test_KT_shape_perm, 
                    train_R2_shape_perm=train_R2_shape_perm)

        if args.pcamarginal: 
            # check performance with varying numbers of PCA components
            for nComps in nCompsToTry:

                XsgPC = zscore(PCA(n_components=nComps).fit_transform(Xsg))
                XshapePC = zscore(PCA(n_components=nComps).fit_transform(Xshape))

                # assess performance of styleGAN2 latents
                test_R2_sgPC_observed, train_R2_sgPC_observed, test_KT_sgPC_observed, __ = \
                    trainModelExtractPerformances(XsgPC, Y, nBlocks)
                # assess performance of shape latents (MICA)
                test_R2_shapePC_observed, train_R2_shapePC_observed, test_KT_shapePC_observed, __ = \
                    trainModelExtractPerformances(XshapePC, Y, nBlocks)

                np.savez(f"{outBaseDir}/performancesWithNPCcomps/F{ss+1:02d}_ncomps{nComps:03d}_performances_observed.npz", 
                    test_R2_sgPC_observed=test_R2_sgPC_observed,
                    test_KT_sgPC_observed=test_KT_sgPC_observed,
                    train_R2_sgPC_observed=train_R2_sgPC_observed,
                    test_R2_shapePC_observed=test_R2_shapePC_observed,
                    test_KT_shapePC_observed=test_KT_shapePC_observed,
                    train_R2_shapePC_observed=train_R2_shapePC_observed)

        if args.pcajoint:
            # check performance of joint feature space with varying numbers of PCA components
            for nCompsSG in nCompsToTryPlus:
                for nCompsSh in nCompsToTryPlus:

                    if nCompsSG!='raw':
                        XsgPC = zscore(PCA(n_components=nCompsSG).fit_transform(Xsg))
                        nCompsSgTxt = f"{nCompsSG:03d}"
                    elif nCompsSG=='raw':
                        XsgPC = zscore(Xsg)
                        nCompsSgTxt = nCompsSG

                    if nCompsSh!='raw':    
                        XshapePC = zscore(PCA(n_components=nCompsSh).fit_transform(Xshape))
                        nCompsShTxt = f"{nCompsSh:03d}"
                    elif nCompsSh=='raw':
                        XshapePC = zscore(Xshape)
                        nCompsShTxt = nCompsSh

                    # assess performance of joint feature space
                    Xjoint = np.hstack((XsgPC,XshapePC))
                    test_R2_joint_observed, train_R2_joint_observed, test_KT_joint_observed, coef_joint = \
                        trainModelExtractPerformances(Xjoint, Y, nBlocks)

                    fileName = f"{outBaseDir}/performancesWithNPCcomps/F{ss+1:02d}_ncompsSG{nCompsSgTxt}_ncompsSh{nCompsShTxt}_performances_observed.npz"
                    np.savez(fileName, 
                        test_R2_joint_observed=test_R2_joint_observed,
                        test_KT_joint_observed=test_KT_joint_observed,
                        train_R2_joint_observed=train_R2_joint_observed,
                        coef_joint=coef_joint)

        if args.trainOnSubset:
            for nTrlPerFold in trainSubsets:
                test_R2_sg_observed, train_R2_sg_observed, test_KT_sg_observed, __, test_R2_sg_observed_subset = \
                    subset_trainModelExtractPerformances(Xsg, Y, nBlocks, nTrlPerFold)
                test_R2_shape_observed, train_R2_shape_observed, test_KT_shape_observed, __, test_R2_shape_observed_subset = \
                    subset_trainModelExtractPerformances(Xshape, Y, nBlocks, nTrlPerFold)

                np.savez(f"{outBaseDir}/performancesSubset/F{ss+1:02d}_nTrlPerFold{nTrlPerFold:03d}_performances_observed.npz", 
                    test_R2_sg_observed=test_R2_sg_observed,
                    test_R2_sg_observed_subset=test_R2_sg_observed_subset,
                    test_KT_sg_observed=test_KT_sg_observed,
                    train_R2_sg_observed=train_R2_sg_observed,
                    test_R2_shape_observed=test_R2_shape_observed,
                    test_R2_shape_observed_subset=test_R2_shape_observed_subset,
                    test_KT_shape_observed=test_KT_shape_observed,
                    train_R2_shape_observed=train_R2_shape_observed)


def parseArgs(argv):
    # parameters determine what parts of the code is run
    parser = argparse.ArgumentParser(description='analyser_main')

    parser.add_argument('--ss',type=int,nargs='+',default=[1,2,3,4,5],
                        help="Participant selection.")
    parser.add_argument('--noiseceiling', action='store_true') # store_true -> if argument is added, this is true, if argument is not added, this is false
    parser.add_argument('--observedperformance', action='store_true')
    parser.add_argument('--permutations', action='store_true')
    parser.add_argument('--pcamarginal', action='store_true')
    parser.add_argument('--pcajoint', action='store_true')
    parser.add_argument('--trainOnSubset', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)