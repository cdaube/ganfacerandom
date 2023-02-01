from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import Ridge
from skopt import BayesSearchCV
from scipy.io import loadmat
from scipy.stats import kendalltau
import numpy as np
import datetime
import warnings

# helper to get p value from observed and permuted statistic
def getP(observed,permuted):
    nPermutations = permuted.size
    tmpP = np.sum(permuted>observed)/nPermutations
    if tmpP == 0:
        tmpP = 1/nPermutations

    return tmpP

# helper to get output of inplace functiom
def shuffled(input):
    np.random.shuffle(input)
    return(input)

# helper to extract block wise Kendall's Tau of predictions
def extractKT(Xsplit,Ysplit,mdl,nBlocks):
    # preallocation of output structure
    test_KT_observed = np.zeros((nBlocks))
    # loop through blocks
    for bb in range(nBlocks):
        # get this block's model and predict responses
        thsPredictions = mdl['estimator'][bb].predict(Xsplit[bb])
        thsObservations = Ysplit[bb]
        # measure Kendall's Tau
        test_KT_observed[bb] = kendalltau(thsPredictions,thsObservations,method='asymptotic')[0]

    return test_KT_observed


def loadOrigOrderLatents(nTrials,nDimsSG2,nDimsShape,nBlocks,nTrialsPerBlock,proj0257Dir):
    # load styleGAN2 latents and shape latents order of original experiment
    XsgOrig = np.zeros((nTrials,nDimsSG2))
    XshapeOrig = np.zeros((nTrials,nDimsShape))
    for bb in range(nBlocks):
        # shape latents are stored per trial
        for tt in range(nTrialsPerBlock):
            tmp = np.load(f"{proj0257Dir}studentProjects/abbie/face_recon/block_{bb+1}/{tt+1}_shape.npy")
            XshapeOrig[bb*nTrialsPerBlock+tt,:] = tmp

        # styleGAN2 latents are stored block wise
        tmp = loadmat(f"{proj0257Dir}studentProjects/abbie/stimuli/block_{bb+1}/latents_{bb+1}.mat")
        thsStart = 0 + bb*nTrialsPerBlock
        thsEnd = nTrialsPerBlock + bb*nTrialsPerBlock
        XsgOrig[thsStart:thsEnd,:] = tmp['latents']
        
    return XsgOrig, XshapeOrig


def reorderXShape(nTrials,nDimsShape,XsgOrig,XshapeOrig,Xsg):
    # format shape predictors in order of this participant
    # preallocate Xshape matrix with inferred shape latents in correct order for current participant
    Xshape = np.zeros((nTrials,nDimsShape))

    # go through each row of styleGAN2 latents in order of current participant
    for r1 in range(Xsg.shape[0]):

        # preallocate vector of where this stimulus was used in orig order
        thsInstances = np.zeros((Xsg.shape[0]))

        # go through XsgOrig and search for rows where current stimulus occurs
        for r2 in range(Xsg.shape[0]):
            thsInstances[r2] = (Xsg[r1,:]==XsgOrig[r2,:]).all()

        # if this stimulus 
        if np.sum(thsInstances)==1:
            # then get the indices
            idx = np.argwhere(thsInstances)
            # store first duplicate response
            Xshape[r1,:] = XshapeOrig[idx[0][0],:]

    return Xshape


def measureNoiseCeiling(X,Y,nDuplicates,nPermutations):

    # preallocate vectors for responses to stimuli presented twice
    rep1 = np.zeros(nDuplicates)
    rep2 = np.zeros(nDuplicates)

    # initiate duplicate counter
    duplicateCounter = 0

    for rr in range(X.shape[0]):

        # preallocate vector of where this stimulus was used
        thsInstances = np.zeros((X.shape[0]))

        # go through X and search for rows where current stimulus occurs
        for r2 in range(X.shape[0]):
            thsInstances[r2] = (X[rr,:]==X[r2,:]).all()

        # if this stimulus has occurred more than once and we haven't already found all 200 duplicates
        if np.sum(thsInstances)>1 and duplicateCounter < 200:
            # then get the indices
            idx = np.argwhere(thsInstances)
            # store first duplicate response
            rep1[duplicateCounter] = Y[idx[0]]
            # store 2nd duplicate response
            rep2[duplicateCounter] = Y[idx[1]]
            # increase duplicate counter
            duplicateCounter += 1

    # compute Kendall's Tau
    kt = kendalltau(rep1,rep2)
    # store the actual KT value
    retest_KT = kt[0]
    retest_KT_p_param = kt[1]

    retest_KT_perm = np.zeros((nPermutations))
    for pp in range(nPermutations):
        kt = kendalltau(rep1,shuffled(rep2))
        retest_KT_perm[pp] = kt[0]
        if np.mod(pp,100)==0:
            print(f"noise ceiling permutation {pp+1}")
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S"))

    return retest_KT, retest_KT_p_param, retest_KT_perm


def trainModelExtractPerformances(X, Y, nBlocks, n_iter=30):

    Ysplit = np.array_split(Y.flatten(),nBlocks,axis=0)
    Xsplit = np.array_split(X,nBlocks,axis=0)

    # define cross-validation for inner and outer loops
    cvOuter = KFold(nBlocks)
    cvInner = KFold(nBlocks-1)
    # define hyperparameter optimisation
    search = {'alpha': (1e-4, 1e6, "log-uniform")}
    bayesRidgeTuner = BayesSearchCV(estimator=Ridge(), search_spaces=search, cv=cvInner, n_iter=n_iter)
    # train model on observed data
    mdl = cross_validate(bayesRidgeTuner, X, Y, cv=cvOuter, 
                         return_estimator=True, 
                         return_train_score=True, 
                         verbose=2,
                         n_jobs=10)
    # extract outputs (performances)
    test_R2_sg_observed = mdl['test_score']
    train_R2_sg_observed = mdl['train_score']
    test_KT_sg_observed = extractKT(Xsplit,Ysplit,mdl,nBlocks)

    allCoef = np.zeros((X.shape[1],nBlocks))
    for ii, thsMdl in enumerate(mdl['estimator']):
        allCoef[:,ii] = thsMdl.best_estimator_.coef_

    return test_R2_sg_observed, train_R2_sg_observed, test_KT_sg_observed, allCoef


def perm_trainModelExtractPerformances(X, Y, nBlocks, nPermutations, ss, n_iter=30):

    # preallocate outputs of permutations
    test_R2_perm = np.zeros((nPermutations,nBlocks))
    test_KT_perm = np.zeros((nPermutations,nBlocks))
    train_R2_perm = np.zeros((nPermutations,nBlocks))

    Ysplit = np.array_split(Y.flatten(),nBlocks,axis=0)
    Xsplit = np.array_split(X,nBlocks,axis=0)

    # define cross-validation for inner and outer loops
    cvOuter = KFold(nBlocks)
    cvInner = KFold(nBlocks-1)
    # loop through permutations
    for pp in range(nPermutations):
        # communicate progress
        if np.mod(pp,100)==0:
            print(f"participant {ss} permutation {pp+1}")
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S"))
        # shufflle responses
        permYsplit = [shuffled(item) for item in Ysplit]
        permY = np.array(permYsplit).flatten()
        # fit model to permuted data
        # define hyperparameter optimisation
        search = {'alpha': (1e-4, 1e6, "log-uniform")}
        bayesRidgeTuner = BayesSearchCV(estimator=Ridge(), search_spaces=search, cv=cvInner, n_iter=n_iter)
        # let's ignore the warning of having evaluated parameter combination previously
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdlPerm_sg = cross_validate(bayesRidgeTuner, X, permY, cv=cvOuter, 
                                     return_estimator=True, 
                                     return_train_score=True, 
                                     verbose=0,
                                     n_jobs=10)
        # store scores
        test_R2_perm[pp,:] = mdlPerm_sg['test_score']
        train_R2_perm[pp,:] = mdlPerm_sg['train_score']
        test_KT_perm[pp,:] = extractKT(Xsplit,permYsplit,mdlPerm_sg,nBlocks)

    return test_R2_perm, test_KT_perm, train_R2_perm