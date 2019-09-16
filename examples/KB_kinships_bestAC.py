#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('HRESCAL')

import numpy as np
import datetime as dt
from numpy import dot, array, zeros, setdiff1d
from numpy.linalg import norm
from numpy.random import shuffle
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve, auc ,accuracy_score
from BE import BE_als,Tensor2Matrix,HimmingDistance


def predict_BE_als(T):
    A, R, _, _, _ = BE_als(
        T, 2, init='nvecs', conv=1e-3,
        lambda_A=0.05, lambda_R=0.1
    )
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return P


def normalize_predictions(P, e, k):
    for a in range(e):
        for b in range(e):
            nrm = norm(P[a, b, :k])
            if nrm != 0:
                # round values for faster computation of AUC-PR
                P[a, b, :k] = np.round_(P[a, b, :k] / nrm, decimals=3)
    return P


def innerfold(T, mask_idx, target_idx, e, k, sz):
    Tc = [Ti.copy() for Ti in T]
    mask_idx = np.unravel_index(mask_idx, (e, e, k))
    target_idx = np.unravel_index(target_idx, (e, e, k))

    # set values to be predicted to zero
    for i in range(len(mask_idx[0])):
        Tc[mask_idx[2][i]][mask_idx[0][i], mask_idx[1][i]] = 0

    # predict unknown values
    P = predict_BE_als(Tc)
    P = normalize_predictions(P, e, k)

    # compute area under precision recall curve
    prec, recall, _ = precision_recall_curve(GROUND_TRUTH[target_idx], P[target_idx])
    ac = accuracy_score(GROUND_TRUTH[target_idx], P[target_idx].round())
    return auc(recall, prec),ac


if __name__ == '__main__':
    # load data
    st = dt.datetime.now()
    mat = loadmat('E:/experiments/rescal-bilinear/data/alyawarradata.mat')
    #mat = loadmat('F:/experiment/rescal-bilinear/data/umls.mat')
    #mat = loadmat('F:/experiment/rescal-bilinear/data/nations.mat')
    K = array(mat['Rs'], np.float32)
    #print(K[0],K.shape)
    K = Tensor2Matrix(K)
    #print(K[0],'\r',K.shape)
    K = HimmingDistance(K,-0.0005)
    #print(K)

    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k
    # copy ground truth before preprocessing
    GROUND_TRUTH = K.copy()

    # construct array for rescal
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    _log.info('Datasize: %d x %d x %d | No. of classes: %d' % (
        T[0].shape + (len(T),) + (k,))
    )

    # Do cross-validation
    FOLDS = 10
    IDX = list(range(SZ))
    shuffle(IDX)

    fsz = int(SZ / FOLDS)
    offset = 0
    AUC_train = zeros(FOLDS)
    AUC_test = zeros(FOLDS)

    AC_train = zeros(FOLDS)
    AC_test = zeros(FOLDS)
    for f in range(FOLDS):
        idx_test = IDX[offset:offset + fsz]
        idx_train = setdiff1d(IDX, idx_test)
        shuffle(idx_train)
        idx_train = idx_train[:fsz].tolist()
        _log.info('Train Fold %d' % f)
        AUC_train[f],AC_train[f] = innerfold(T, idx_train + idx_test, idx_train, e, k, SZ)
        _log.info('Test Fold %d' % f)
        AUC_test[f],AC_test[f] = innerfold(T, idx_test, idx_test, e, k, SZ)

        offset += fsz

    _log.info('Test Accuracy: %f ' % (AC_test.mean()))
    _log.info('Train Accuracy: %f ' % (AC_train.mean()))

    et = dt.datetime.now()
    print("Time Used:",et-st)