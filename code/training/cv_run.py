'''

__file name__

    cv_run.py

__description__

    This file will perform internal CV diagnostics on my training set.
    FoldTubeID is meant to split the training set (train_cv and tube_cv) such
    that tube_assembly_id's are not found in both the train_cv and tube_cv.
    This is important because if the same tube_assembly_id is in both the
    train_cv and tube_cv, then we will have overfitting problems.


'''

import sys, os, glob
from collections import defaultdict
import pandas as pd
import numpy as np
#from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
#from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb #extreme gradient boosting

sys.path.append(os.path.abspath(".."))
from parameters import *

def xgboost_model(train, test, num_round, params):
    '''
    Takes in: training set, test set, number of estimators, params is a list

    Returns: predictions in a np.log1p format!
    '''
    X = train.as_matrix(train.columns[:-1]).astype(float)
    y = train.as_matrix(['cost'])[:,0].astype(float)
    ylog1p = np.log1p(y)
    X_test = test.as_matrix(test.columns[:-1]).astype(float)

    xgb_train = xgb.DMatrix(X, label = ylog1p)
    xgb_test = xgb.DMatrix(X_test)

    #bst = xgb.train(params, xgb_train, num_round)
    #y_pred = bst.predict(xgb_test)

    # Round 1
    bst1 = xgb.train(params, xgb_train, num_round)
    y_pred1 = bst1.predict(xgb_test)

    # Round 2
    num_round2 = 2000
    bst2 = xgb.train(params, xgb_train, num_round2)
    y_pred2 = bst2.predict(xgb_test)

    #Power Train

    #ypower2 = np.power(y,1/5.0)
    ypower3 = np.power(y,1/20.0)

    #xgb_train2 = xgb.DMatrix(X, label = ypower2)
    xgb_train3 = xgb.DMatrix(X, label = ypower3)


    #xst2 = xgb.train(params, xgb_train2, num_round)
    #y_predp2 = xst2.predict(xgb_test)

    xst3 = xgb.train(params, xgb_train3, num_round)
    y_predp3 = xst3.predict(xgb_test)

    #y_power = (np.power(y_predp2,5.0) + np.power(y_predp3,10.0))/2.0
    y_power = np.power(y_predp3,20.0)

    y_pred = (np.expm1(0.75*y_pred1+0.25*y_pred2) + y_power)/2.0

    return y_pred

class FoldTubeID():
    '''
    Class to split the data on unique tube_assembly_id's
    Does a standard KFold split in unique id's and then I make a boolean
    mask to to subset out the data used for train_cv and test_cv

    Returns: an array of indices for train_cv (1st) and test_cv (2nd)
    for use in subsetting for CV splits
    '''
    def __init__(self, tubeid, n_folds=3):
        self.tubeid = tubeid
        self.n_folds = n_folds

    def __iter__(self):
        uniq_id = self.tubeid.unique()

        # Randomize on unique id
        np.random.seed(0)
        uniq_id = uniq_id[np.random.permutation(len(uniq_id))]

        cv = cross_validation.KFold(len(uniq_id), self.n_folds)
        for trainID, testID in cv:
            train_id = uniq_id[trainID]
            test_id = uniq_id[testID]
            trainID_boolmask = self.tubeid.isin(train_id)
            #testID_boolmask = self.tubeid.isin(test_id)
            testID_boolmask = np.logical_not(trainID_boolmask)
            yield (np.where(trainID_boolmask)[0], np.where(testID_boolmask)[0])



class CVeval():

    def __init__(self, trainset, split_id, n_folds=3):
        # split_id must be a column of ID values to split on
        # For example, train['tube_assembly_id']
        if sum(pd.isnull(trainset['cost'])) > 0 and len(trainset) != 30213:
            raise ValueError('Trainset wrong size (30213, n) or NaN values exist.')

        if len(split_id) != 30213:
            raise ValueError('split_id must be a column of 30213 observations')

        self.split_id = split_id
        self.trainset = trainset
        self.n_folds = n_folds
        self.rmsle_score = []
        self.cv = FoldTubeID(split_id, n_folds)


    def run_cv(self, num_round, params):
        '''
        Using FoldTubeID split, loop over CV to get RMSLE for each split
        params is a list of parameters for XGBoost.

        After finishing CV, run score() to get the results
        '''
        self.pred = []
        self.real = []
        if len(params) == 0:
            raise ValueError('Please read in parameters')

        for tr, te in self.cv:
            self.train = self.trainset.loc[tr,:].copy()
            self.test = self.trainset.loc[te,:].copy()

            # Randomize and set seed
            # np.random.permutation(len(trainp1))
            np.random.seed(1)
            self.train = self.train.iloc[np.random.permutation(len(self.train))]
            np.random.seed(2)
            self.test = self.test.iloc[np.random.permutation(len(self.test))]
            y_real = np.array(self.test.iloc[:,-1])


            # Section for training multi-models if you like
            y_pred = xgboost_model(self.train, self.test, num_round, params)

            self.pred += [y_pred]
            self.real += [y_real]
            self.rmsle_score += [np.sqrt(mean_squared_error(np.log1p(y_real), np.log1p(y_pred)))]
        print '==========================================================='
        print 'Finished Cross-validation'
        print '==========================================================='

    def score(self):
        # Report the mean score of RMSLE and the corresponding std deviation
        self.mean = np.mean(self.rmsle_score)
        self.std = np.std(self.rmsle_score)

        print "Loss: %0.5f (+/- %0.5f)" % (self.mean, self.std)
