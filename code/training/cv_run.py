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

import sys
import os
import glob
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb #extreme gradient boosting
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

sys.path.append(os.path.abspath(".."))
from parameters import *

def xgboost_model(train, test, num_round, params):
    '''
    Takes in: training set, test set, number of estimators, params is a list

    Returns: predictions in correct format
    '''
    X = train.as_matrix(train.columns[:-1]).astype(float)
    y = train.as_matrix(['cost'])[:,0].astype(float)
    ylog1p = np.log1p(y)
    X_test = test.as_matrix(test.columns[:-1]).astype(float)

    xgb_train = xgb.DMatrix(X, label = ylog1p)
    xgb_test = xgb.DMatrix(X_test)


    # Round 1
    bst1 = xgb.train(params, xgb_train, num_round)
    y_pred1 = bst1.predict(xgb_test)

    # Round 2
    #num_round2 = 2000
    #bst2 = xgb.train(params, xgb_train, 2000)
    #y_pred2 = bst2.predict(xgb_test)

    #Power Train
    ypower3 = np.power(y,1/47.0)
    xgb_train3 = xgb.DMatrix(X, label = ypower3)
    xst3 = xgb.train(params, xgb_train3, num_round)
    y_predp3 = xst3.predict(xgb_test)

    p = 0.5
    y_pred = p*np.expm1(y_pred1) + (1-p)*np.power(y_predp3,47.0)

    return y_pred

def rf_model(train, test, params):
    '''
    Takes in: training set, test set, params is a list

    Returns: predictions in correct format
    '''
    X = train.as_matrix(train.columns[:-1]).astype(float)
    y = train.as_matrix(['cost'])[:,0].astype(float)
    X_test = test.as_matrix(test.columns[:-1]).astype(float)
    print '#############################################'
    print 'Building Random Forest Model from:'
    print 'rf training set:', rf_train.split('/')[-1]
    print 'rf testing set:', rf_test.split('/')[-1]
    print
    print 'Parameters:'
    print params
    print '#############################################'
    print '...'
    print

    rf = RandomForestRegressor(**params)
    ylog1p = np.log1p(y)
    rf.fit(X, ylog1p)
    y_pred1 = rf.predict(X_test)

    rf2 = RandomForestRegressor(**params)
    ypower3 = np.power(y,1/45.0)
    rf2.fit(X, ypower3)
    y_pred2 = rf2.predict(X_test)

    y_pred = (np.expm1(y_pred1) + np.power(y_pred2,45.0))/2.0
    return y_pred

def keras_model(train, test, params):
    '''
    Takes in: training set, test set, number of estimators,
    params is a dictionary

    Returns: predictions in a np.log1p format!
    '''
    y = train['cost'].values
    train = train.drop(['cost'], axis = 1)
    test = test.drop(['cost'], axis = 1)
    train = np.array(train)
    test = np.array(test)

    X = train.astype(float)
    X_test = test.astype(float)
    ylog1p = np.log1p(y).astype(float)

    model = Sequential()
    model.add(Dense(train.shape[1], 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, 1))

    model.compile(loss='mse', optimizer='rmsprop')

    # train model
    model.fit(X, ylog1p, batch_size= params["batch_size"],
              nb_epoch=params["nb_epoch"], verbose=params["verbose"],
              validation_split=params["validation_split"])


    preds = np.expm1(model.predict(X_test, verbose=0).flatten())

    return preds

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
            testID_boolmask = self.tubeid.isin(test_id)
            # Generator to give row positions for each fold
            yield (np.where(trainID_boolmask)[0], np.where(testID_boolmask)[0])



class CVeval():

    def __init__(self, trainset, rftrainset, split_id, n_folds=3):
        # split_id must be a column of ID values to split on
        # For example, train['tube_assembly_id']
        if sum(pd.isnull(trainset['cost'])) > 0 and len(trainset) != 30213:
            raise ValueError('Trainset wrong size (30213, n)'+\
                             ' or NaN values exist.')

        if len(split_id) != 30213:
            raise ValueError('split_id must be a column of 30213 observations')

        self.split_id = split_id
        self.trainset = trainset
        self.rftrainset = rftrainset
        self.n_folds = n_folds
        self.rmsle_score = []
        self.cv = FoldTubeID(split_id, n_folds)

    def ens_cv(self, num_round, params, paramsrf):
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
            self.rftrain = self.rftrainset.loc[tr,:].copy()
            self.rftest = self.rftrainset.loc[te,:].copy()
            # Randomize and set seed
            # np.random.permutation(len(trainp1))
            np.random.seed(1)
            self.train = self.train.iloc[np.random.permutation(len(self.train))]
            np.random.seed(2)
            self.test = self.test.iloc[np.random.permutation(len(self.test))]
            y_real = np.array(self.test.iloc[:,-1])

            np.random.seed(3)
            mask = np.random.permutation(len(self.rftrain))
            self.rftrain = self.rftrain.iloc[mask]
            np.random.seed(4)
            mask2 = np.random.permutation(len(self.rftest))
            self.rftest = self.rftest.iloc[mask2]
            y_real = np.array(self.rftest.iloc[:,-1])

            # Section for training multi-models if you like
            y_pred_xgb = xgboost_model(self.train, self.test, num_round, params)
            y_pred_rf = rf_model(self.rftrain, self.rftest, paramsrf)

            p = 0.8
            y_pred = p*y_pred_xgb + (1-p)*y_pred_rf
            self.pred += [y_pred]
            self.real += [y_real]
            self.rmsle_score += [np.sqrt(mean_squared_error(np.log1p(y_real),
                                 np.log1p(y_pred)))]
        print '==========================================================='
        print 'Finished Cross-validation'
        print '==========================================================='

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
            y_pred_xgb = xgboost_model(self.train, self.test, num_round, params)

            y_pred = y_pred_xgb
            self.pred += [y_pred]
            self.real += [y_real]
            self.rmsle_score += [np.sqrt(mean_squared_error(np.log1p(y_real),
                                 np.log1p(y_pred)))]
        print '==========================================================='
        print 'Finished Cross-validation'
        print '==========================================================='


    def rf_cv(self, params):
        '''
        Using FoldTubeID split, loop over CV to get RMSLE for each split
        params is a list of parameters for RandomForestRegressor.

        After finishing CV, run score() to get the results
        '''
        self.pred = []
        self.real = []
        if len(params) == 0:
            raise ValueError('Please read in parameters')
        cvround = 1
        for tr, te in self.cv:
            print 'CV round:', cvround
            self.rftrain = self.rftrainset.loc[tr,:].copy()
            self.rftest = self.rftrainset.loc[te,:].copy()

            # Randomize and set seed
            # np.random.permutation(len(trainp1))
            np.random.seed(1)
            mask = np.random.permutation(len(self.rftrain))
            self.rftrain = self.rftrain.iloc[mask]
            np.random.seed(2)
            mask2 = np.random.permutation(len(self.rftest))
            self.rftest = self.rftest.iloc[mask2]
            y_real = np.array(self.rftest.iloc[:,-1])


            # Section for training multi-models if you like
            y_pred = rf_model(self.rftrain, self.rftest, params)

            self.pred += [y_pred]
            self.real += [y_real]
            self.rmsle_score += [np.sqrt(mean_squared_error(np.log1p(y_real),
                                 np.log1p(y_pred)))]
            cvround += 1
        print '==========================================================='
        print 'Finished Random Forest Cross-validation'
        print '==========================================================='


    def keras_cv(self, params):
        '''
        Using FoldTubeID split, loop over CV to get RMSLE for each split
        params is a list of parameters for Keras Neural Networks.

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
            y_pred = keras_model(self.train, self.test, params)

            self.pred += [y_pred]
            self.real += [y_real]
            self.rmsle_score += [np.sqrt(mean_squared_error(np.log1p(y_real),
                                 np.log1p(y_pred)))]
        print '==========================================================='
        print 'Finished Keras Cross-validation'
        print '==========================================================='

    def score(self):
        # Report the mean score of RMSLE and the corresponding std deviation
        self.mean = np.mean(self.rmsle_score)
        self.std = np.std(self.rmsle_score)

        print "Loss: %0.5f (+/- %0.5f)" % (self.mean, self.std)


################################################################################
################################################################################
################################################################################
# Cross-Validation
################################################################################
################################################################################
################################################################################

if __name__ == "__main__":

    train = pd.read_csv('../my_data/train' + name_file, header = 0)
    rftrain = pd.read_csv('../my_data/train' + rf_file, header = 0)
    tube_id = pd.read_csv('../my_data/tube_assembly_cv.csv',header = 0)
    t = CVeval(train, rftrain, tube_id['tube_assembly_id'], n_folds = 5)
    t.run_cv(num_round, params_xgb)
    #t.ens_cv(num_round, params_xgb, params_rf)
    #t.rf_cv(params_rf)
    #t.keras_cv(params_keras)
    print '#############################################'
    print 'Built a model from:'
    if x_gb:
        print 'training set:', file_train.split('/')[-1]
        print 'XGB Parameters:'
        print params_xgb
    if rf:
        print 'training set:', rf_train.split('/')[-1]
        print 'RF Parameters:'
        print params_rf
#    print 'testing set:', file_test.split('/')[-1]
    print '#############################################'
    print
    print "Results:"
    print t.score()
    print "Fold Scores:"
    print t.rmsle_score
