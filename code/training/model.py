'''

__file name__

    model.py

__description__

    This file will train the Ensemble model for submission. Output is to the
    specifications of the competition.


'''

import sys, os, glob
from collections import defaultdict
import pandas as pd
import numpy as np
#from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb #extreme gradient boosting

sys.path.append(os.path.abspath(".."))
from parameters import *

def xgboost_model(train, test, num_round, params):
    '''
    Takes in: training set, test set, number of estimators, params is a list

    Returns: predictions in correct format
    '''

    '''
    train_shfl = train.iloc[np.random.permutation(len(train))]
    X = train_shfl.as_matrix(train_shfl.columns[:-1]).astype(float)
    y = train_shfl.as_matrix(['cost'])[:,0].astype(float)
    ylog1p = np.log1p(y).astype(float)
    '''

    X = train.as_matrix(train.columns[:-1]).astype(float)
    y = train.as_matrix(['cost'])[:,0].astype(float)
    ylog1p = np.log1p(y)
    X_test = test.as_matrix(test.columns[:-1]).astype(float)

    xgb_train = xgb.DMatrix(X, label = ylog1p)
    xgb_test = xgb.DMatrix(X_test)

    #bst = xgb.train(params, xgb_train, num_round)
    #y_pred = bst.predict(xgb_test)
    print '#############################################'
    print 'Building XGB Model'
    print '#############################################'
    # Round 1
    bst1 = xgb.train(params, xgb_train, num_round)
    y_pred1 = bst1.predict(xgb_test)

    # Round 2
    #num_round2 = 2000
    #bst2 = xgb.train(params, xgb_train, num_round2)
    #y_pred2 = bst2.predict(xgb_test)

    #Power Train

    ypower3 = np.power(y,1/47.0)
    xgb_train3 = xgb.DMatrix(X, label = ypower3)
    xst3 = xgb.train(params, xgb_train3, num_round)
    y_predp3 = xst3.predict(xgb_test)

    p = 0.5
    y_pred = p*np.expm1(y_pred1) + (1-p)*np.power(y_predp3,47.0)
    #y_pred=p*np.expm1(0.75*y_pred1+0.25*y_pred2)+(1-p)*np.power(y_predp3,20.0)
    #y_pred = 0.35*np.expm1(0.75*y_pred1+0.25*y_pred2) + 0.65*y_power

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
    print 'Building Random Forest Model'
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
    ypower3 = np.power(y,1/40.0)
    rf2.fit(X, ypower3)
    y_pred2 = rf2.predict(X_test)


    y_pred = (np.expm1(y_pred1) + np.power(y_pred2,40.0))/2.0
    #y_pred = np.expm1(y_pred1)
    return y_pred

class Model():

    def __init__(self, train, test, rftrain, rftest, num_round, params_xgb, params_rf):
        self.train = train
        self.test = test
        self.rftrain = rftrain
        self.rftest = rftest
        self.num_round = num_round
        self.params_xgb = params_xgb
        self.params_rf = params_rf

    def build_ensemble(self):
        if (len(params_xgb) == 0) or (len(params_rf) == 0) :
            raise ValueError('Please read in parameters')

        y_pred_xgb = xgboost_model(self.train, self.test, num_round, params_xgb)
        #y_pred_rf = rf_model(self.rftrain, self.rftest, params_rf)

        self.y_pred = y_pred_xgb
        print
        print "================================================================"
        print "================  Finished with Prediction   ==================="
        print "================================================================"

    def convertPred(self):
        self.y_pred[self.y_pred < 0] = 2
        self.y_pred = pd.DataFrame({'id':range(1,self.test.shape[0]+1),
                                        'cost':self.y_pred})
        self.y_pred = self.y_pred[['id','cost']]
        self.y_pred['id'] = self.y_pred['id'].astype(int)

        self.y_pred.to_csv(pred_file,index = False)
        print
        print "================================================================"
        print "===============  Finished with Writing File   =================="
        print "================================================================"
        print 'File:', pred_file


################################################################################
################################################################################
################################################################################
# Model Building
################################################################################
################################################################################
################################################################################
if __name__ == "__main__":

    train = pd.read_csv('../my_data/train' + name_file, header = 0)
    test = pd.read_csv('../my_data/test' + name_file, header = 0)
    rftrain = pd.read_csv('../my_data/train' + rf_file, header = 0)
    rftest = pd.read_csv('../my_data/test' + rf_file, header = 0)
    build = Model(train, test, rftrain, rftest, num_round, params_xgb, params_rf)
    build.build_ensemble()
    pred_file = '../' + pred_file
    print '#############################################'
    print 'Built a model from:'
    print 'xgb training set:', file_train.split('/')[-1]
    print 'xgb testing set:', file_test.split('/')[-1]
    print
    print 'Parameters:'
    print params_xgb, num_round
    #print 'rf training set:', rf_train.split('/')[-1]
    #print 'rf testing set:', rf_test.split('/')[-1]
    print '#############################################'
    print
    build.convertPred()
