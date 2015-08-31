'''

__file name__

    train_model.py

__description__

    This file will train the XGB model for submission. Output is to the
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


class xgbModel():

    def __init__(self, train, test, num_round, params):
        self.train = train
        self.test = test
        self.num_round = num_round
        self.params = params

    def buildXGB(self):
        '''
        train_shfl = train.iloc[np.random.permutation(len(train))]
        X = train_shfl.as_matrix(train_shfl.columns[:-1]).astype(float)
        y = train_shfl.as_matrix(['cost'])[:,0].astype(float)
        ylog1p = np.log1p(y).astype(float)
        '''

        X = self.train.as_matrix(self.train.columns[:-1]).astype(float)
        y = self.train.as_matrix(['cost'])[:,0].astype(float)
        ylog1p = np.log1p(y).astype(float)
        X_test = self.test.as_matrix(self.test.columns[:-1]).astype(float) # cost is still last column

        xgb_train = xgb.DMatrix(X, label = ylog1p)
        xgb_test = xgb.DMatrix(X_test)

        #Train multiple times

        # Round 1
        num_round1 = 4000
        self.bst1 = xgb.train(self.params, xgb_train, num_round1)
        y_pred1 = self.bst1.predict(xgb_test)

        # Round 2
        num_round2 = 2000
        self.bst2 = xgb.train(self.params, xgb_train, num_round2)
        y_pred2 = self.bst2.predict(xgb_test)

        #Power Train
        #ypower2 = np.power(y,1/5.0)
        ypower3 = np.power(y,1/20.0)

        #xgb_train2 = xgb.DMatrix(X, label = ypower2)
        xgb_train3 = xgb.DMatrix(X, label = ypower3)


        #self.xst2 = xgb.train(self.params, xgb_train2, self.num_round)
        #y_predp2 = self.xst2.predict(xgb_test)

        self.xst3 = xgb.train(self.params, xgb_train3, self.num_round)
        y_predp3 = self.xst3.predict(xgb_test)

        #y_power = (np.power(y_predp2,5.0) + np.power(y_predp3,10.0))/2.0
        y_power = np.power(y_predp3,20.0)

        self.y_pred = (np.expm1(0.75*y_pred1+0.25*y_pred2) + y_power)/2.0
        #self.y_pred = 0.35*np.expm1(0.75*y_pred1+0.25*y_pred2) + 0.65*y_power
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
    build = xgbModel(train, test, num_round, params_xgb)
    build.buildXGB()
    pred_file = '../' + pred_file
    print '#############################################'
    print 'Built a model from:'
    print 'training set:', file_train.split('/')[-1]
    print 'testing set:', file_test.split('/')[-1]
    print '#############################################'
    print
    build.convertPred()
