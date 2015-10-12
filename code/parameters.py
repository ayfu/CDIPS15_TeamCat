'''
__file__

    parameters.py

__description__

    This file is used by other py files for building the dataset and models.
    For example, bestcol is used to explore the columns

'''
import os
import glob
from collections import defaultdict

import pandas as pd
import numpy as np

# file names for data for xgb, random forest, and keras.

name_file = 'TEST3.csv'
rf_file = 'TEST3.csv'
keras_file = 'KERAS02.csv'
pred_file = '../notebooks/predictions/cost_predicted_model28o.csv'

# encoding cutoff
TRANSFORM_CUTOFF = 1
one_hot = True
x_gb = True
rf = False
keras = False

# MAKE SURE 'COST' is ALWAYS LAST!
# Base bestcol


bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',           #'end_a_1x',
                'end_a_2x',            'end_x_1x',            'end_x_2x',
               'forming_a',           'forming_x',       #'end_a','end_x',
               'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
             #'total_specs', #'volume',
             #'uniq_specs',
             'mean_specs',
             #'mean_weight',
             #'mean_ann_usage',
             #'mean_quantity',
             #'count_specs',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
                    'cost'])

file_train = '../my_data/train' + name_file
file_test = '../my_data/test' + name_file
rf_train = '../my_data/train' + rf_file
rf_test = '../my_data/test' + rf_file
keras_train = '../my_data/train' + keras_file
keras_test = '../my_data/test' + keras_file


################################################################################
# Parameters for models

'''
Parameters for XGBoost modeling
'''

params_xgb = {}
params_xgb["objective"] = "reg:linear"
params_xgb["eta"] = 0.02
#params_xgb["gamma"] = 2
params_xgb["min_child_weight"] = 6
params_xgb["subsample"] = 0.7
params_xgb["colsample_bytree"] = 0.7
params_xgb["scale_pos_weight"] = 0.8
params_xgb["silent"] = 0 # 0 means printing running messages, 1 means silent mode. print information of performance
params_xgb["max_depth"] = 9
params_xgb["max_delta_step"]=2

params_xgb = list(params_xgb.items())
num_round = 6000

'''
Parameters for RandomForestRegressor
'''
params_rf = {'n_estimators': 200,
          'criterion': "mse",
          'max_features': "auto",
          'max_depth': None,
          'min_samples_split': 2,
          'min_samples_leaf': 1,
          'min_weight_fraction_leaf': 0,
          #'max_leaf_notes': None
          'verbose': 1
          }


"""
Parameters for Keras Neural Networks
"""
params_keras = {}
params_keras["batch_size"] = 100
params_keras["nb_epoch"] = 200
params_keras["verbose"] = 2
params_keras["validation_split"] = 0
