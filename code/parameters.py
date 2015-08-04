
import os, glob
from collections import defaultdict
import pandas as pd
import numpy as np

bestcol = list(['supplier',                'year',               'month',
                      'day',        'annual_usage',  'min_order_quantity',
           'bracket_pricing',            'quantity',         'material_id',
                  'diameter',                'wall',              'length',
                 'num_bends',         'bend_radius',            'end_a_2x',
                  'end_x_1x',            'end_x_2x',               'end_a',
                     'end_x',            'num_boss',         'num_bracket',
                     'other',      'component_id_1',          'quantity_1',
            'component_id_2',          'quantity_2',      'component_id_3',
                'quantity_3',                'cost'])

TRANSFORM_CUTOFF = 25



'''
Parameters for XGBoost modeling
'''

params_xgb = {}
params_xgb["objective"] = "reg:linear"
params_xgb["eta"] = 0.02
params_xgb["min_child_weight"] = 6
params_xgb["subsample"] = 0.7
params_xgb["colsample_bytree"] = 0.7
params_xgb["scale_pos_weight"] = 0.8
params_xgb["silent"] = 0 # 0 means printing running messages, 1 means silent mode. print information of performance
params_xgb["max_depth"] = 10
params_xgb["max_delta_step"]=2

params_xgb = list(params_xgb.items())

num_round = 4000
