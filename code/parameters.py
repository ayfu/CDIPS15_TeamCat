'''
__file__
    parameters.py

__description__
    This file is used by other py files for building the dataset and models.
    For example, bestcol is used to explore the columns
'''
import os, glob
from collections import defaultdict
import pandas as pd
import numpy as np


# MAKE SURE 'COST' is ALWAYS LAST!
# Base bestcol



bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
#                   'spec1',               'spec2',               'spec3',
#                   'spec4',               'spec5',               'spec6',
#                   'spec7',               'spec8',               'spec9',
#                  'spec10',
             'total_specs',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
                    'cost'])
file_train = '../my_data/trainTOTSPEC1.csv'
file_test = '../my_data/testTOTSPEC1.csv'



TRANSFORM_CUTOFF = 1



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

#pred_file = 'C:/Users/Anthony/Dropbox/Background Understanding/Data Science/CDIPS/CDIPS15_TeamCat/notebooks/predictions/cost_predicted_model15d.csv'
pred_file = '../notebooks/predictions/cost_predicted_model21e.csv'



"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',      'component_id_2',      'component_id_3',
          'component_id_4',      'component_id_5',      'component_id_6',
                   'spec1',               'spec2',               'spec3',
                   'spec4',               'spec5',               'spec6',
                  'cost'])
file_train = '../my_data/train150810noQuant_id.csv'
file_test = '../my_data/test150810noQuant_id.csv'
"""

#Component_type_id inclusion
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
                   'spec1',               'spec2',               'spec3',
                   'spec4',               'spec5',               'spec6',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
 'component_type_id_comp4','component_type_id_comp5','component_type_id_comp6',
                   'cost'])
file_train = '../my_data/trainCOMPTYPEfreq_02.csv'
file_test = '../my_data/testCOMPTYPEfreq_02.csv'
"""























# SPECS
''''
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
                   'spec1',               'spec2',               'spec3',
                   'spec4',               'spec5',               'spec6',
                  'cost'])
file_train = '../my_data/train150816.csv'
file_test = '../my_data/test150816.csv'
'''
# overall_lengths
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
                   'spec1',               'spec2',               'spec3',
                   'spec4',               'spec5',               'spec6',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
    'overall_length_comp1','overall_length_comp2','overall_length_comp3',
#    'overall_length_comp4','overall_length_comp5','overall_length_comp6',
                    'cost'])
file_train = '../my_data/trainOV13.csv'
file_test = '../my_data/testOV13.csv'
"""
#SLIM
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',      'component_id_2',      'component_id_3',
          'component_id_4',      'component_id_5',      'component_id_6',
             'total_specs',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
                    'cost'])
file_train = '../my_data/trainSLIM.csv'
file_test = '../my_data/testSLIM.csv'
"""


# Orientation
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
                   'spec1',               'spec2',               'spec3',
                   'spec4',               'spec5',               'spec6',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
       'orientation_comp1',   'orientation_comp2',   'orientation_comp3',
              'orient_sum',                 'cost'])
file_train = '../my_data/trainORIENT.csv'
file_test = '../my_data/testORIENT.csv'
"""


#WEIGHT AND OVERALL LENGTH
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
                   'spec1',               'spec2',               'spec3',
                   'spec4',               'spec5',               'spec6',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
             'weight_id_1',         'weight_id_2',         'weight_id_3',
             'weight_id_4',         'weight_id_5',         'weight_id_6',
    'overall_length_comp1','overall_length_comp2','overall_length_comp3',
    'overall_length_comp4','overall_length_comp5','overall_length_comp6',
                    'cost'])
file_train = '../my_data/trainWEIGHTOV.csv'
file_test = '../my_data/testWEIGHTOV.csv'
"""


# SPECS + COMPONENTS
'''
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',      'component_id_1',          'quantity_1',
          'component_id_2',          'quantity_2',      'component_id_3',
              'quantity_3',      'component_id_4',          'quantity_4',
          'component_id_5',          'quantity_5',      'component_id_6',
              'quantity_6',               'spec1',
                   'spec2',               'spec3',               'spec4',
                   'spec5',               'spec6',                'cost'])
file_train = '../my_data/trainSScomp16.csv'
file_test = '../my_data/testSScomp16.csv'
'''
# SPECS + ALL COMPONENTS
'''
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',      'component_id_1',          'quantity_1',
          'component_id_2',          'quantity_2',      'component_id_3',
              'quantity_3',      'component_id_4',          'quantity_4',
          'component_id_5',          'quantity_5',      'component_id_6',
              'quantity_6',      'component_id_7',          'quantity_7',
          'component_id_8',          'quantity_8',               'spec1',
                   'spec2',               'spec3',               'spec4',
                   'spec5',               'spec6',                'cost'])
file_train = '../my_data/trainSScomp.csv'
file_test = '../my_data/testSScomp.csv'
'''





# All specs
'''
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',      'component_id_1',          'quantity_1',
          'component_id_2',          'quantity_2',      'component_id_3',
              'quantity_3',               'spec1',               'spec2',
                   'spec3',               'spec4',               'spec5',
                   'spec6',               'spec7',               'spec8',
                   'spec9',              'spec10',                'cost'])
file_train = '../my_data/trainALLSPEC01.csv'
file_test = '../my_data/testALLSPEC01.csv'
'''

# base model!
'''
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',      'component_id_1',          'quantity_1',
          'component_id_2',          'quantity_2',      'component_id_3',
              'quantity_3',               'cost'])
file_train = '../my_data/trainCUT1.csv'
file_test = '../my_data/testCUT1.csv'
'''




# no Num_bracket - tried on 150806
'''
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',
                   'other',      'component_id_1',          'quantity_1',
          'component_id_2',          'quantity_2',      'component_id_3',
              'quantity_3',               'cost'])

file_train = '../my_data/trainCUT1noNumBrk.csv'
file_test = '../my_data/testCUT1noNumBrk.csv'
'''


#CONNECTION TYPE (EVERYTHING!)
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
#                   'spec1',               'spec2',               'spec3',
#                   'spec4',               'spec5',               'spec6',
#                   'spec7',               'spec8',               'spec9',
#                  'spec10',
             'total_specs',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
 'connection_type_id_comp1',
 'connection_type_id_1_comp1',
 'connection_type_id_2_comp1',
 'connection_type_id_3_comp1',
 'connection_type_id_comp2',
 'connection_type_id_1_comp2',
 'connection_type_id_2_comp2',
 'connection_type_id_3_comp2',
 'connection_type_id_comp3',
 'connection_type_id_1_comp3',
 'connection_type_id_2_comp3',
 'connection_type_id_3_comp3',
 'connection_type_id_4_comp3',
 'connection_type_id_comp4',
 'connection_type_id_1_comp4',
 'connection_type_id_2_comp4',
 'connection_type_id_3_comp4',
 'connection_type_id_1_comp5',
 'connection_type_id_2_comp5',
 'connection_type_id_comp5',
 'connection_type_id_1_comp6',
 'connection_type_id_comp6',
                    'cost'])
file_train = '../my_data/trainCONNDUMB.csv'
file_test = '../my_data/testCONNDUMB.csv'
"""
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
#                   'spec1',               'spec2',               'spec3',
#                   'spec4',               'spec5',               'spec6',
#                   'spec7',               'spec8',               'spec9',
#                  'spec10',
             'total_specs',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
 'connection_type_id_1_comp1',
 'connection_type_id_2_comp1',
 'connection_type_id_comp2',
 'connection_type_id_1_comp3',
                    'cost'])
file_train = '../my_data/trainCONNSOMEFREQ.csv'
file_test = '../my_data/testCONNSOMEFREQ.csv'
"""
"""
bestcol = list(['supplier',                'year',               'month',
                     'day',        'annual_usage',  'min_order_quantity',
         'bracket_pricing',            'quantity',         'material_id',
                'diameter',                'wall',              'length',
               'num_bends',         'bend_radius',            'end_a_2x',
                'end_x_1x',            'end_x_2x',               'end_a',
                   'end_x',            'num_boss',         'num_bracket',
                   'other',        'total_weight',      'total_quantity',
          'component_id_1',          'quantity_1',      'component_id_2',
              'quantity_2',      'component_id_3',          'quantity_3',
          'component_id_4',          'quantity_4',      'component_id_5',
              'quantity_5',      'component_id_6',          'quantity_6',
#                   'spec1',               'spec2',               'spec3',
#                   'spec4',               'spec5',               'spec6',
#                   'spec7',               'spec8',               'spec9',
#                  'spec10',
             'total_specs',          'total_conn',
 'component_type_id_comp1','component_type_id_comp2','component_type_id_comp3',
                    'cost'])
file_train = '../my_data/trainCONNTOTFREQ.csv'
file_test = '../my_data/testCONNTOTFREQ.csv'
"""
