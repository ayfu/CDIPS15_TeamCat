'''
__file__
    encoding.py

__description__
    This file is meant to take the CSV files constructed by fulldatamerge.py
    and encode the categorical features into numbers for modeling

    writes to train.csv and test.csv for modeling


'''
import sys
import os, glob
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(".."))
from parameters import *
sys.path.append(os.path.abspath(".."))
from dataclean import *

traintest = pd.read_csv(os.path.join('..','my_data','traintestNOCOMP.csv'), header=0)
tube_id = pd.read_csv(os.path.join('..','my_data','tube_assembly_id.csv'), header = 0)
#tube_id.columns = ['tube_assembly_id']


'''
Encode file and save train and test
'''

traintest2 = traintest[bestcol].copy() #bestcol from parameters.py

### supplier and material_id and specs - encode from dataclean
if 'spec1' in traintest2.columns:
    # Convert all NaN values to 0 before you do anything
    specs = traintest2.columns[traintest2.columns.str.contains('spec')]
    traintest2.loc[pd.isnull(traintest2['material_id']),'material_id'] = 0 # NaN -> 0
    sp = np.array(traintest2['material_id'].values)
    for s in specs:
        traintest2.loc[pd.isnull(traintest2[s]),s] = 0
        sp = np.concatenate((sp,traintest2[s].values))
    ple_spec = PruneLabelEncoder()
    ple_spec.fit(sp, cutoff=TRANSFORM_CUTOFF)
    traintest2['material_id'] = ple_spec.transform(traintest2.material_id.values)
    for s in specs:
        traintest2[s] = ple_spec.transform(traintest2[s].values)
    traintest2 = encode(traintest2,['supplier'],TRANSFORM_CUTOFF)
else:
    ### supplier and material_id - encode from dataclean
    lecolumns = ['supplier','material_id']
    traintest2 = encode(traintest2,lecolumns,TRANSFORM_CUTOFF)

### bracket_pricing
traintest2['bracket_pricing'] = [1 if x=='Yes' else 0 for x in traintest2.bracket_pricing.values]

### end_a_2x, end_x_1x, end_x_2x, end_a, end_x: PruneLabelEncoder from dataclean
traintest2['end_x_1x'] = [1 if x=='Y' else 0 for x in traintest2['end_x_1x']]
traintest2['end_x_2x'] = [1 if x=='Y' else 0 for x in traintest2['end_x_2x']]
traintest2['end_a_2x'] = [1 if x=='Y' else 0 for x in traintest2['end_a_2x']]
enda_vals = traintest2.end_a.values
endx_vals = traintest2.end_x.values
end_vals = np.concatenate((enda_vals, endx_vals))
ple_end = PruneLabelEncoder()
ple_end.fit(end_vals, cutoff=TRANSFORM_CUTOFF)
traintest2['end_a'] = ple_end.transform(traintest2.end_a.values)
traintest2['end_x'] = ple_end.transform(traintest2.end_x.values)

### quantity_1, quantity_2, quantity_3
traintest2['quantity_1'] = [0 if np.isnan(x) else x for x in traintest2.quantity_1.values]
traintest2['quantity_2'] = [0 if np.isnan(x) else x for x in traintest2.quantity_2.values]
traintest2['quantity_3'] = [0 if np.isnan(x) else x for x in traintest2.quantity_3.values]

### component_id_1, component_id_2, component_id_3
# PruneLabelEncoder from dataclean
comp1 = traintest2.component_id_1.values
comp2 = traintest2.component_id_2.values
comp3 = traintest2.component_id_3.values
comp_vals = np.concatenate((comp1,comp2,comp3))
ple_comptype = PruneLabelEncoder()
ple_comptype.fit(comp_vals, cutoff=TRANSFORM_CUTOFF)
traintest2['component_id_1'] = ple_comptype.transform(traintest2.component_id_1.values)
traintest2['component_id_2'] = ple_comptype.transform(traintest2.component_id_2.values)
traintest2['component_id_3'] = ple_comptype.transform(traintest2.component_id_3.values)


'''
Save files to my_files
'''
print
print '========================================================================'
print '========================   Saving Files   =============================='
print '========================================================================'
print

train = traintest2.iloc[0:30213]
test = traintest2.iloc[30213:]
tube_id2 = tube_id.iloc[0:30213]

# file_train from parameters.py
train.to_csv(file_train, index = False)
print 'File created:', file_train
print 'DataFrame shape:', train.shape
print

# file_test from parameters.py
test.to_csv(file_test, index = False)
print 'File created:', file_test
print 'DataFrame shape:', test.shape
print

'''
file_tubes = '../my_data/tube_assembly_id.csv'
tube_id.to_csv(file_tubes, index = False)
print 'File created:', file_tubes
print 'DataFrame shape:', tube_id.shape
print
'''

file_tubes2 = '../my_data/tube_assembly_cv.csv'
tube_id2.to_csv(file_tubes2, index = False)
print 'File created:', file_tubes2
print 'DataFrame shape:', tube_id2.shape
print
