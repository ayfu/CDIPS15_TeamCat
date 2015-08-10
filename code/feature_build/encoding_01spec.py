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
traintest2 = traintest[bestcol].copy()
if 'spec1' in traintest2.columns:
    # Convert all NaN values to 0 before you do anything
    specs = traintest2.columns[traintest2.columns.str.contains('spec')]
    for s in specs:
        traintest2.loc[pd.notnull(traintest2[s]),s] = 1
        traintest2.loc[pd.isnull(traintest2[s]),s] = 0
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

### all quantity_*
quantids = traintest2.columns[traintest2.columns.str.contains('quantity_')]
for qid in quantids:
    traintest2[qid] = [0 if np.isnan(x) else x for x in traintest2[qid].values]

### all component_id_* in bestcol
# PruneLabelEncoder from dataclean
compids = traintest2.columns[traintest2.columns.str.contains('component_id_')]
comp_vals = np.array([])
for cid in compids:
    traintest2.loc[pd.isnull(traintest2[cid]),cid] = 0
    comp_vals = np.concatenate((comp_vals, np.array(traintest2[cid])))
ple_comptype = PruneLabelEncoder()
ple_comptype.fit(comp_vals, cutoff=TRANSFORM_CUTOFF)
for cid in compids:
    traintest2[cid] = ple_comptype.transform(traintest2[cid].values)


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
