'''
__file__
    encoding.py

__description__
    This file is meant to take the CSV files constructed by fulldatamerge.py
    and encode the categorical features into numbers for modeling

    writes to train.csv and test.csv for modeling


'''
import sys
import os
import glob
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(".."))
from parameters import *
sys.path.append(os.path.abspath(".."))
from dataclean import *

traintest = pd.read_csv(os.path.join('..','my_data','traintest150818.csv'),
                        header=0)
tube_id = pd.read_csv(os.path.join('..','my_data','tube_assembly_id.csv'),
                      header = 0)
#tube_id.columns = ['tube_assembly_id']


'''
Encode file and save train and test
'''
#traintest2 = traintest[bestcol].copy() #bestcol from parameters.py
traintest2 = traintest.copy()

### specs - encode from dataclean
# SPECS already encoded
# add total spec
"""
# specs encoded in traintest150817
if 'spec1' in traintest2.columns:
    # Convert all NaN values to 0 before you do anything
    specs = traintest2.columns[traintest2.columns.str.contains('spec')]
    specs = specs[:-1]
    for s in specs:
        traintest2.loc[pd.notnull(traintest2[s]),s] = 1
        traintest2.loc[pd.isnull(traintest2[s]),s] = 0
"""

################################################################################
### Supplier and material_id
################################################################################
lecolumns = ['supplier','material_id']
traintest2 = encode(traintest2,lecolumns,TRANSFORM_CUTOFF)

################################################################################
### bracket_pricing
################################################################################
vl = traintest2.bracket_pricing.values
traintest2['bracket_pricing'] = [1 if x=='Yes' else 0 for x in vl]

################################################################################
### end_a_2x, end_x_1x, end_x_2x, end_a, end_x: PruneLabelEncoder from dataclean
################################################################################
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

################################################################################
### all component_id_* in bestcol
################################################################################
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

################################################################################
### all component_type_id_comp* in bestcol
################################################################################
"""
# use a labelEncoder for this
m = traintest2.columns.str.contains('component_type_id')
comp_type = traintest2.columns[m]
comptype_vals = np.array([])
for ct in comp_type:
    traintest2.loc[pd.isnull(traintest2[ct]),ct] = 0
    comptype_vals = np.concatenate((comptype_vals, np.array(traintest2[ct])))
ple_comptypeID = PruneLabelEncoder()
ple_comptypeID.fit(comptype_vals, cutoff=TRANSFORM_CUTOFF)
for ct in comp_type:
    traintest2[ct] = ple_comptypeID.transform(traintest2[ct].values)
"""
# Use Frequency encoding
mask = traintest2.columns.str.contains('component_type_id')
comp_type = traintest2.columns[mask]
for y in range(len(comp_type)):
    for x in traintest2[comp_type[y]].value_counts().index:
        traintest2.loc[traintest2[comp_type[y]] == x, comp_type[y]] =
                       traintest2[comp_type[y]].value_counts()[x]
    traintest2.loc[pd.isnull(traintest2[comp_type[y]]),comp_type[y]] = 0


################################################################################
# Orientation standard encoding
################################################################################
"""
orient = list(traintest2.columns[traintest2.columns.str.contains('orient')])
traintest2 = encode(traintest2,orient,TRANSFORM_CUTOFF)
sum_orient = np.zeros(len(traintest2))
for x in orient:
    sum_orient += np.array(traintest2[x])
traintest2['orient_sum'] = sum_orient
"""
################################################################################
# Encoding overall_length
################################################################################
"""
ovlength_total = np.zeros(len(traintest2))
ovlength = traintest2.columns[traintest2.columns.str.contains('overall_length')]
for x in ovlength:
    traintest2.loc[pd.isnull(traintest2[x]),x] = 0
    ovlength_total += np.array(traintest2[x])
traintest2['total_ovlength'] = ovlength_total
"""
################################################################################
# Connection TYPE
################################################################################
mask2 = traintest2.columns.str.contains('connection_type_id')
conn_type = traintest2.columns[mask2]
"""
for x in conn_type:
    traintest2.loc[pd.isnull(traintest2[x]),x] = 0
traintest2 = encode_force(traintest2,conn_type,TRANSFORM_CUTOFF)
# frequency encode
for y in conn_type:
    for x in traintest2[y].value_counts().index:
        traintest2.loc[traintest2[y] == x, y] = traintest2[y].value_counts()[x]
"""


conn_total = np.zeros(len(traintest2))
for x in conn_type:
    traintest2.loc[pd.notnull(traintest2[x]),x] = 1
    traintest2.loc[pd.isnull(traintest2[x]),x] = 0
    conn_total += np.array(traintest2[x])
traintest2['total_conn'] = conn_total
"""
# frequency encode
for x in traintest2['total_conn'].value_counts().index:
    traintest2.loc[traintest2['total_conn'] == x, 'total_conn'] =
                   traintest2['total_conn'].value_counts()[x]
"""


################################################################################
# Scaling my continuous variables (ie. length)
################################################################################
#fields = ["annual_usage", "min_order_quantity", "quantity",
#          "diameter", "wall", "length", "num_bends", "bend_radius"]
#fields = ["diameter", "wall", "length", "bend_radius"]
"""
fields = ["length", "diameter"]
# length
for col in fields:
    traintest2[col] = np.log1p(traintest2[col])
"""


################################################################################
# Create train and test
################################################################################


traintest2 = traintest2[bestcol] #bestcol from parameters.py
train = traintest2.iloc[0:30213]
test = traintest2.iloc[30213:]
tube_id2 = tube_id.iloc[0:30213]



################################################################################
# One Hot encoding
################################################################################

train_cost = train['cost'].copy()
test_cost = test['cost'].copy()
train = train.drop(['cost'], axis = 1)
test = test.drop(['cost'], axis = 1)


onehotcol = ['supplier', 'material_id', 'component_id_1', 'component_id_2',
            'component_id_3','end_a','end_x']

for col in onehotcol:
    train_temp = train[col].values
    test_temp = test[col].values
    combine = np.concatenate((train_temp,test_temp))
    lbl = OneHotEncoder()
    lbl.fit(np.resize(np.array(combine).astype(float), (len(combine),1)))

    train_temp = lbl.transform(np.resize(np.array(train_temp).astype(float),
                               (len(train_temp),1))).toarray()
    test_temp = lbl.transform(np.resize(np.array(test_temp).astype(float),
                              (len(test_temp),1))).toarray()

    for i in range(train_temp.shape[1]):
        train[col + "_" + str(i)] = train_temp[:,i]
    for i in range(test_temp.shape[1]):
        test[col + "_" + str(i)] = test_temp[:, i]
    train = train.drop([col], axis = 1)
    test = test.drop([col],axis = 1)





################################################################################
# feature scaling
################################################################################


for col in train.columns:
    train[col] = train[col].astype(float)
    test[col] = test[col].astype(float)
    sc = MinMaxScaler()
    sc.fit(train[col])
    train[col] = sc.transform(train[col])
    test[col] = sc.transform(test[col])
    """
    if np.max(train[col]) == 0 or np.max(test[col]) == 0:
        continue
    else:
        train[col] = np.array(train[col])/np.max(train[col]).astype(float)
        test[col] = np.array(test[col])/np.max(train[col]).astype(float)
    """
# Add back on cost
train['cost'] = train_cost
test['cost'] = test_cost


'''
Save files to my_files
'''
print
print '========================================================================'
print '========================   Saving Files   =============================='
print '========================================================================'
print


print 'columns used:'
print bestcol
print
print "One Hot Encoding:", one_hot
print "Used One Hot Encoding on:", onehotcol
print

# keras_train from parameters.py
train.to_csv(keras_train, index = False)
print 'File created:', keras_train
print 'DataFrame shape:', train.shape
print

# keras_test from parameters.py
test.to_csv(keras_test, index = False)
print 'File created:', keras_test
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
