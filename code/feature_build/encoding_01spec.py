'''
__file__

    encoding_01spec.py

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

traintest = pd.read_csv(os.path.join('..','my_data','traintest150827c.csv'), header=0)
tube_id = pd.read_csv(os.path.join('..','my_data','tube_assembly_id.csv'), header = 0)
#tube_id.columns = ['tube_assembly_id']


'''
Encode file and save train and test
'''
#traintest2 = traintest[bestcol].copy() #bestcol from parameters.py
traintest2 = traintest.copy()

################################################################################
### mean_weight, mean_quantity
################################################################################

traintest3 = traintest.copy()
component_id = traintest3.columns[traintest3.columns.str.contains('component_id')]
for x in component_id:
    traintest3.loc[pd.notnull(traintest[x]),x] = 1
    traintest3.loc[pd.isnull(traintest[x]),x] = 0
temp = traintest3[component_id[0]]
for x in component_id[1:]:
    temp += traintest3[x]
traintest2['comp_num'] = temp
traintest2.loc[traintest2['comp_num'] == 0, 'comp_num'] = 1.0
traintest2['mean_weight'] = traintest2['total_weight'].copy()
traintest2['mean_weight'] = traintest2['total_weight']/traintest2['comp_num'].astype(float)

################################################################################
### 1st component_id (make new column) engineering
### 1st component_type_id (make new column) engineering
################################################################################

traintest3 = traintest.copy()
componid = traintest2.columns[traintest2.columns.str.contains('component_id')]
for x in componid:
    traintest3.loc[pd.isnull(traintest3[x]),x] = ''
traintest2['comp_ids'] = traintest3['component_id_1']
for x in componid[1:]:
    traintest2['comp_ids'] += traintest3[x]
traintest2.loc[traintest2['comp_ids'] == '','comp_ids'] = np.nan


traintest3 = traintest.copy()
comp_type = traintest2.columns[traintest2.columns.str.contains('component_type_id')]
for x in comp_type:
    traintest3.loc[pd.isnull(traintest3[x]),x] = ''
traintest2['comp_type'] = traintest3['component_type_id_comp1']
for x in comp_type[1:]:
    traintest2['comp_type'] += traintest3[x]
traintest2.loc[traintest2['comp_type'] == '','comp_type'] = np.nan


traintest3 = traintest.copy()
conn_type = traintest2.columns[traintest2.columns.str.contains('connection_type_id')]
for x in conn_type:
    traintest3.loc[pd.isnull(traintest3[x]),x] = ''
traintest2['conn_type'] = traintest3['connection_type_id_comp1']
for x in conn_type[1:]:
    traintest2['conn_type'] += traintest3[x]
traintest2.loc[traintest2['conn_type'] == '','conn_type'] = np.nan
################################################################################
### SPECS engineering
################################################################################

### specs - encode from dataclean

traintest3 = traintest.copy()
specs = traintest2.columns[traintest2.columns.str.contains('spec')]
for x in specs:
    traintest3.loc[pd.isnull(traintest3[x]),x] = ''
traintest2['uniq_specs'] = traintest3['spec1']
for x in specs[1:]:
    traintest2['uniq_specs'] += traintest3[x]
traintest2.loc[traintest2['uniq_specs'] == '','uniq_specs'] = np.nan


# SPECS already encoded
# add total spec
# Add an empty column for total_specs
traintest2['total_specs'] = np.zeros(len(traintest2))
# Encode spec 01
total_spec = np.array(traintest2['total_specs'])
specs = traintest2.columns[traintest2.columns.str.contains('spec')]
specs = specs[:-2]
for s in specs:
    traintest2.loc[pd.notnull(traintest2[s]),s] = 1
    traintest2.loc[pd.isnull(traintest2[s]),s] = 0
for s in specs:
    total_spec += np.array(traintest2[s])
traintest2['total_specs'] = total_spec

################################################################################
### Supplier and material_id
################################################################################
lecolumns = ['supplier','material_id', 'uniq_specs', 'comp_ids', 'comp_type', 'conn_type']
traintest2 = encode(traintest2,lecolumns,TRANSFORM_CUTOFF)

################################################################################
### bracket_pricing
################################################################################
traintest2['bracket_pricing'] = [1 if x=='Yes' else 0 for x in traintest2.bracket_pricing.values]

################################################################################
### end_a_2x, end_x_1x, end_x_2x, end_a, end_x: PruneLabelEncoder from dataclean
################################################################################
traintest2['end_x_1x'] = [1 if x=='Y' else 0 for x in traintest2['end_x_1x']]
traintest2['end_x_2x'] = [1 if x=='Y' else 0 for x in traintest2['end_x_2x']]
traintest2['end_a_1x'] = [1 if x=='Y' else 0 for x in traintest2['end_a_1x']]
traintest2['end_a_2x'] = [1 if x=='Y' else 0 for x in traintest2['end_a_2x']]

enda_vals = traintest2.end_a.values
endx_vals = traintest2.end_x.values
end_vals = np.concatenate((enda_vals, endx_vals))
ple_end = PruneLabelEncoder()
ple_end.fit(end_vals, cutoff=TRANSFORM_CUTOFF)
traintest2['end_a'] = ple_end.transform(traintest2.end_a.values)
traintest2['end_x'] = ple_end.transform(traintest2.end_x.values)


forma_vals = traintest2.forming_a.values
formx_vals = traintest2.forming_x.values
form_vals = np.concatenate((forma_vals, formx_vals))
ple_form = PruneLabelEncoder()
ple_form.fit(form_vals, cutoff=TRANSFORM_CUTOFF)
traintest2['forming_a'] = ple_form.transform(traintest2.forming_a.values)
traintest2['forming_x'] = ple_form.transform(traintest2.forming_x.values)
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
comp_type = traintest2.columns[traintest2.columns.str.contains('component_type_id')]
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
comp_type = traintest2.columns[traintest2.columns.str.contains('component_type_id')]
for y in range(len(comp_type)):
    for x in traintest2[comp_type[y]].value_counts().index:
        traintest2.loc[traintest2[comp_type[y]] == x, comp_type[y]] = traintest2[comp_type[y]].value_counts()[x]
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
################################################################################# Connection TYPE
################################################################################
conn_type = traintest2.columns[traintest2.columns.str.contains('connection_type_id')]
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
    traintest2.loc[traintest2['total_conn'] == x, 'total_conn'] = traintest2['total_conn'].value_counts()[x]
"""

################################################################################# Incorporating Angle Features
################################################################################


seat_angle = traintest2.columns[traintest2.columns.str.contains('seat_angle')]
sangle = np.zeros(traintest2.shape[0])
for x in seat_angle:
    traintest2.loc[pd.notnull(traintest2[x]),x] = 1
    traintest2.loc[pd.isnull(traintest2[x]),x] = 0
    sangle += np.array(traintest2[x])
traintest2['angle_seat'] = sangle


elbow_angle = traintest2.columns[traintest2.columns.str.contains('elbow_angle')]
eangle = np.zeros(traintest2.shape[0])
for x in elbow_angle:
    traintest2.loc[pd.notnull(traintest2[x]),x] = 1
    traintest2.loc[pd.isnull(traintest2[x]),x] = 0
    eangle += np.array(traintest2[x])
traintest2['angle_elbow'] = eangle

adap_angle = traintest2.columns[traintest2.columns.str.contains('adaptor_angle')]
aangle = np.zeros(traintest2.shape[0])
for x in adap_angle:
    traintest2.loc[pd.notnull(traintest2[x]),x] = 1
    traintest2.loc[pd.isnull(traintest2[x]),x] = 0
    aangle += np.array(traintest2[x])
traintest2['angle_adaptor'] = aangle

traintest2['total_angle'] = traintest2['angle_seat'] + traintest2['angle_elbow'] + traintest2['angle_adaptor']

################################################################################# Incorporating volume
################################################################################
traintest2['volume'] = traintest2['length']*np.pi*(traintest2['diameter']/2.0)**2
traintest2['volume'] = np.log1p(traintest2['volume'])

################################################################################# Incorporating plating
################################################################################
plating = traintest2.columns[traintest2.columns.str.contains('plating')]
plate = np.zeros(traintest2.shape[0])
for x in plating:
    traintest2.loc[traintest2[x]!='Yes',x] = 0
    traintest2.loc[traintest2[x]=='Yes',x] = 1
    plate += np.array(traintest2[x])
traintest2['plate'] = plate

################################################################################# Base Type
################################################################################


base_type = traintest2.columns[traintest2.columns.str.contains('base_type')]
btype = np.zeros(traintest2.shape[0])
for x in base_type:
    traintest2.loc[pd.notnull(traintest2[x]),x] = 1
    traintest2.loc[pd.isnull(traintest2[x]),x] = 0
    btype += np.array(traintest2[x])
traintest2['base'] = btype

################################################################################# Scaling my continuous variables (ie. length)
################################################################################
#fields = ["annual_usage", "min_order_quantity", "quantity", "diameter", "wall", "length", "num_bends", "bend_radius"]
#fields = ["diameter", "wall", "length", "bend_radius"]


#fields = ["total_weight"]
"""
fields = []
# length
for col in fields:
    if col in ['quantity']:
        traintest2[col] = 1/(np.log1p(traintest2[col])).astype(float)
    if col in ['day']:
        traintest2[col] = np.exp(traintest2[col]/max(traintest2[col]).astype(float))
    if col in ['total_weight']:
        traintest2[col] = np.log1p(np.log1p(np.log1p(traintest2[col])))
    if col in ['annual_usage', 'bend_radius', 'diameter']:
        traintest2[col] = np.log1p(traintest2[col])
    if col in ['length']:
        lengths = traintest2.columns[traintest2.columns.str.contains('length_')]
        for x in lengths:
            traintest2.loc[pd.isnull(traintest2[x]),x] = 0
            traintest2['length'] += traintest2[x]
"""
################################################################################# Adding groupby features
################################################################################


temp = traintest2[['supplier','total_specs']].groupby('supplier').count()
temp['supplier'] = temp.index
temp.columns = ['count_specs','supplier']
traintest2 = pd.merge(traintest2,temp, on = 'supplier', how = 'left')


temp2 = traintest2[['supplier','total_specs']].groupby('supplier').mean()
temp2['supplier'] = temp2.index
temp2.columns = ['mean_specs','supplier']
traintest2 = pd.merge(traintest2,temp2, on = 'supplier', how = 'left')

temp3 = traintest2[['supplier','annual_usage']].groupby('supplier').mean()
temp3['supplier'] = temp3.index
temp3.columns = ['mean_ann_usage','supplier']
traintest2 = pd.merge(traintest2,temp3, on = 'supplier', how = 'left')

temp3 = traintest2[['tube_assembly_id','quantity']].groupby('tube_assembly_id').mean()
temp3['tube_assembly_id'] = temp3.index
temp3.columns = ['mean_quantity','tube_assembly_id']
traintest2 = pd.merge(traintest2,temp3, on = 'tube_assembly_id', how = 'left')

"""
temp3 = traintest2[['supplier','quantity']].groupby('supplier').mean()
temp3['supplier'] = temp3.index
temp3.columns = ['mean_quantity','supplier']
traintest2 = pd.merge(traintest2,temp3, on = 'supplier', how = 'left')
"""

################################################################################# Create train and test
################################################################################


traintest2 = traintest2[bestcol] #bestcol from parameters.py
train = traintest2.iloc[0:30213]
test = traintest2.iloc[30213:]
tube_id2 = tube_id.iloc[0:30213]



################################################################################
# One Hot encoding - if one_hot is True, perform one hot encoding on select
# columns. one_hot is from parameters.py
################################################################################
if one_hot:
    train_cost = train['cost'].copy()
    test_cost = test['cost'].copy()
    train = train.drop(['cost'], axis = 1)
    test = test.drop(['cost'], axis = 1)

    #onehotcol = ['supplier', 'month', 'total_specs', 'total_quantity', 'num_bends', 'wall', 'week', 'bend_radius','min_order_quantity']
    #for one hot encoding - need to make sure test can fit in

    #test.loc[test['total_specs'] == 10, 'total_specs'] = 9
    onehotcol = ['supplier', 'month', 'total_quantity', 'bend_radius']
    #onehotcol = ['supplier', 'month', 'total_quantity', 'bend_radius', 'end_a', 'end_x']
    for col in onehotcol:
        train_temp = train[col].values
        test_temp = test[col].values
        combine = np.concatenate((train_temp,test_temp))
        lbl = OneHotEncoder()
        lbl.fit(np.resize(np.array(combine).astype(float), (len(combine),1)))
        #lbl.fit(np.resize(np.array(train_temp).astype(float), (len(train_temp),1)))

        train_temp = lbl.transform(np.resize(np.array(train_temp).astype(float), (len(train_temp),1))).toarray()
        test_temp = lbl.transform(np.resize(np.array(test_temp).astype(float), (len(test_temp),1))).toarray()

        for i in range(train_temp.shape[1]):
            train[col + "_" + str(i)] = train_temp[:,i]
        for i in range(test_temp.shape[1]):
            test[col + "_" + str(i)] = test_temp[:, i]
        train = train.drop([col], axis = 1)
        test = test.drop([col],axis = 1)

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
if one_hot:
    print "Used One Hot Encoding on:", onehotcol
print

if x_gb:
    # file_train from parameters.py
    train.to_csv(file_train, index = False)
    print 'XGB'
    print
    print 'File created:', file_train
    print 'DataFrame shape:', train.shape
    print

    # file_test from parameters.py
    test.to_csv(file_test, index = False)
    print 'File created:', file_test
    print 'DataFrame shape:', test.shape
    print

elif rf:
    train.to_csv(rf_train, index = False)
    print 'Random Forest'
    print
    print 'File created:', rf_train
    print 'DataFrame shape:', train.shape
    print

    # file_test from parameters.py
    test.to_csv(rf_test, index = False)
    print 'File created:', rf_test
    print 'DataFrame shape:', test.shape
    print

elif keras:
    train.to_csv(file_train, index = False)
    print 'Neural Network (Keras)'
    print
    print 'File created:', keras_train
    print 'DataFrame shape:', train.shape
    print

    # file_test from parameters.py
    test.to_csv(file_test, index = False)
    print 'File created:', keras_test
    print 'DataFrame shape:', test.shape
    print
else:
    print 'Did not save to file. Please specify booleans for x_gb, rf, keras'
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


"""
GRAVEYARD
"""

### all quantity_*
"""
quantids = traintest2.columns[traintest2.columns.str.contains('quantity_')]
for qid in quantids:
    traintest2[qid] = [0 if np.isnan(x) else x for x in traintest2[qid].values]
"""





"""
################################################################################
# BOOM
################################################################################
### material_id - frequency encoding
for x in traintest2['material_id'].value_counts().index:
    traintest2.loc[traintest2['material_id'] == x, 'material_id'] = traintest2['material_id'].value_counts()[x]
    traintest2.loc[pd.isnull(traintest2['material_id']),'material_id'] = 0

### supplier - encode from dataclean
for x in traintest2['supplier'].value_counts().index:
    traintest2.loc[traintest2['supplier'] == x, 'supplier'] = traintest2['supplier'].value_counts()[x]
    traintest2.loc[pd.isnull(traintest2['supplier']),'supplier'] = 0

#traintest2['end_a'] = [1 if x =='Yes' else 0 for x in traintest['end_a']]
#traintest2['end_x'] = [1 if x =='Yes' else 0 for x in traintest['end_x']]
################################################################################
# BOOM
################################################################################



ENDVALS
################################################################################
# BOOM
################################################################################
end_vals = ['end_a','end_x']
for end in end_vals:
    for x in traintest2[end].value_counts().index:
        traintest2.loc[traintest2[end] == x, end] = traintest2[end].value_counts()[x]
    traintest2.loc[pd.isnull(traintest2[end]),end] = 0
################################################################################
# BOOM
################################################################################




COMPONENT_ID
################################################################################
# BOOM
################################################################################
# frequency encode
compids = traintest2.columns[traintest2.columns.str.contains('component_id_')]
comp_vals = np.array([])
for cid in compids:
    traintest2.loc[pd.isnull(traintest2[cid]),cid] = 0
    comp_vals = np.concatenate((comp_vals, np.array(traintest2[cid])))
comp_vals = pd.Series(comp_vals)
for cid in compids:
    for x in comp_vals.value_counts().index:
        if x == 0:
            continue
        elif x in traintest2[cid].unique():
            traintest2.loc[traintest2[cid] == x, cid] = traintest2[cid].value_counts()[x]
################################################################################
# BOOM
################################################################################

################################################################################
# BOOM
################################################################################
compids = traintest2.columns[traintest2.columns.str.contains('component_id_')]
for cid in compids:
    for x in traintest2[cid].value_counts().index:
        traintest2.loc[traintest2[cid] == x, cid] = traintest2[cid].value_counts()[x]
    traintest2.loc[pd.isnull(traintest2[cid]),cid] = 0
################################################################################
# BOOM
################################################################################

"""
