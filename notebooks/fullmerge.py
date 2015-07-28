# %matplotlib inline
import os, glob, sys
import pandas as pd
import numpy as np
import datetime as dt
from math import sqrt
from collections import Counter
import copy



## Read in files

allfiles = glob.glob(os.path.join('..','competition_data','*.csv'))
comp_files = glob.glob(os.path.join('..','competition_data','comp_*'))
newtrain_df = pd.read_csv(os.path.join('..','train.csv'),index_col = 0)

all_files = {}
for afile in allfiles:
    key = afile.split('\\')[-1].split('.')[0]
    all_files[key] = pd.read_csv(afile, header = 0)
all_files['train_set'] = pd.read_csv('../competition_data/train_set.csv', header = 0, parse_dates = ['quote_date'])
all_files['test_set'] = pd.read_csv('../competition_data/test_set.csv', header = 0, parse_dates = ['quote_date'])

# Components
comp_dict = {}
for compfile in comp_files:
    key = compfile.split('\\')[-1].split('.')[0]
    comp_dict[key] = pd.read_csv(compfile,header=0, index_col=0)

# The rest of the files
rest = ['components.csv','specs.csv','tube_end_form.csv','type_component.csv','type_connection.csv','type_end_form.csv']
restfile = []
for x in rest:
    restfile += [os.path.join('..','competition_data',x)]
rest_files = {}
for r in restfile:
    key = r.split('\\')[-1].split('.')[0]
    rest_files[key] = pd.read_csv(r ,header=0)


'''
Initially merge train_set/test_set with bill_of_materials and tube.

Create year, month, and day variables.

Add weight to each component_id_*
'''


## Start merging


train = pd.merge(all_files['train_set'],all_files['bill_of_materials'], on = 'tube_assembly_id')
test = pd.merge(all_files['test_set'],all_files['bill_of_materials'], on = 'tube_assembly_id')
train = pd.merge(train,all_files['tube'], on = 'tube_assembly_id')
test = pd.merge(test,all_files['tube'], on = 'tube_assembly_id')

train['year'] = train['quote_date'].dt.year
train['month'] = train['quote_date'].dt.month
train['day'] = [date.days for date in train['quote_date'] - dt.date(1982,9,22)]

test['year'] = test['quote_date'].dt.year
test['month'] = test['quote_date'].dt.month
test['day'] = [date.days for date in test['quote_date'] - dt.date(1985,11,16)]


comp_id = ['component_id_'+str(i) for i in range(1,9)]
weight_id = ['weight_id_'+str(i) for i in range(1,9)]

# Add weight to train
for key1 in comp_id:
    i = comp_id.index(key1)
    weight = []
    for key2 in sorted(train[key1].unique(),reverse = True):
        for filename in comp_dict:
            if key2 in comp_dict[filename].index:
                weight.append(comp_dict[filename].loc[key2]['weight'])
    else:
        weight.append(np.nan)
    dfTemp = pd.DataFrame({key1 : sorted(train[key1].unique(),reverse = True),weight_id[i]: weight})
    train = pd.merge(train,dfTemp, how = 'left', on = key1)
train = train[['tube_assembly_id', 'supplier', 'year','month','day', 'annual_usage', 'min_order_quantity', 'bracket_pricing', 'quantity', 'cost', 'component_id_1', 'quantity_1','weight_id_1', 'component_id_2', 'quantity_2', 'weight_id_2', 'component_id_3', 'quantity_3', 'weight_id_3', 'component_id_4', 'quantity_4','weight_id_4', 'component_id_5', 'quantity_5', 'weight_id_5', 'component_id_6', 'quantity_6', 'weight_id_6', 'component_id_7', 'quantity_7', 'weight_id_7', 'component_id_8', 'quantity_8', 'weight_id_8', 'material_id', 'diameter', 'wall', 'length', 'num_bends', 'bend_radius', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x', 'num_boss', 'num_bracket', 'other']]

# Add weight to test


for key in comp_id:
    j = comp_id.index(key)
    temp = pd.DataFrame({comp_id[j]: []})
    for filename in comp_dict:
        for key2 in sorted(test[comp_id[j]].unique(),reverse = True):
            i = sorted(test[comp_id[j]].unique(),reverse = True).index(key2)
            if key2 in comp_dict[filename].index:
                temp.loc[i,comp_id[j]] = key2
                temp.loc[i,weight_id[j]] = comp_dict[filename].loc[key2]['weight']
    test = pd.merge(test,temp,how = 'left',on = comp_id[j])
test = test[['id', 'tube_assembly_id', 'supplier', 'year','month','day', 'annual_usage', 'min_order_quantity', 'bracket_pricing', 'quantity', 'component_id_1', 'quantity_1', 'weight_id_1', 'component_id_2', 'quantity_2', 'weight_id_2', 'component_id_3', 'quantity_3', 'weight_id_3', 'component_id_4', 'quantity_4', 'weight_id_4', 'component_id_5', 'quantity_5', 'weight_id_5', 'component_id_6', 'quantity_6', 'weight_id_6', 'component_id_7', 'quantity_7', 'component_id_8', 'quantity_8', 'material_id', 'diameter', 'wall', 'length', 'num_bends', 'bend_radius', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x', 'num_boss', 'num_bracket', 'other']]


'''
## MERGING COMP_*.CSV information
**Add component columns to each `component_id_*`**
'''
# TRAIN

def comp_column(name):
    if name not in ['component_id_'+str(i) for i in range(1,9)]:
        return "Error"
    else:
        temp = pd.DataFrame({name: []})
        for filename in comp_dict:
            tempdict = {x: [] for x in comp_dict[filename].drop('weight',axis = 1).columns}
            for key in sorted(train[name].unique(),reverse = True):
                i = sorted(train[name].unique(),reverse = True).index(key)
                if key in comp_dict[filename].index:
                    for col in tempdict.keys():
                        temp.loc[i,name] = key
                        temp.loc[i,col] = comp_dict[filename].drop('weight',axis = 1).loc[key][col]

        # Dropping any pure NaN columns
        tempcol = temp.columns.copy()
        dropcol = []
        for x in tempcol:
            if sum(pd.isnull(temp[x])) == len(temp[x]):
                dropcol.append(x)
        temp = temp.drop(dropcol,axis=1)

        # rename columns for number
        c = name.split('_')[-1]
        temp.columns = [temp.columns[i] + '_comp' + c if i > 0 else temp.columns[i] for i in range(len(temp.columns))]
        return temp

def mergeframe(comp_num):
    global train

    # Call comp_column()
    comp = 'component_id_'+str(comp_num)
    merge = comp_column(comp)

    train = pd.merge(train,merge, how = 'left', on = comp)
    startIndex = list(train.columns).index('weight_id_'+str(comp_num))
    x = list(train.columns.copy())
    newColOrder = []
    for i in range(len(x)):
        if i <= startIndex:
            newColOrder += [x[i]]
        elif i > startIndex and i < startIndex+len(merge.columns):
            newColOrder += [x[-len(merge.columns)+i-startIndex]]
        else:
            newColOrder += [x[i-len(merge.columns)+1]]
    return train[newColOrder]

# TEST

def comp_columnTest(name):
    if name not in ['component_id_'+str(i) for i in range(1,9)]:
        return "Error"
    else:
        temp = pd.DataFrame({name: []})
        for filename in comp_dict:
            tempdict = {x: [] for x in comp_dict[filename].drop('weight',axis = 1).columns}
            for key in sorted(test[name].unique(),reverse = True):
                i = sorted(test[name].unique(),reverse = True).index(key)
                if key in comp_dict[filename].index:
                    for col in tempdict.keys():
                        temp.loc[i,name] = key
                        temp.loc[i,col] = comp_dict[filename].drop('weight',axis = 1).loc[key][col]

        # Dropping any pure NaN columns
        tempcol = temp.columns.copy()
        dropcol = []
        for x in tempcol:
            if sum(pd.isnull(temp[x])) == len(temp[x]):
                dropcol.append(x)
        temp = temp.drop(dropcol,axis=1)

        # rename columns for number
        c = name.split('_')[-1]
        temp.columns = [temp.columns[i] + '_comp' + c if i > 0 else temp.columns[i] for i in range(len(temp.columns))]
        return temp

def mergeframetest(comp_num):
    global test

    # Call comp_column()
    comp = 'component_id_'+str(comp_num)
    merge = comp_columnTest(comp)

    test = pd.merge(test,merge, how = 'left', on = comp)
    startIndex = list(test.columns).index('weight_id_'+str(comp_num))
    x = list(test.columns.copy())
    newColOrder = []
    for i in range(len(x)):
        if i <= startIndex:
            newColOrder += [x[i]]
        elif i > startIndex and i < startIndex+len(merge.columns):
            newColOrder += [x[-len(merge.columns)+i-startIndex]]
        else:
            newColOrder += [x[i-len(merge.columns)+1]]
    return test[newColOrder]

# Merge them all! only doing 1-6 on test because 7 and 8 are empty - drop them!
for i in range(1,7):
    test = mergeframetest(i)

for i in range(1,9):
    train = mergeframe(i)

'''
Checking for full null columns in test

Any fully null column will be dropped

print statements are there to check if script works
'''
# print 'test'
droptestcol = []
for x in test.columns:
    if sum(pd.isnull(test[x])) == len(test[x]):
        droptestcol += [x]
        # print x
# print
# print 'train'
droptraincol = []
for x in train.columns:
    if sum(pd.isnull(train[x])) == len(train[x]):
        droptraincol += [x]
        # print x
test = test.drop(droptestcol, axis = 1)
train = train.drop(droptraincol, axis = 1)
# print '# of train columns:', len(train.columns)
# print '# of test columns:',  len(test.columns)

'''
**TUBE_END_FORM** incorporation

taking care of the 'end_a' and 'end_x' columns
'''

def tube_end(col):
    global train
    train2 = train.copy()
    re = rest_files['tube_end_form'].copy()
    tem = []
    if col not in ['end_a','end_x']:
        return 'Error'
    else:
        for i in train2[col]:
            if i == 'NONE':
                tem += [np.nan]
            else:
                t = re[re['end_form_id'] == i]['forming']
                tem += [t.reset_index()['forming'][0]]
    train[col] = tem
    return train

def tube_endTest(col):
    global test
    test2 = test.copy()
    re = rest_files['tube_end_form'].copy()
    tem = []
    if col not in ['end_a','end_x']:
        return 'Error'
    else:
        for i in test2[col]:
            if i == 'NONE':
                tem += [np.nan]
            else:
                t = re[re['end_form_id'] == i]['forming']
                tem += [t.reset_index()['forming'][0]]
    test[col] = tem
    return test


train = tube_end('end_a')
train = tube_end('end_x')

test = tube_endTest('end_a')
test = tube_endTest('end_x')


'''
TYPE_END_FORM incorporation

'''


endformtype = train.columns[train.columns.str.contains('end_form')]
endformtypeTest = test.columns[test.columns.str.contains('end_form')]

def type_end(col):
    train2 = train.copy()
    re = rest_files['type_end_form'].copy()
    typeEnd = []
    if col not in endformtype:
        return 'Error'
    else:
        for i in train2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i):
                typeEnd += [np.nan]
            else:
                t = re[re['end_form_id'] == i]['name']
                typeEnd += [t.reset_index()['name'][0]]
    return typeEnd


def type_endTest(col):
    test2 = test.copy()
    re = rest_files['type_end_form'].copy()
    typeEnd = []
    if col not in endformtypeTest:
        return 'Error'
    else:
        for i in test2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i):
                typeEnd += [np.nan]
            else:
                t = re[re['end_form_id'] == i]['name']
                typeEnd += [t.reset_index()['name'][0]]
    return typeEnd


for i in endformtype:
    train[i] = type_end(i)
for i in endformtypeTest:
    test[i] = type_endTest(i)

'''
CONNECTION_TYPE_ incorporation

'''

contype = train.columns[train.columns.str.contains('conn')]
contypeTest = test.columns[test.columns.str.contains('conn')]
def conn_type(col):
    train2 = train.copy()
    re = rest_files['type_connection'].copy()
    conn = []
    if col not in contype:
        return 'Error'
    else:
        for i in train2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i):
                conn += [np.nan]
            else:
                t = re[re['connection_type_id'] == i]['name']
                conn += [t.reset_index()['name'][0]]
    return conn


def conn_typeTest(col):
    test2 = test.copy()
    re = rest_files['type_connection'].copy()
    conn = []
    if col not in contypeTest:
        return 'Error'
    else:
        for i in test2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i):
                conn += [np.nan]
            else:
                t = re[re['connection_type_id'] == i]['name']
                conn += [t.reset_index()['name'][0]]
    return conn

# train3 = train.copy()
# test3 = test.copy()
for i in contype:
    train[i] = conn_type(i)
for i in contypeTest:
    test[i] = conn_typeTest(i)


'''
TYPE_COMPONENT incorporation
'''

comptype = train.columns[train.columns.str.contains('component_type')]
comptypeTest = test.columns[test.columns.str.contains('component_type')]
def comp_type(col):
    train2 = train.copy()
    re = rest_files['type_component'].copy()
    comp = []
    if col not in comptype:
        return 'Error'
    else:
        for i in train2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i) or i == 'OTHER':
                comp += [np.nan]
            else:
                t = re[re['component_type_id'] == i]['name']
                comp += [t.reset_index()['name'][0]]
    return comp


def comp_typeTest(col):
    test2 = test.copy()
    re = rest_files['type_component'].copy()
    comp = []
    if col not in comptypeTest:
        return 'Error'
    else:
        for i in test2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i) or i == 'OTHER':
                comp += [np.nan]
            else:
                t = re[re['component_type_id'] == i]['name']
                comp += [t.reset_index()['name'][0]]
    return comp

#train3 = train.copy()
#test3 = test.copy()
for i in comptype:
    train[i] = comp_type(i)
for i in comptypeTest:
    test[i] = comp_typeTest(i)

'''
Add SPECS data frame by merging
'''

spc = rest_files['specs'].copy()
specstrain = spc[spc['tube_assembly_id'].isin(train['tube_assembly_id'])]
specstest = spc[spc['tube_assembly_id'].isin(test['tube_assembly_id'])]

#train3 = train.copy()
#test3 = test.copy()

train = pd.merge(train,specstrain, how = 'left', on = 'tube_assembly_id')
test = pd.merge(test,specstest, how = 'left', on = 'tube_assembly_id')


'''
Add COMPONENT NAMES to test and train
'''


componName = train.columns[train.columns.str.contains('component_id')]
componNameTest = test.columns[test.columns.str.contains('component_id')]

def comp_name(col):
    train2 = train.copy()
    re = rest_files['components'].copy()
    compName = []
    if col not in componName:

        return 'Error'
    else:
        for i in train2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i) or i == 'OTHER':
                compName += [np.nan]
            else:
                t = re[re['component_id'] == i]['name']
                compName += [t.reset_index()['name'][0]]
    return compName


def comp_nameTest(col):
    test2 = test.copy()
    re = rest_files['components'].copy()
    compName = []
    if col not in componNameTest:
        return 'Error'
    else:
        for i in test2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i) or i == 'OTHER':
                compName += [np.nan]
            else:
                t = re[re['component_id'] == i]['name']
                compName += [t.reset_index()['name'][0]]
    return compName

#train3 = train.copy()
#test3 = test.copy()
for i in componName:
    train[i] = comp_name(i)
for i in componNameTest:
    test[i] = comp_nameTest(i)
