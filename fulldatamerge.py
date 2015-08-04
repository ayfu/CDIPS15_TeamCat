# %matplotlib inline
import os, glob, sys
import pandas as pd
import numpy as np
import datetime as dt
from math import sqrt
from collections import Counter
import copy

## Read in files

allfiles = glob.glob(os.path.join('competition_data','*.csv'))
comp_files = glob.glob(os.path.join('competition_data','comp_*'))
#newtrain_df = pd.read_csv(os.path.join('train.csv'),index_col = 0)

all_files = {}
for afile in allfiles:
    key = afile.split('\\')[-1].split('.')[0]
    all_files[key] = pd.read_csv(afile, header = 0)
all_files['train_set'] = pd.read_csv('competition_data/train_set.csv', header = 0, parse_dates = ['quote_date'])
all_files['test_set'] = pd.read_csv('competition_data/test_set.csv', header = 0, parse_dates = ['quote_date'])

# Components
comp_dict = {}
for compfile in comp_files:
    key = compfile.split('\\')[-1].split('.')[0]
    comp_dict[key] = pd.read_csv(compfile,header=0, index_col=0)

# The rest of the files
rest = ['components.csv','specs.csv','tube_end_form.csv','type_component.csv','type_connection.csv','type_end_form.csv']
restfile = []
for x in rest:
    restfile += [os.path.join('competition_data',x)]
rest_files = {}
for r in restfile:
    key = r.split('\\')[-1].split('.')[0]
    rest_files[key] = pd.read_csv(r ,header=0)



'''
concatenate Train and Test together and build a traintest data set to label encode you can subset out the data later with
traintest.loc[len(temptrain):,:] -- something like that
'''


temptrain = all_files['train_set']
temptest = all_files['test_set'].drop('id',axis=1)
print '# of observations train:', len(temptrain['supplier'])
print '# of observations test:', len(temptest['supplier'])
temptest['cost'] = [np.nan]*len(temptest['supplier'])
traintest = pd.concat([temptrain,temptest], ignore_index = True)

traintest = pd.merge(traintest, all_files['bill_of_materials'], on = 'tube_assembly_id')
traintest = pd.merge(traintest, all_files['tube'], on = 'tube_assembly_id')


# Deal with dates - 1/1/1982 is used as the reference date

traintest['year'] = traintest['quote_date'].dt.year
traintest['month'] = traintest['quote_date'].dt.month
traintest['day'] = [date.days for date in traintest['quote_date'] - dt.date(1982,1,1)]


# Add weight
comp_id = ['component_id_'+str(i) for i in range(1,9)]
weight_id = ['weight_id_'+str(i) for i in range(1,9)]

for key1 in comp_id:
    i = comp_id.index(key1)
    weight = []
    for key2 in sorted(traintest[key1].unique(),reverse = True):
        if pd.isnull(key2) or key2 == '9999':
            weight += [np.nan]
        for filename in comp_dict:
            if key2 in comp_dict[filename].index:
                weight.append(comp_dict[filename].loc[key2]['weight'])
    #else:
        #weight.append(np.nan)
    dfTemp = pd.DataFrame({key1 : sorted(traintest[key1].unique(),reverse = True),weight_id[i]: weight})
    traintest = pd.merge(traintest,dfTemp, how = 'left', on = key1)

traintest = traintest[['tube_assembly_id', 'supplier','year','month','day', 'annual_usage', 'min_order_quantity', 'bracket_pricing', 'quantity', 'cost', 'component_id_1', 'quantity_1','weight_id_1', 'component_id_2', 'quantity_2', 'weight_id_2', 'component_id_3', 'quantity_3', 'weight_id_3', 'component_id_4', 'quantity_4','weight_id_4', 'component_id_5', 'quantity_5', 'weight_id_5', 'component_id_6', 'quantity_6', 'weight_id_6', 'component_id_7', 'quantity_7', 'weight_id_7', 'component_id_8', 'quantity_8', 'weight_id_8', 'material_id', 'diameter', 'wall', 'length', 'num_bends', 'bend_radius', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a', 'end_x', 'num_boss', 'num_bracket', 'other']]


# ADDING comp_*.csv files
'''
## MERGING COMP_*.CSV information
**Add component columns to each `component_id_*`**
'''

def comp_column(name):
    if name not in ['component_id_'+str(i) for i in range(1,9)]:
        return "Error"
    else:
        temp = pd.DataFrame({name: []})
        for filename in comp_dict:
            tempdict = {x: [] for x in comp_dict[filename].drop('weight',axis = 1).columns}
            for key in sorted(traintest[name].unique(),reverse = True):
                i = sorted(traintest[name].unique(),reverse = True).index(key)
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
    global traintest

    # Call comp_column()
    comp = 'component_id_'+str(comp_num)
    merge = comp_column(comp)

    traintest = pd.merge(traintest,merge, how = 'left', on = comp)
    startIndex = list(traintest.columns).index('weight_id_'+str(comp_num))
    x = list(traintest.columns.copy())
    newColOrder = []
    for i in range(len(x)):
        if i <= startIndex:
            newColOrder += [x[i]]
        elif i > startIndex and i < startIndex+len(merge.columns):
            newColOrder += [x[-len(merge.columns)+i-startIndex]]
        else:
            newColOrder += [x[i-len(merge.columns)+1]]
    return traintest[newColOrder]

for i in range(1,9):
    traintest = mergeframe(i)

'''
Checking for full null columns in test

Any fully null column will be dropped

print statements are there to check if script works
'''


droptraincol = []
for x in traintest.columns:
    if sum(pd.isnull(traintest[x])) == len(traintest[x]):
        droptraincol += [x]
        # print x
traintest = traintest.drop(droptraincol, axis = 1)
# print '# of traintest columns:', len(traintest.columns)



'''
**TUBE_END_FORM** incorporation

taking care of the 'end_a' and 'end_x' columns
'''


def tube_end(col):
    global traintest
    train2 = traintest.copy()
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
    traintest[col] = tem
    return traintest

traintest = tube_end('end_a')
traintest = tube_end('end_x')



'''
TYPE_END_FORM incorporation

'''


endformtype = traintest.columns[traintest.columns.str.contains('end_form')]

def type_end(col):
    train2 = traintest.copy()
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

for i in endformtype:
    traintest[i] = type_end(i)



'''
CONNECTION_TYPE_ incorporation

'''

contype = traintest.columns[traintest.columns.str.contains('conn')]
def conn_type(col):
    train2 = traintest.copy()
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



# train3 = traintest.copy()

for i in contype:
    traintest[i] = conn_type(i)


'''
TYPE_COMPONENT incorporation
'''

comptype = traintest.columns[traintest.columns.str.contains('component_type')]
def comp_type(col):
    train2 = traintest.copy()
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

#train3 = traintest.copy()

for i in comptype:
    traintest[i] = comp_type(i)



'''
Add SPECS data frame by merging
'''

spc = rest_files['specs'].copy()
specstrain = spc[spc['tube_assembly_id'].isin(traintest['tube_assembly_id'])]

#train3 = traintest.copy()

traintest = pd.merge(traintest,specstrain, how = 'left', on = 'tube_assembly_id')



'''
Add COMPONENT TYPE ID to traintest
'''


componName = traintest.columns[traintest.columns.str.contains('component_id')]

def comp_name(col):
    train2 = traintest.copy()
    re = rest_files['components'].copy()
    compName = []
    if col not in componName:

        return 'Error'
    else:
        for i in train2[col]:
            if i == 'NONE' or i == np.nan or pd.isnull(i) or i == 'OTHER':
                compName += [np.nan]
            else:
                t = re[re['component_id'] == i]['component_type_id']
                compName += [t.reset_index()['component_type_id'][0]]
    return compName

#train3 = traintest.copy()

for i in componName:
    traintest[i] = comp_name(i)



'''
CLEAN OUT DIRTY COLUMN
'''

c = traintest['nominal_size_1_comp1'].copy()
print "# of 'See Drawing' observations:", len(c[c == 'See Drawing'])

for x in range(len(c)):
    if pd.isnull(c[x]):
        continue
    else:
        try:
            c[x] = float(c[x])
        except:
            c[x] = np.nan

traintest['nominal_size_1_comp1'] = c
