%matplotlib inline
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from math import sqrt
filepath = 'C:/Users/Anthony/Dropbox/Background Understanding/Data Science/CDIPS/CDIPS15_TeamCat/data/competition_data/'
files = [x for x in os.listdir(filepath)] #if os.path.isfile(os.path.join(filepath,x)) if you want to check isfile
# Or
# all_files = glob.glob(os.path.join('..','data\competition_data','*.csv'))
allfiles = glob.glob(os.path.join('..','data\competition_data','*.csv'))
comp_files = glob.glob(os.path.join('..','data\competition_data','comp_*'))

all_files = {}
for afile in allfiles:
    key = afile.split('\\')[-1].split('.')[0]
    all_files[key] = pd.read_csv(afile, header = 0)
all_files['train_set'] = pd.read_csv('../data/competition_data/train_set.csv', header = 0, parse_dates = ['quote_date'])
all_files['test_set'] = pd.read_csv('../data/competition_data/test_set.csv', header = 0, parse_dates = ['quote_date'])
print all_files.keys()

comp_dict = {}
for compfile in comp_files:
    key = compfile.split('\\')[-1].split('.')[0]
    comp_dict[key] = pd.read_csv(compfile)
comp_dict.keys()


'''
Merge files into train and test dataframes
merging 'bill_of_materials.csv' and 'tube.csv' with training set and test set.

'''

billdf = pd.read_csv(os.path.join(filepath,'bill_of_materials.csv'),header=0)
tubedf = pd.read_csv(os.path.join(filepath,'tube.csv'),header=0)
traindf = pd.read_csv(os.path.join(filepath,'train_set.csv'), header = 0, parse_dates = ['quote_date'])
testdf = pd.read_csv(os.path.join(filepath,'test_set.csv'), header = 0, parse_dates = ['quote_date'])

train = pd.merge(traindf,billdf, on = 'tube_assembly_id')
test = pd.merge(testdf,billdf, on = 'tube_assembly_id')
train = pd.merge(train,tubedf, on = 'tube_assembly_id')
test = pd.merge(test,tubedf, on = 'tube_assembly_id')

'''
Clean NaN values

For component_id_1,2,3,... columns, I replaced NaN values with an empty string.

For quantity_1,2,3,... columns, I replaced NaN values with 0

from TrainTest02
'''

for x in range(7,24,2):
    column = train.columns[x]
    train[column].replace(np.nan,0, regex=True, inplace= True)

for x in range(8,23,2):
    column = train.columns[x]
    train[column].replace(np.nan,'',regex=True, inplace= True)

for x in range(7,24,2):
    column = test.columns[x]
    test[column].replace(np.nan,0,regex=True, inplace= True)

for x in range(8,23,2):
    column = test.columns[x]
    test[column].replace(np.nan,'', regex=True, inplace= True)


'''
Dropping Columns

From bill_of_materials.csv, I dropped these columns from train and test:

    'component_id_4', 'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8'
    'quantity_4','quantity_5', 'quantity_6', 'quantity_7', 'quantity_8'

From tube.csv, I dropped these columns from train and test:

    'end_a_1x','end_x_1x','num_bracket','num_boss','other'

From test, I dropped:

    'id','annual_usage','min_order_quantity'

From train, I dropped:

    'annual_usage','min_order_quantity'

These choices were made based of "Initial Exploration" ipython notebook and Bharat's TrainTest01 updated ipython notebook. Most of these columns are essentially single valued or are useless (like 'id')
'''

traindrop = ['annual_usage','min_order_quantity','component_id_4','component_id_5', 'component_id_6', 'component_id_7', 'component_id_8','quantity_4','quantity_5', 'quantity_6', 'quantity_7', 'quantity_8','end_a_1x','end_x_1x','num_bracket','num_boss','other']
testdrop =  ['id','annual_usage','min_order_quantity','component_id_4','component_id_5', 'component_id_6', 'component_id_7', 'component_id_8','quantity_4','quantity_5', 'quantity_6', 'quantity_7', 'quantity_8','end_a_1x','end_x_1x','num_bracket','num_boss','other']
train = train.drop(traindrop, axis = 1)
test = test.drop(testdrop, axis = 1)

'''
Add year, month, and days column

Since we only care about 2011-2014, I referenced days from the 0 point 12/31/2010. So 01/01/2011 is 1, 01/02/2011 is 2, etc...

After that, I dropped the 'quote_date' column because it's now superfluous.
'''

train['year'] = train['quote_date'].dt.year
train['month'] = train['quote_date'].dt.month
train['day'] = [date.days for date in train['quote_date'] - dt.date(2010,12,31)]

test['year'] = test['quote_date'].dt.year
test['month'] = test['quote_date'].dt.month
test['day'] = [date.days for date in test['quote_date'] - dt.date(2010,12,31)]


'''
Subsetting
'bracket_pricing': 'Yes'
supplier: 'S-0066'
year: only 2011-2015
'''
train = train[(train['bracket_pricing'] == 'Yes') & (train['supplier'] == 'S-0066') & train['year'].isin(range(2011,2015))]
test = test[(test['bracket_pricing'] == 'Yes') & (test['supplier'] == 'S-0066') & test['year'].isin(range(2011,2015))]
