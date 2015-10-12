'''
__file__

    dataclean.py

__description__

    This file is meant to provide utility functions for encoding and cleaning the full dataframe.

'''

import os
import glob
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

class PruneLabelEncoder(LabelEncoder):
    def __init___(self):
        super(PruneLabelEncoder, self).__init__()
    def fit(self, series, cutoff=10):
        self.cutoff = cutoff
        # Generate the transformation classes and also the map for low output munging
        super(PruneLabelEncoder, self).fit(series)
        trans_series = super(PruneLabelEncoder, self).transform(series)
        self.val_count_map = defaultdict(int)
        for i in trans_series:
            self.val_count_map[i] += 1
        # identify the first key with low frequency and use it for all low freq vals
        for key, val in self.val_count_map.items():
            if val < self.cutoff:
                self.low_cnt_target = key
                break
    def transform(self, series):
        trans_series = super(PruneLabelEncoder, self).transform(series)
        # Transform all the low frequency keys into the low frequency target key
        for key, val in self.val_count_map.items():
            if val < self.cutoff:
                trans_series[trans_series==key] = self.low_cnt_target
        return trans_series


def whichcolumns(df, percentnull = 0.9):
# Returns a list of which columns have greater than or equal to 90% null values
    col = []
    for x in df.columns:
        if np.mean(pd.isnull(df[x])) >= percentnull:
            col += [x]
    return col


def dropcolumns(df, percentnull = 0.9):
# function to drop columns with 90% NaN values or user can specify percentage
    for x in df.columns:
        if np.mean(pd.isnull(df[x])) >= percentnull:
            df = df.drop(x,axis = 1)
    return df



def encode(df, columns, TRANSFORM_CUTOFF):
    '''
    Takes in a dataframe, columns of interest, and a cutoff value for bucketing
    encoding values

    If the frequency of an encoded value is below the cutoff, it will bucket
    everything to the first value it encounters that is below the cutoff value
    '''
    temp = df.copy()

    # Checking if there are 2 or more unique values in each column
    for x in columns:
        if len(df[x].unique()) < 2:
            return 'Error: Fewer than 2 unique values in a column'

    for col in columns:
        if type(df[col].unique()[1]) == str:
            le = PruneLabelEncoder()
            le.fit(df[col],TRANSFORM_CUTOFF)
            df[col] = le.transform(df[col])

    return df

def encode_force(df, columns, TRANSFORM_CUTOFF):
    '''
    takes in a dataframe, list of columns, and TRANSFORM_CUTOFF

    same as encode but it doesn't do the str check
    '''
    temp = df.copy()

    # Checking if there are 2 or more unique values in each column
    for x in columns:
        if len(df[x].unique()) < 2:
            return 'Error: Fewer than 2 unique values in a column'

    for col in columns:
        le = PruneLabelEncoder()
        le.fit(df[col],TRANSFORM_CUTOFF)
        df[col] = le.transform(df[col])

    return df
