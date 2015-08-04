'''
__file__
    testing.py

__description__
    This file is to used to generate all the features in a dataframe
    The dataframe that's created: traintest

    After encoding, you can split the dataframe ...


'''

import os
from dataclean import *

cmd = "python fulldatamerge.py"
os.system(cmd)
print traintest.shape
print 'Done'
