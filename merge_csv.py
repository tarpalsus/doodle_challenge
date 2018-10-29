# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:26:13 2018

@author: kondrat
"""

import os
import pandas as pd

path = r"c"
 
data_all = pd.read_csv(os.path.join(path,os.listdir(path)[0]))



for file in os.listdir(path)[1:]:
    print('Currently processing {}'.format(file))
    data = pd.read_csv(os.path.join(path,file))
    data_all = pd.concat([data_all,data])
    
shuffled = data_all.sample(frac=1)
print('Saving file...')
shuffled.to_csv('all_10%.csv')