# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:26:13 2018

@author: kondrat
"""

import os
import pandas as pd
NCSVS = 100

path = r"C:\Users\user\Desktop\doodle\c"
 
data_all = pd.read_csv(os.path.join(path,os.listdir(path)[0]))



for file in os.listdir(path)[1:]:
    print('Currently processing {}'.format(file))
    data = pd.read_csv(os.path.join(path,file))
    data_all = pd.concat([data_all,data])
    
shuffled = data_all.sample(frac=1)

batch_size = int(len(shuffled)/NCSVS) - 1

for k in range(NCSVS):
    print('Saving to file {}...'.format(k))
    subset = shuffled.iloc[k * batch_size: (k+1) * batch_size, :]
    subset.to_csv('all_10%_k{}.csv'.format(k))
    del subset