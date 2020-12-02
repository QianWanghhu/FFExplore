"""Apply experiment with Morris."""

import pandas as pd
import numpy as np
import json
from utils.group_fix import index_fix
from settings import MORRIS_DATA_DIR

# cache_file = f'{MORRIS_DATA_DIR}morris_1010.json'
cache_file = f'../output/morris/morris_1010.json'
partial_key = 'result_80'; file_exist = True
with open(cache_file, 'r') as fp: partial_order = json.load(fp)
ind_fix_dict = {}; 
combs_accumulate = {str(i) : 0 for i in range(1, 22)}

for key, value in partial_order.items():
    print(key)
    ind_fix = []
    num_group = len(value) - 1
    i = num_group
    for i in np.arange(num_group, -1, -1):
        print(i)        
        ind_fix = index_fix(value, i, file_exist, ind_fix) 
        num_fix = str(len(ind_fix))
        if ind_fix not in list(ind_fix_dict.values()):
            
            ind_fix_dict[num_fix] = ind_fix
        else:
            combs_accumulate[num_fix] = combs_accumulate[num_fix] + 1

partial_stable = {'0': [i for i in range(15, 21)], 
                '1': [i for i in range(11, 15)], 
                '2': [i for i in range(0, 11)]}
with open('../output/morris/morris_stable.json', 'w') as fp:
    json.dump(partial_stable, fp,indent=2)