# @Time: 2022.8.30 23:16
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)

import os

from constants import *


def count_file_line(path):
    res = os.popen(f'wc -l {path}').readlines()
    if res == []: 
        raise Exception('file line not countable')
    try:
        num_line = int(res[0].split()[0])
        return num_line
    except:
        raise Exception('file line not countable')
    

def count_code_line():
    code_line = 0
    for root, _, files in os.walk(work_dir):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    code_line += len(f.readlines())
    return code_line
    
