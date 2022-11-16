# @Time: 2022.8.30 23:16
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)

import os

# * working directory
work_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(work_dir, 'data')
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')
src_dir = code_dir = os.path.join(work_dir, 'code')
model_dir = os.path.join(work_dir, 'model') # for saving models
res_dir = os.path.join(work_dir, 'result') # for submission.csv
tmp_dir = os.path.join(work_dir, 'tmp') # for saving pyg files

# * arch, compiler, optimize
archs_dict = {
    'x86_32': 0,
    'x86_64': 1,
    'mipsel_32': 2,
    'mipsel_64': 3,
    'arm_32': 4,
    'arm_64': 5,
}

compilers_dict = {
    'gcc': 0,
    'llvm': 1
}

optimizers_dict = {
    'O0': 0,
    'O1': 1,
    'O2': 2,
    'O3': 3,
    'Os': 4
}

# * seed
seed = 42

# * maximum basic block token length
# 99% of blocks have a length smaller than 130
# 98% of blocks have a length smaller than 96
# 95% of blocks have a length smaller than 64
max_block_len = 64

# * train / val ratio
train_ratio = 0.95
