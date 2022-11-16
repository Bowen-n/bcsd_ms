# @Time: 2022.8.31 16:19
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)

import json
import os
import re
from collections import Counter

import tqdm

from constants import *
from utils import count_file_line


class InsNormalizer(object):
    def __init__(self):
        self.name = 'ins_normalizer'
    
    def normalize(self, instruction, arch):
        ins_id = instruction[0]
        ins = instruction[2].split('\t')
        
        if len(ins) == 1: return ins_id, ins[0], []
        elif len(ins) > 2: raise Exception('invalid instruction')

        opcode, operands = ins[0], ins[1]
        opcode, operands = self.__clean(opcode, operands)
        
        if 'x86' in arch:
            operands = self.__normalize_x86(operands)
        elif 'arm' in arch:
            operands = self.__normalize_arm(operands)
        elif 'mips' in arch:
            operands = self.__normalize_mips(operands)
    
        if 'arm' in arch:
            i = 0
            operands = list(operands)
            while i < len(operands) - 1:
                if operands[i]+operands[i+1] == ', ' and \
                '[' in operands[:i] and ']' in operands[i+2:]:
                    operands.pop(i+1)
                i += 1
            operands = ''.join(operands)
            
        operands = operands.split(', ')
        return ins_id, opcode, operands
        
    def __clean(self, opcode, operands):
        operands = operands.strip()
        operands = operands.replace('ptr ','')
        operands = operands.replace('offset ','')
        operands = operands.replace('xmmword ','')
        operands = operands.replace('dword ','')
        operands = operands.replace('qword ','')
        operands = operands.replace('word ','')
        operands = operands.replace('byte ','')
        operands = operands.replace('short ','')
        operands = operands.replace(' + ', '+')
        operands = operands.replace(' - ', '-')
        operands = operands.replace('!', '')
        return opcode, operands
        
    def __normalize_x86(self, operands):
        ## ? address, how to difference between immediates and addresses?
        # operands = re.sub(r'0x[0-9a-f]{5,7}', 'ADDRESS', operands)
        
        ## * immediate_value
        operands = re.sub(r'0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r'\+[0-9]', '+CONST', operands)
        operands = re.sub(r' [0-9]', ' CONST', operands)
        operands = re.sub(r' -[0-9]', ' -CONST', operands)
        operands = re.sub(r'^[0-9]', 'CONST', operands)
        
        ## * register
        base_pointer = r'(bp|ebp|rbp)'
        stack_pointer = r'(sp|esp|rsp)'
        program_counter = r'(ip|eip|rip)'
        operands = re.sub(base_pointer, 'reg_base_pointer', operands)
        operands = re.sub(stack_pointer, 'reg_stack_pointer', operands)
        operands = re.sub(program_counter, 'reg_pc', operands)
        
        return operands

    def __normalize_arm(self, operands):
        operands = operands.replace('{', '')
        operands = operands.replace('}', '')
        
        ## * immediate_value
        operands = re.sub(r'#0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r'#-0x[0-9a-f]+', '-CONST', operands)
        operands = re.sub(r'#[0-9]+', 'CONST', operands)
        operands = re.sub(r'#-[0-9]+', '-CONST', operands)
        
        ## * register
        reg_gen_r = r'r[0-9]+'
        base_pointer = r'fp'
        stack_pointer = r'sp'
        program_counter = r'pc'
        
        operands = re.sub(base_pointer, 'reg_base_pointer', operands)
        operands = re.sub(stack_pointer, 'reg_stack_pointer', operands)
        operands = re.sub(program_counter, 'reg_pc', operands)
        
        def __map_r_register(x):
            start, end = x.span()
            return 'arm_' + operands[start:end]
        
        operands = re.sub(reg_gen_r, __map_r_register, operands)
        
        return operands
    
    def __normalize_mips(self, operands):
        ## * immediate_value
        operands = re.sub(r'-0x[0-9a-f]+', '-CONST', operands)
        operands = re.sub(r'0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r' [0-9]', ' CONST', operands)
        operands = re.sub(r' -[0-9]', ' -CONST', operands)
        
        ## * register
        base_pointer = r'$fp'
        stack_pointer = r'$sp'        
        operands = re.sub(base_pointer, 'reg_base_pointer', operands)
        operands = re.sub(stack_pointer, 'reg_stack_pointer', operands)
        return operands
    
    
def normalize_instruction(instruction, arch):
    ins_id = instruction[0]
    ins = instruction[2].split('\t')
    
    if len(ins) == 1: return ins_id, ins[0], []
    elif len(ins) > 2: raise Exception('invalid instruction')

    opcode, operands = ins[0], ins[1]
    operands = operands.strip()
    operands = operands.replace('ptr ','')
    operands = operands.replace('offset ','')
    operands = operands.replace('xmmword ','')
    operands = operands.replace('dword ','')
    operands = operands.replace('qword ','')
    operands = operands.replace('word ','')
    operands = operands.replace('byte ','')
    operands = operands.replace('short ','')
    operands = operands.replace(' - ','+')
    operands = operands.replace(' + ', '+')
    operands = operands.replace('!', '')
    
    # arm
    if arch.startswith('arm'):
        operands = re.sub(r'#0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r'#-0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r'#[0-9]+', 'CONST', operands)
        operands = re.sub(r'#-[0-9]+', 'CONST', operands)
    
    # mips
    if arch.startswith('mips'):
        operands = re.sub(r'-0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r'0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r' [0-9]', ' CONST', operands)
        operands = re.sub(r' -[0-9]', ' CONST', operands)
        
    # x86
    if arch.startswith('x86'):
        operands = re.sub(r'0x[0-9a-f]+', 'CONST', operands)
        operands = re.sub(r'\+[0-9]', '+CONST', operands)
        operands = re.sub(r' [0-9]', ' CONST', operands)
        operands = re.sub(r' -[0-9]', ' CONST', operands)
        operands = re.sub(r'^[0-9]', 'CONST', operands)
        
    if arch.startswith('arm'):
        i = 0
        operands = list(operands)
        while i < len(operands) - 1:
            if operands[i]+operands[i+1] == ', ' and \
            '[' in operands[:i] and ']' in operands[i+2:]:
                operands.pop(i+1)
            i += 1
        operands = ''.join(operands)    
        operands = operands.split(', ')     
    else:
        operands = operands.split(', ')

    return ins_id, opcode, operands
    

if __name__ == '__main__':
    vocab_dir = os.path.join(model_dir, 'vocab')
    os.makedirs(vocab_dir, exist_ok=True)
    
    freq = 3
    vocab_counter_path = os.path.join(vocab_dir, 'vocab_counter.json')
    ins_normalizer = InsNormalizer()
    
    if not os.path.exists(vocab_counter_path):
        vocab_counter = Counter()
        train_fp = os.path.join(train_data_dir, 'train.func.json')
        with open(train_fp, 'r') as f:
            for line in tqdm.tqdm(f, total=count_file_line(train_fp), desc='building vocabulary'):
                data = json.loads(line)
                arch, code = data['arch'], data['code']
                
                norm_insns = []   
                for ins in code:
                    ins_id, opcode, operands = ins_normalizer.normalize(ins, arch)
                    vocab_counter[opcode] += 1
                    for operand in operands: vocab_counter[operand] += 1

        print(f'entire vocab size: {len(vocab_counter)}.')
        with open(vocab_counter_path, 'w') as f:
            json.dump(vocab_counter, f, indent=1)

    else:
        with open(vocab_counter_path, 'r') as f:
            vocab_counter = json.load(f)
        print(f'entire vocab size: {len(vocab_counter)}.')

    vocab = [[k, v] for k, v in vocab_counter.items() if v >= freq]
    vocab = sorted(vocab, key=lambda x: x[1], reverse=True)
    
    vocab_freq = {v: f for (v, f) in vocab}
    with open(os.path.join(vocab_dir, 'vocab_freq.json'), 'w') as f:
        json.dump(vocab_freq, f, indent=1)
        
    vocab = [x[0] for x in vocab]
    vocab = ['<unk>', '<pad>'] + vocab
    vocab = {v: i for i, v in enumerate(vocab)}
    
    print(f'cleaned vocab size: {len(vocab)}')
    
    with open(os.path.join(vocab_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=1)
        
