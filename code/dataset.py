# @Time: 2022.9.1 16:35
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)

import argparse
import copy
import json
import os
import random

import torch
import tqdm
from torch_geometric.data import Data, InMemoryDataset

from constants import *
from corpus import InsNormalizer
from utils import count_file_line


class BaseDataset(InMemoryDataset):
    def __init__(self, root, mode, vocab_dir, norm_size=-1):
        assert mode in ('train', 'val', 'test')
        assert norm_size == -1 or norm_size > 10
        self.mode = mode
        self.norm_size = norm_size
        self.vocab_dir = vocab_dir
        super(BaseDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def norm(self):
        return True if self.norm_size != -1 else False

    @property
    def vocab_dir_name(self):
        return os.path.basename(self.vocab_dir.strip('/'))

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def select_gids(self, gids):
        random.seed(seed)
        random.shuffle(gids)
        divide = int(train_ratio * len(gids))
        if self.mode == 'train': return gids[:divide]
        elif self.mode == 'val': return gids[divide:]
    
    def get_group(self):
        group_fp = os.path.join(train_data_dir, 'train.group.csv')
        gid_to_fids = {}
        with open(group_fp, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                line = list(map(lambda x: int(x), line))
                gid_to_fids[line[0]] = line[1:]
        return gid_to_fids
    
    def get_vocab(self):
        with open(os.path.join(self.vocab_dir, 'vocab.json'), 'r') as f:
            vocab = json.load(f)
        return vocab
            
    def get_gids(self):
        return self.select_gids(list(self.get_group().keys()))

    def fid_to_gid(self):
        gid_to_fids = self.get_group()
        gids = self.select_gids(list(gid_to_fids.keys()))
        
        fid_to_gid = {}
        for gid in gids:
            for fid in gid_to_fids[gid]:
                fid_to_gid[fid] = gid
        return fid_to_gid
    
    def process(self):
        raise NotImplementedError


class BinaryCFGwEStaTrainDataset(BaseDataset):
    """Binary CFG with edge attributes and statistical features"""
    def __init__(self, root, mode, vocab_dir, norm_size=-1):
        super(BinaryCFGwEStaTrainDataset, self).__init__(root, mode, vocab_dir, norm_size)
        sta = torch.load(self.processed_paths[1])
        self.data.arch_id = sta[:, 0].long()
        self.data.compiler_id = sta[:, 1].long()
        self.data.opti_id = sta[:, 2].long()
        self.slices['arch_id'] = copy.deepcopy(self.slices['gid'])
        self.slices['compiler_id'] = copy.deepcopy(self.slices['gid'])
        self.slices['opti_id'] = copy.deepcopy(self.slices['gid'])

    @property
    def processed_dir(self):
        return os.path.join(os.path.join(tmp_dir, 'train'))
    
    @property
    def processed_file_names(self):
        if self.norm: fname = f'binary_cfg.e.{self.vocab_dir_name}.norm{self.norm_size}.{self.mode}.pt'
        else: fname = f'binary_cfg.e.{self.vocab_dir_name}.{self.mode}.pt'
        return [fname, f'sta.{self.mode}.pt']

    def process_sta(self):
        fid_to_gid = self.fid_to_gid()
        train_fp = os.path.join(train_data_dir, 'train.func.json')
        f = open(train_fp, 'r')
        datas = []
        for line in tqdm.tqdm(f, total=count_file_line(train_fp)):
            data = json.loads(line)
            fid, arch, compiler, opti = data['fid'], data['arch'], data['compiler'], data['opti']
            if fid not in fid_to_gid: continue
            
            ## * arch, compiler, opti
            arch_id = archs_dict[arch]
            compiler_id = compilers_dict[compiler]
            opti_id = optimizers_dict[opti]
            
            datas.append([arch_id, compiler_id, opti_id])
        
        data = torch.tensor(datas, dtype=torch.float)
        f.close()
        torch.save(data, self.processed_paths[1])
            
    def process(self):
        vocab = self.get_vocab()
            
        fid_to_gid = self.fid_to_gid()
        
        train_fp = os.path.join(train_data_dir, 'train.func.json')
        f = open(train_fp, 'r')
        
        graph_list, normalizer = [], InsNormalizer()
        for line in tqdm.tqdm(f, total=count_file_line(train_fp)):
            data = json.loads(line)
            fid, arch = data['fid'], data['arch']
            if fid not in fid_to_gid: continue
            
            code, blocks, cfg = data['code'], data['block'], data['cfg']            
            if self.norm:
                blocks = blocks[:self.norm_size]
                num_blocks = len(blocks)
                cfg = list(filter(lambda x: x[0]<num_blocks and x[1]<num_blocks, cfg))
                
            norm_insns = []
            for ins in code:
                ins_id, opcode, operands = normalizer.normalize(ins, arch)
                norm_insns.append([ins_id, [opcode]+operands])
            
            ## * x
            x, i = [], 0
            for block in blocks:
                block_insns = []
                _, start_addr, end_addr = block
                while i < len(norm_insns):
                    if start_addr <= norm_insns[i][0] < end_addr:
                        block_insns.extend(norm_insns[i][1])
                        i += 1
                    else:
                        break
                
                # truncate
                block_insns = block_insns[:max_block_len]
                # to token id
                _map_f = lambda x: vocab[x] if x in vocab else vocab['<unk>']
                block_insns = list(map(_map_f, block_insns))
                # padding
                block_insns = block_insns + [vocab['<pad>'] for _ in range(max_block_len - len(block_insns))]
                x.append(block_insns)

            ## * edge_index
            edge_index = cfg

            ## * edge_attr
            edge_attr = []
            for e in edge_index:
                if e[0] + 1 == e[1]: edge_attr.append(0) # seq exec
                else: edge_attr.append(1) # jump exec

            ## * gid
            gid = fid_to_gid[fid]

            graph = Data(x=torch.tensor(x, dtype=torch.int16),
                         edge_index=torch.tensor(edge_index, dtype=torch.int32).t(),
                         edge_attr=torch.tensor(edge_attr, dtype=torch.int8),
                         gid=torch.tensor(gid, dtype=torch.int32))

            graph_list.append(graph)

        f.close()
        torch.save(self.collate(graph_list), self.processed_paths[0])
        del graph_list

        # * sta
        if not os.path.exists(self.processed_paths[1]):
            self.process_sta()


class BinaryCFGwEStaTestDataset(BaseDataset):
    def __init__(self, root, vocab_dir, norm_size=-1):
        super(BinaryCFGwEStaTestDataset, self).__init__(root, mode='test', vocab_dir=vocab_dir, norm_size=norm_size)
        sta = torch.load(self.processed_paths[1])
        self.data.arch_id = sta[:, 0].long()
        self.data.compiler_id = sta[:, 1].long()
        self.data.opti_id = sta[:, 2].long()
        self.slices['arch_id'] = copy.deepcopy(self.slices['fid'])
        self.slices['compiler_id'] = copy.deepcopy(self.slices['fid'])
        self.slices['opti_id'] = copy.deepcopy(self.slices['fid'])

    @property
    def processed_dir(self):
        return os.path.join(os.path.join(tmp_dir, 'test'))

    @property
    def processed_file_names(self):
        if self.norm: fname = f'binary_cfg.e.{self.vocab_dir_name}.norm{self.norm_size}.test.pt'
        else: fname = f'binary_cfg.e.{self.vocab_dir_name}.test.pt'
        return [fname, 'sta.test.pt']

    def process_sta(self):
        test_fp = os.path.join(test_data_dir, 'test.func.json')
        f = open(test_fp, 'r')
        datas = []
        for line in tqdm.tqdm(f, total=count_file_line(test_fp)):
            data = json.loads(line)
            arch, compiler, opti = data['arch'], data['compiler'], data['opti']
            
            ## * arch, compiler, opti
            arch_id = archs_dict[arch]
            compiler_id = compilers_dict[compiler]
            opti_id = optimizers_dict[opti]
            
            datas.append([arch_id, compiler_id, opti_id])
        
        data = torch.tensor(datas, dtype=torch.float)
        f.close()
        torch.save(data, self.processed_paths[1])
        
    def process(self):
        vocab = self.get_vocab()

        graph_list = []
        test_fp = os.path.join(test_data_dir, 'test.func.json')
        f = open(test_fp, 'r')
        
        normalizer = InsNormalizer()
        for line in tqdm.tqdm(f, total=count_file_line(test_fp)):
            data = json.loads(line)
            fid, arch = data['fid'], data['arch']
            code, blocks, cfg = data['code'], data['block'], data['cfg']
            
            if self.norm:
                blocks = blocks[:self.norm_size]
                num_blocks = len(blocks)
                cfg = list(filter(lambda x: x[0]<num_blocks and x[1]<num_blocks, cfg))
                
            norm_insns = []
            for ins in code:
                ins_id, opcode, operands = normalizer.normalize(ins, arch)
                norm_insns.append([ins_id, [opcode]+operands])
            
            ## * x
            x, i = [], 0
            for block in blocks:
                block_insns = []
                _, start_addr, end_addr = block
                while i < len(norm_insns):
                    if start_addr <= norm_insns[i][0] < end_addr:
                        block_insns.extend(norm_insns[i][1])
                        i += 1
                    else:
                        break
                
                # truncate
                block_insns = block_insns[:max_block_len]
                # to token id
                _map_f = lambda x: vocab[x] if x in vocab else vocab['<unk>']
                block_insns = list(map(_map_f, block_insns))
                # padding
                block_insns = block_insns + [vocab['<pad>'] for _ in range(max_block_len - len(block_insns))]
                x.append(block_insns)
            
            ## * edge_index
            edge_index = cfg

            ## * edge_attr
            edge_attr = []
            for e in edge_index:
                if e[0] + 1 == e[1]: edge_attr.append(0) # seq exec
                else: edge_attr.append(1) # jump exec
            
            graph = Data(x=torch.tensor(x, dtype=torch.int32),
                         edge_index=torch.tensor(edge_index, dtype=torch.int32).t(),
                         edge_attr=torch.tensor(edge_attr, dtype=torch.int8),
                         fid=torch.tensor(fid, dtype=torch.int32))

            graph_list.append(graph)
            
        f.close()
        torch.save(self.collate(graph_list), self.processed_paths[0])
        del graph_list
        
        # * sta
        if not os.path.exists(self.processed_paths[1]):
            self.process_sta()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_dir', type=str)
    parser.add_argument('--norm_size', type=int, default=96)
    parser.add_argument('--mode', type=str, choices=('train', 'val', 'test'))
    args = parser.parse_args()

    if args.mode == 'train':
        dataset = BinaryCFGwEStaTrainDataset(root=train_data_dir, norm_size=args.norm_size, vocab_dir=args.vocab_dir, mode='train')
    elif args.mode == 'val':
        dataset = BinaryCFGwEStaTrainDataset(root=train_data_dir, norm_size=args.norm_size, vocab_dir=args.vocab_dir, mode='val')
    elif args.mode == 'test':
        dataset = BinaryCFGwEStaTestDataset(root=test_data_dir, norm_size=-1, vocab_dir=args.vocab_dir)

