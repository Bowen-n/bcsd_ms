# @Time: 2022.9.2 16:38
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)

import argparse
import json
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.loader import DataLoader

from constants import *
from dataset import BinaryCFGwEStaTestDataset
from models import BCSD_GNNModule
from utils import count_file_line

if __name__ == '__main__':
    # init
    torch.set_grad_enabled(False)
    pl.seed_everything(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str, help='model directory')
    args = parser.parse_args()
    args.result_dir = os.path.expanduser(args.result_dir)

    with open(os.path.join(args.result_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    test_repr_name = 'test_repr.pickle'
    test_repr_path = os.path.join(tmp_dir, 'test', test_repr_name)

    if 'use_edge_attr' in config and config['use_edge_attr'] == True: 
        use_edge_attr = True
    else:
        use_edge_attr = False

    if not os.path.exists(test_repr_path):
        model_dir = os.path.join(args.result_dir, 'checkpoints')
        model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
        model = BCSD_GNNModule(embedding_dims=config['embedding_dims'],
                               use_edge_attr=use_edge_attr,
                               seq_model=config['seq_model'],
                               lstm_hidden_dims=config['lstm_hidden_dims'],
                               lstm_layers=config['lstm_layers'],
                               gnn_model=config['gnn_model'],
                               gnn_hidden_dims=config['gnn_hidden_dims'],
                               gnn_out_dims=config['gnn_out_dims'],
                               gnn_layers=config['gnn_layers'],
                               bn=config['batch_norm'],
                               miner_type=config['miner_type'],
                               loss_type=config['loss_type'])
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        model = model.to(device)
        model.eval()
        print(model)
        
        test_set = BinaryCFGwEStaTestDataset(root=test_data_dir, norm_size=-1, vocab_dir=config['vocab_dir'])
        test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8)
        print(f'Test size: {len(test_set)}.')

        fid_to_repr = {}
        for data in tqdm.tqdm(test_loader):
            data = data.to(device)
            graph_repr = model(data)
            
            fids = data.fid.cpu().numpy()
            graph_repr = graph_repr.cpu().numpy()
            for fid, graph_repr in zip(fids, graph_repr):
                fid_to_repr[fid] = graph_repr
            
        with open(test_repr_path, 'wb') as f:
            pickle.dump(fid_to_repr, f)
        
    else:

        with open(test_repr_path, 'rb') as f:
            fid_to_repr = pickle.load(f)

    sub_fname = 'submission.csv'
    sub_f = open(os.path.join(res_dir, sub_fname), 'w')
    sub_f.write('fid,fid0:sim0,fid1:sim1,fid2:sim2,fid3:sim3,fid4:sim4,fid5:sim5,fid6:sim6,fid7:sim7,fid8:sim8,fid9:sim9\n')

    ques_f = open(os.path.join(test_data_dir, 'test.question.csv'), 'r')
    for line in tqdm.tqdm(ques_f, total=count_file_line(os.path.join(test_data_dir, 'test.question.csv'))):
        line = line.split(',')
        line = list(map(lambda x: int(x), line))
        src_fid, tar_fids = line[0], line[1:]

        src_repr = fid_to_repr[src_fid].reshape(1, -1)
        tar_reprs = np.stack([fid_to_repr[fid] for fid in tar_fids])

        cosine_simi = cosine_similarity(src_repr, tar_reprs)
        simi = 1 - np.arccos(cosine_simi) / np.pi
        simi = simi[0].tolist()

        merge = [[_fid, _simi] for _fid, _simi in zip(tar_fids, simi)]
        merge.sort(key=lambda x: x[1], reverse=True)
        top_10 = merge[:10]
        
        sub_f.write(str(src_fid))
        for _top in top_10:
            sub_f.write(f',{_top[0]}:{_top[1]:.4f}')
        sub_f.write('\n')

    ques_f.close()
    sub_f.close()
