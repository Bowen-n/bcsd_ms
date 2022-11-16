# @Time: 2022.9.2 16:38
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)

import argparse
import json
import os
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import samplers
from torch_geometric.loader import DataLoader

from constants import *
from dataset import BinaryCFGwEStaTrainDataset
from models import BCSD_GNNModule

if __name__ == '__main__':
    # seed
    pl.seed_everything(seed)

    # args
    parser = argparse.ArgumentParser()
    ## args for dataset
    parser.add_argument('--norm_size', type=int, default=96,
                        help='maximum number of node in each graph')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab/v2',
                        help='vocab directory which contains vocab.json')
    
    ## args for model
    parser.add_argument('--embedding_dims', type=int, default=128,
                        help='embedding dims for all embedding layer')
    parser.add_argument('--use_edge_attr', action='store_true', default=True,
                        help='whether to use edge attributes.')
    parser.add_argument('--seq_model', type=str, choices=('lstm', 'gru'), default='lstm',
                        help='the sequentual model for computing block embeddings')
    parser.add_argument('--lstm_hidden_dims', type=int, default=128,
                        help='lstm hidden dims')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='number of lstm layers')
    parser.add_argument('--gnn_model', type=str, choices=('graphsage', 'gatedgcn', 'gatedgcn-e'), default='gatedgcn-e',
                        help='the gnn model to use')
    parser.add_argument('--gnn_hidden_dims', type=int, default=128,
                        help='gnn hidden dims')
    parser.add_argument('--gnn_layers', type=int, default=3,
                        help='number of gnn layers')
    parser.add_argument('--gnn_out_dims', type=int, default=128,
                        help='gnn output dims')
    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='use BatchNorm1d in GNN')
    
    ## args for training
    parser.add_argument('--train_batch_size', type=int, default=84,
                        help='training batch size, must be times of batch_k')
    parser.add_argument('--batch_k', type=int, default=4,
                        help='the number of each class in each batch')
    parser.add_argument('--val_batch_size', type=int, default=84,
                        help='validation batch size, must be times of 3.')
    parser.add_argument('--train_num_each_epoch', type=int, default=400000,
                        help='training samples for each epoch')
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloader')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--milestones', type=int, nargs='+', default=[40,60],
                        help='milestones for MultiStepLR, -1 for constant learning rate')
    parser.add_argument('--miner_type', type=str, choices=('distance', 'multisimi'), default='multisimi',
                        help='type of miner(sampler), distance for DistanceWeightedMiner, multisimi for MultiSimilarityMiner')
    parser.add_argument('--loss_type', type=str, default='multisimi', choices=('marginloss', 'tripletloss', 'multisimi'),
                        help='loss function to use')
    parser.add_argument('--early_stopping', type=int, default=50,
                        help='early stopping patience')
    parser.add_argument('--precision', type=int, default=32, choices=(16, 32),
                        help='precision (16,32)')
    
    ## util args
    parser.add_argument('--save_name', type=str, default='lstm_gatedgcn-e',
                        help='model save directory name')
    args = parser.parse_args()
    ## check args
    assert args.train_batch_size % args.batch_k == 0 and args.val_batch_size % 3 == 0
    if -1 in args.milestones: args.milestones = None
    print(args)
    
    # dataset
    print('Loading dataset...')
    since = time.time()
    train_set = BinaryCFGwEStaTrainDataset(root=data_dir, mode='train', norm_size=args.norm_size, vocab_dir=args.vocab_dir)
    val_set = BinaryCFGwEStaTrainDataset(root=data_dir, mode='val', norm_size=args.norm_size, vocab_dir=args.vocab_dir)
    print(f'Train size: {len(train_set)}. Val size: {len(val_set)}. Time: {time.time()-since}s.')

    # sampler
    print('Initialing sampler...')
    since = time.time()
    train_sampler = samplers.MPerClassSampler(train_set.data.gid, args.batch_k, args.train_batch_size, length_before_new_iter=args.train_num_each_epoch)
    val_sampler = samplers.FixedSetOfTriplets(val_set.data.gid, num_triplets=int(args.train_num_each_epoch*0.05))
    print(f'Sampler loading time: {time.time()-since}s.')

    # dataloader
    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=val_set, sampler=val_sampler, batch_size=args.val_batch_size, num_workers=args.num_workers)

    # model
    model = BCSD_GNNModule(embedding_dims=args.embedding_dims,
                           use_edge_attr=args.use_edge_attr,
                           seq_model=args.seq_model,
                           lstm_hidden_dims=args.lstm_hidden_dims,
                           lstm_layers=args.lstm_layers,
                           gnn_model=args.gnn_model,
                           gnn_hidden_dims=args.gnn_hidden_dims,
                           gnn_out_dims=args.gnn_out_dims,
                           gnn_layers=args.gnn_layers,
                           bn=args.batch_norm,
                           vocab_size=len(train_set.get_vocab()),
                           lr=args.learning_rate,
                           milestones=args.milestones,
                           miner_type=args.miner_type,
                           loss_type=args.loss_type)

    # callbacks
    ckpt_callback = ModelCheckpoint(monitor='val_map', mode='max', save_weights_only=True, filename='{val_map:.4f}_{epoch}_{step}')
    early_stopping_callback = EarlyStopping(monitor='val_map', patience=args.early_stopping, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(save_dir=model_dir, name=args.save_name)

    # save config
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(os.path.join(logger.log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)

    # trainer
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         accelerator='gpu', devices=[0],
                         log_every_n_steps=200, logger=logger,
                         callbacks=[ckpt_callback, lr_monitor, early_stopping_callback],
                         precision=args.precision)
    # train
    trainer.fit(model, train_loader, val_loader)
    
    # save val result
    val_result = trainer.test(model, val_loader, ckpt_path='best')[0]
    with open(os.path.join(logger.log_dir, 'best_val_result.json'), 'w') as f:
        json.dump(val_result, f, indent=1)

