# @Time: 2022.8.31 16:48
# @Author: Bolun Wu (e-mail: bowenwu@sjtu.edu.cn)


import pytorch_lightning as pl
import torch
import torchmetrics.functional as tmf
from pytorch_metric_learning import losses as L
from pytorch_metric_learning import miners
from torch import nn
from torch_geometric.nn import (AttentionalAggregation, GraphSAGE,
                                ResGatedGraphConv)
from torch_geometric.nn.models.basic_gnn import BasicGNN

from constants import *
from kernels import GatedGCN_EKernel


class BCSD_GNNModule(pl.LightningModule):
    def __init__(self, 
                 embedding_dims=128,
                 use_edge_attr=False,
                 seq_model='lstm',
                 lstm_hidden_dims=64,
                 lstm_layers=3,
                 gnn_model='gin',
                 gnn_hidden_dims=128,
                 gnn_out_dims=128,
                 gnn_layers=3,
                 dropout=0.5,
                 bn=False,
                 vocab_size=12532,
                 lr=1e-3,
                 milestones=[10],
                 miner_type='norm',
                 loss_type='marginloss'):
        super(BCSD_GNNModule, self).__init__()

        assert seq_model in ('lstm', 'gru')
        assert gnn_model in ('graphsage', 'gatedgcn', 'gatedgcn-e')
        assert loss_type in ('marginloss', 'tripletloss', 'multisimi')
        assert miner_type in ('distance', 'multisimi')

        # network
        self.use_edge_attr = use_edge_attr
        if use_edge_attr: self.edge_embedding = nn.Embedding(2, embedding_dims)
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dims)
        self.arch_embedding = nn.Embedding(len(archs_dict), embedding_dims)
        self.compiler_embedding = nn.Embedding(len(compilers_dict), embedding_dims)
        self.opti_embedding = nn.Embedding(len(optimizers_dict), embedding_dims)
        self.layer_norm = nn.LayerNorm(embedding_dims)

        self.seq_model = seq_model
        if seq_model == 'lstm':
            self.seq = nn.LSTM(embedding_dims, lstm_hidden_dims, lstm_layers, bidirectional=True, batch_first=True, dropout=dropout)
            gnn_in_dims = lstm_hidden_dims * 2
        elif seq_model == 'gru':
            self.seq = nn.GRU(embedding_dims, lstm_hidden_dims, lstm_layers, bidirectional=True, batch_first=True, dropout=dropout)
            gnn_in_dims = lstm_hidden_dims * 2
        batch_norm = nn.BatchNorm1d(gnn_hidden_dims) if bn else None

        if gnn_model == 'graphsage':
            self.gnn = GraphSAGE(in_channels=gnn_in_dims, hidden_channels=gnn_hidden_dims, num_layers=gnn_layers, 
                                 out_channels=gnn_out_dims, dropout=dropout, norm=batch_norm, jk='cat')
        elif gnn_model == 'gatedgcn':
            self.gnn = GatedGCN(in_channels=gnn_in_dims, hidden_channels=gnn_hidden_dims, num_layers=gnn_layers, 
                                out_channels=gnn_out_dims, dropout=dropout, norm=batch_norm, jk='cat')
        elif gnn_model == 'gatedgcn-e':
            self.gnn = GatedGCN_E(in_channels=gnn_in_dims, hidden_channels=gnn_hidden_dims, num_layers=gnn_layers, 
                                  out_channels=gnn_out_dims, dropout=dropout, norm=batch_norm, jk='cat', edge_dims=embedding_dims)

        self.pooling = AttentionalAggregation(nn.Linear(gnn_out_dims, 1))

        # sampler
        if miner_type == 'distance':
            print('using DistanceWeightedMiner.')
            self.miner = miners.DistanceWeightedMiner()
        elif miner_type == 'multisimi':
            print('using MultiSimilarityMiner.')
            self.miner = miners.MultiSimilarityMiner()

        # criterion
        if loss_type == 'marginloss':
            print('using MaginLoss.')
            self.criterion = L.MarginLoss()
        elif loss_type == 'tripletloss':
            print('using TripletLoss.')
            self.criterion = L.TripletMarginLoss(margin=0.5)
        elif loss_type == 'multisimi':
            print('using MultiSimilarityLoss.')
            self.criterion = L.MultiSimilarityLoss()

        # args
        self.lr = lr
        self.milestones = milestones

    def forward(self, data):

        # * lstm + gnn
        x, edge_index, batch = data.x.long(), data.edge_index.long(), data.batch
        arch, compiler, opti = data.arch_id.long(), data.compiler_id.long(), data.opti_id.long()
        
        arch = self.arch_embedding(arch)[batch].unsqueeze(1)
        compiler = self.compiler_embedding(compiler)[batch].unsqueeze(1)
        opti = self.opti_embedding(opti)[batch].unsqueeze(1)

        x = self.token_embedding(x)
        x = x + arch + compiler + opti
        x = self.layer_norm(x)

        x, _ = self.seq(x)
        x = torch.mean(x, dim=1)

        if self.use_edge_attr:
            edge_attr = data.edge_attr.long()
            edge_attr = self.edge_embedding(edge_attr)
            x = self.gnn(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.gnn(x, edge_index)
        x = self.pooling(x, batch)

        return x

    def training_step(self, data, *args):
        x = self(data)
        apn_indices_tuple = self.miner(x, data.gid)
        loss = self.criterion(embeddings=x, indices_tuple=apn_indices_tuple)
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))
        return {'loss': loss}

    def validation_step(self, data, *args):
        x = self(data)
        val_map = self.cal_map(x, data.gid)
        self.log_dict({'val_map': val_map}, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        
    def test_step(self, data, *args):
        x = self(data)
        test_map = self.cal_map(x, data.gid)
        self.log_dict({'test_map': test_map}, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.milestones is None:
            return [optimizer]
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
            return [optimizer], [scheduler]

    def cal_map(self, x, gid):
        """calculate map for validaiton and test stage
        since the val input is random, `gid` is given to indicate the source
        
        # ? how to convert cosine similarity to angular similarity
        # ? ref: https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
        
        Args:
            x (torch.tensor): shape (batch_size, repr_dim)
            gid (torch.tensor): shape (batch_size)

        Returns:
            float: MAP (mean average precision)
        """
        x, n = x.clone(), x.size(0)
        # ! the following operation will change x (destroy gradient map), therefore, it must be used on x.clone()
        cos_matrix = tmf.pairwise_cosine_similarity(x, x, zero_diagonal=True)
        simi_matrix = 1 - torch.arccos(cos_matrix) / torch.pi
        simi_matrix -= torch.diag_embed(simi_matrix.diag() - 1)
        
        _map = 0.0
        for i in range(n):
            simis = simi_matrix[i].tolist()
            simis.pop(i)
            
            labels = []
            for j in range(n):
                if i == j: continue
                if gid[j] == gid[i]: labels.append(1)
                else: labels.append(0)
            if sum(labels) == 0: continue
            
            _ap = tmf.retrieval_average_precision(torch.tensor(simis), torch.tensor(labels))
            _map += _ap
        _map /= n
        
        return _map


class GatedGCN(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = False
    def init_conv(self, in_channels: int, out_channels: int, **kwargs):
        return ResGatedGraphConv(in_channels, out_channels)


class GatedGCN_E(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True
    def init_conv(self, in_channels: int, out_channels: int, edge_dims: int, **kwargs):
        return GatedGCN_EKernel(in_channels, out_channels, edge_dims=edge_dims)

