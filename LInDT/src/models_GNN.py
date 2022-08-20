#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: GNN Models
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

from dgl.nn.pytorch import GraphConv, SGConv, SAGEConv, GATConv
import torch


# GNNs model --------------------
class GNN(torch.nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 model_name,
                 **kwargs):
        super(GNN, self).__init__()
        self.g = g  # graph DGLGraph
        self.layers = torch.nn.ModuleList()
        self.aggregator_type = kwargs["aggregator_type"]
        self.n_filter = kwargs["n_filter"]
        self.n_heads = kwargs["n_heads"]
        self.dropout = dropout
        # Select the model layer
        if model_name == "GCN":
            model_in = GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True)
            model_h = GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True)
            model_out = GraphConv(n_hidden, n_classes, allow_zero_in_degree=True)
            self.w0, self.w1 = model_in.weight, model_out.weight
        elif model_name == "SGC":  # k for the size of filter
            model_in = SGConv(in_feats, n_hidden, k=self.n_filter, allow_zero_in_degree=True)
            model_h = SGConv(n_hidden, n_hidden, k=self.n_filter, allow_zero_in_degree=True)
            model_out = SGConv(n_hidden, n_classes, k=self.n_filter, allow_zero_in_degree=True)
            self.w0, self.w1 = \
                torch.transpose(model_in.fc.weight, 0, 1), torch.transpose(model_out.fc.weight, 0, 1)
        elif model_name == "GraphSAGE":  # Aggregator type: mean, gcn, pool, lstm.
            model_in = SAGEConv(in_feats, n_hidden, self.aggregator_type, activation=activation)
            model_h = SAGEConv(n_hidden, n_hidden, self.aggregator_type, activation=activation)
            model_out = SAGEConv(n_hidden, n_classes, self.aggregator_type)
            self.w0, self.w1 = \
                torch.transpose(model_in.fc_neigh.weight, 0, 1), torch.transpose(model_out.fc_neigh.weight, 0, 1)
        elif model_name == "GAT":  # num_heads: Number of heads in Multi-Head Attention
            model_in = GATConv(in_feats, n_hidden, num_heads=self.n_heads, activation=activation)
            model_h = GATConv(n_hidden * self.n_heads, n_hidden, num_heads=self.n_heads, activation=activation)
            model_out = GATConv(n_hidden * self.n_heads, n_classes, num_heads=1)
            self.w0, self.w1 = \
                torch.transpose(model_in.fc.weight, 0, 1), torch.transpose(model_out.fc.weight, 0, 1)
        else:
            print("model_name is incorrect!")
            return 0
        # Build the model
        self.layers.append(model_in)  # input layer
        for _ in range(n_layers - 1):
            self.layers.append(torch.nn.Dropout(p=self.dropout))
            self.layers.append(model_h)  # hidden layers
        self.layers.append(torch.nn.Dropout(p=self.dropout))
        self.layers.append(model_out)  # output layer

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            if type(layer) == torch.nn.modules.dropout.Dropout:
                h = layer(h)
            else:
                h = layer(self.g, h)
                # for GAT (ref: https://modulai.io/graph-neural-networks/)
                if len(h.shape) > 2:  # concat the last 2 dim if h_dim > 2
                    h = h.view(-1, h.size(1) * h.size(2))
        return h
