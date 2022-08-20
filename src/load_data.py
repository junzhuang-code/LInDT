#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: Load dataset
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

from utils import preprocess_dgl_adj
import dgl.data

class LoadDataset():
    def __init__(self, data_name):
        self.data_name = data_name
        print("Current dataset: cora, citeseer, pubmed, amazoncobuy, coauthor, reddit.")
        print("Selecting {0} Dataset ...".format(self.data_name))

    def load_data(self):
        # Load dataset based on given data_name.
        if self.data_name == "cora":  # cora_v2
            dataset = dgl.data.CoraGraphDataset()
        if self.data_name == "citeseer":  # citeseer
            dataset = dgl.data.CiteseerGraphDataset()
        if self.data_name == "pubmed":  # pubmed
            dataset = dgl.data.PubmedGraphDataset()
        if self.data_name == "amazoncobuy":  # amazon_co_buy_photo
            dataset = dgl.data.AmazonCoBuyPhotoDataset()
        if self.data_name == "coauthor":  # coauthor_cs
            dataset = dgl.data.CoauthorCSDataset()
        if self.data_name == "reddit":  # reddit
            dataset = dgl.data.RedditDataset()
        # Load graph, feature matrix, and label.
        graph = dataset[0]
        graph = preprocess_dgl_adj(graph)  # preprocessing adj matrix
        print("Data is stored in: /Users/[user_name]/.dgl")
        print("{0} Dataset Loaded!".format(self.data_name))
        return graph

"""
  #Graph:    cora,  citeseer, pubmed, amazoncobuy, coauthor, reddit
  #Nodes:    2708,  3327,     19717,  7650,        18333     232965
  #Edges:    10556, 9228,     88651,  287326,      327576    114615892
  #Features: 1433,  3703,     500,    745,         6805      602
  #Classes:  7,     6,        3,      8,           15        41
"""
