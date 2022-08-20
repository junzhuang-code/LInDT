#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: Simulate non-malicious perturbations (random connections)
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

import time
import argparse
import numpy as np
import dgl
import torch
import torch.nn.functional as F
import scipy.sparse as ss
from models_GNN import GNN
from load_data import LoadDataset
from utils import read_yaml_file, select_target_nodes, dump_pickle, split_masks
from train_GNN import evaluation, prediction


# Define the arguments
parser = argparse.ArgumentParser(description="Read arguments for training.")
parser.add_argument("--config_file", type=str, default="perturbations", help="The name of config files.")
parser.add_argument("--data_name", type=str, default="cora", help="The name of dataset.",
                    choices=["cora", "citeseer", "pubmed", "amazoncobuy", "coauthor", "reddit"])
parser.add_argument("--model_name", type=str, default="GCN", help="The name of GNNs.")  # GCN, SGC, GraphSAGE, GAT
parser.add_argument("--TARGET_CLASS", type=int, default=-1, help="targeted attack: class_id; non-targeted attack: -1.")
parser.add_argument("--SAMPLE_RATE", type=float, default=0.1, help="The percentage of test nodes being attacked.")
parser.add_argument("--GPU", type=int, default=-1, help="Input GPU device id or -1.")
parser.add_argument("--use_config_file", type=bool, default=True, help="Pass arguments from config files or not.")
args = parser.parse_known_args()[0]

# Read config files
config_file = read_yaml_file("../config", args.config_file)
if config_file and args.use_config_file:
    pert_config = config_file["rdmPert"]
    args.data_name = pert_config["data_name"]
    args.model_name = pert_config["model_name"]
    args.TARGET_CLASS = pert_config["TARGET_CLASS"]
    args.SAMPLE_RATE = pert_config["SAMPLE_RATE"]
    args.GPU = pert_config["GPU"]

# The paramaters for random perturbations
PERT_RATE = 0.01  # num_perturbators cannot exceed this ratio
assert args.SAMPLE_RATE + PERT_RATE <= 1.0
dirs_attack = '../data/attacker_data/'
CUT_RATE = [0.1, 0.2]  # the split ratio of train/validation mask.
# The paramaters for node classifier
LR = 0.001
N_LAYERS = 2
N_HIDDEN = 200
DROPOUT = 0
WEIGHT_DECAY = 0
kwargs_dicts = {"GCN": [None, None, None],  # aggregator_type, n_filter, n_heads.
                "GraphSAGE": ["gcn", None, None],
                "SGC": [None, 2, None],
                "GAT": [None, None, 3]}


def random_connections(graph, test_mask, SAMPLE_RATE, PERT_RATE, TARGET_CLASS):
    """
    @topic: Simulate non-malicious perturbations (random connections) on graphs.
    @input:
        graph (dgl.graph): original graph;
        test_mask (1D tensor): the test mask;
        SAMPLE_RATE (float): the sampling ratio of the target (victim) nodes;
        PERT_RATE (float): the number of perturbators cannot exceed this ratio;
        TARGET_CLASS (int): the target class (negative int for random seed).
    @return:
        graph (dgl.graph): perturbed graph;
        victims_nodes_list (list): the list of target (victim) nodes;
        target_mask (1D tensor): the target mask.            
    """
    t0 = time.time()
    adj = graph.adjacency_matrix(scipy_fmt="csr").tolil()  # convert csr to lil
    feat = graph.ndata['feat']
    label = graph.ndata['label']
    # Step1: Select nodes for both perturbators (first num_perturbators nodes) and victims.
    nodes_list, target_mask = select_target_nodes(label, test_mask, \
                                   SAMPLE_RATE*(1+PERT_RATE), TARGET_CLASS)
    num_perturbators = int(len(label[test_mask])*SAMPLE_RATE*PERT_RATE)
    perturbators_nodes_list = nodes_list[:num_perturbators]
    victims_nodes_list = nodes_list[num_perturbators:]
    target_mask[perturbators_nodes_list] = False  # update the target_mask for victims
    # Step2: Apply random connections to victims nodes.
    node_degrees = adj.sum(axis=0).A1  # the degree of each node
    avg_degree_target = np.mean(node_degrees[target_mask])
    num_connect = int(1/PERT_RATE)  # the connection budget of the perturbator
    pert_vict_idx = np.random.choice(victims_nodes_list, \
                     (num_perturbators, num_connect), replace=False) # the index of victims for each perturbator
    for r, p_node in enumerate(perturbators_nodes_list):
        v_node = pert_vict_idx[r]
        adj[p_node, v_node] = 1  # update the perturbators nodes
        feat[p_node] = avg_degree_target
        adj[v_node, p_node] = 1  # update the victims nodes
    adj_sp = ss.csr_matrix(adj)  # convert the adj to sp.csr_matrix format
    graph = dgl.from_scipy(adj_sp)  # update adj_sp to dgl graph
    graph.ndata['feat'] = feat  # update the feat
    graph.ndata['label'] = label
    print("time: ", time.time() - t0)
    return graph, victims_nodes_list, target_mask


if __name__ == "__main__":
    # ---Preprocessing---
    # Load dataset
    data = LoadDataset(args.data_name)
    graph = data.load_data()
    feat, label = graph.ndata['feat'], graph.ndata['label']
    # Randomly split the train, validation, test mask by given CUT_RATE
    _, _, test_mask = split_masks(label, cut_rate=CUT_RATE)

    # ---Simulate Non-malicious random connections---
    graph_rc, target_nodes_list, target_mask = \
        random_connections(graph, test_mask, args.SAMPLE_RATE, PERT_RATE, args.TARGET_CLASS)
    feat_rc = graph_rc.ndata['feat']

    # ---Initialize the node classifier---
    print("Initialize the node classifier...")
    # Setup the GPU if necessary
    if args.GPU < 0:
        print("Using CPU!")
        cuda = False
    else:
        print("Using GPU!")
        cuda = True
        torch.cuda.set_device(args.GPU)
        graph_rc = graph_rc.to('cuda')
        feat_rc = feat_rc.cuda()
        label = label.cuda()
        target_mask = target_mask.cuda()
    # Create the trained model    
    model = GNN(g=graph_rc,
                in_feats=feat_rc.shape[1],
                n_hidden=N_HIDDEN,
                n_classes=len(torch.unique(label)),
                n_layers=N_LAYERS,
                activation=F.relu,
                dropout=DROPOUT,
                model_name=args.model_name,
                aggregator_type=kwargs_dicts[args.model_name][0],
                n_filter=kwargs_dicts[args.model_name][1],
                n_heads=kwargs_dicts[args.model_name][2])
    if cuda:  # if GPU is available
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Path for saving the parameters
    path = 'runs/{0}_{1}/'.format(args.data_name, args.model_name) + 'model_best.pth.tar'

    # ---Evaluation the trained GNN on the perturbed target nodes---
    print("Evaluation on the target nodes.")
    evaluation(model, optimizer, path, graph_rc, feat_rc, label, target_mask, cuda)

    # ---Output the perturbed graph and the predictions---
    print("Save the perturbed graph and the target mask.")
    if cuda:
        graph_rc, target_mask = graph_rc.cpu(), target_mask.cpu()
    dump_pickle(dirs_attack+'G_rdm_C{0}_{1}_{2}.pkl' \
                .format(args.TARGET_CLASS, args.data_name, args.model_name), \
                [graph_rc, target_nodes_list, target_mask])
    print("Generate predicted labels after perturbation.")
    Y_pred, Y_pred_sm = prediction(model, optimizer, path, graph_rc, feat_rc)
    if cuda:
        Y_pred, Y_pred_sm = Y_pred.cpu(), Y_pred_sm.cpu()
    dump_pickle('../data/noisy_label/Y_preds_rdmPert.pkl', [Y_pred, Y_pred_sm])
    print("Y_pred/Y_pred_sm.shape: ", Y_pred.shape, Y_pred_sm.shape)
    