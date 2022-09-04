#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: Implement Nettack
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

import sys
import argparse
import numpy as np
import dgl
import torch
import torch.nn.functional as F
import scipy.sparse as ss
from models_GNN import GNN
from baselines.nettack import nettack as ntk
from load_data import LoadDataset
from utils import read_yaml_file, read_pickle, dump_pickle, split_masks, \
        generate_random_noise_label, select_target_nodes, nodes_filter
from train_GNN import evaluation, prediction


# Define the arguments
parser = argparse.ArgumentParser(description="Read arguments for training.")
parser.add_argument("--config_file", type=str, default="perturbations", help="The name of config files.")
parser.add_argument("--data_name", type=str, default="cora", help="The name of dataset.",
                    choices=["cora", "citeseer", "pubmed", "amazoncobuy", "coauthor", "reddit"])
parser.add_argument("--model_name", type=str, default="GCN", help="The name of GNNs.")  # GCN, SGC, GraphSAGE, GAT
parser.add_argument("--pert_type", type=str, default="clean", help="The type of pre-perturbations on graphs.",
                    choices=["clean", "infoSparse"])
parser.add_argument("--TARGET_CLASS", type=int, default=-1, help="targeted attack: class_id; non-targeted attack: -1.")
parser.add_argument("--attack_type", type=str, default="lf", help="The type of attack for Nettack.",
                    choices=["l", "f", "lf"])
parser.add_argument("--NUM_PERT", type=int, default=2, help="Density of perturbations for target nodes.")
parser.add_argument("--SAMPLE_RATE", type=float, default=0.1, help="The percentage of test nodes being attacked.")
parser.add_argument("--NOISE_RATIO", type=float, default=0.1, help="Noise ratio.")
parser.add_argument("--GPU", type=int, default=-1, help="Input GPU device id or -1.")
parser.add_argument("--is_attack", type=bool, default=False, help="Attack or not.")
parser.add_argument("--use_config_file", type=bool, default=False, help="Pass arguments from config files or not.")
args = parser.parse_known_args()[0]

# Read config files
config_file = read_yaml_file("../config", args.config_file)
if config_file and args.use_config_file:
    pert_config = config_file["advAttack"]
    args.data_name = pert_config["data_name"]
    args.model_name = pert_config["model_name"]
    args.pert_type = pert_config["pert_type"]
    args.TARGET_CLASS = pert_config["TARGET_CLASS"]
    args.attack_type = pert_config["attack_type"]
    args.NUM_PERT = pert_config["NUM_PERT"]
    args.SAMPLE_RATE = pert_config["SAMPLE_RATE"]
    args.NOISE_RATIO = pert_config["NOISE_RATIO"]
    args.GPU = pert_config["GPU"]
    args.is_attack = pert_config["is_attack"]

# The paramaters for adversarial attacks
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


def attack_graph(adj, feat, label, w0, w1, target_nodes, attack_type, num_pert=2):
    """
    @topic: Implement Nettack on target nodes.
    @input:
        adj/feat (sp.csr_matrix): Original adjacency matrix/feature matrix;
        label (list): The label of nodes;
        w0/w1 (list of list): The weights for input/output layer of GCN model;
        target_nodes (list): The target nodes (id \in [0, num_nodes-1]);
        attack_type (string): The type of attack (l/f/lf);
        num_pert (int): The number of perturbations.
    @return:
        graph_atk/feat_atk (dgl.graph/tensor): The attacked graph/feature matrix.
    """
    adj_, feat_ = adj, feat
    node_degrees = adj_.sum(axis=0).A1  # id \in [0, num_nodes-1]
    attack_type_list = ["l", "f", "lf"]
    if attack_type not in attack_type_list:
        print("Please select the correct type of attack!")
        return 0
    for node in target_nodes:  # Traverse all targeted nodes
        if num_pert < 1:
            n_perturbation = int(node_degrees[node-1])
        else:
            n_perturbation = num_pert
        for _, atk_type in enumerate(attack_type):
            if atk_type == "l":  # Select the type of attack
                perturb_structure, perturb_features = True, False
            if atk_type == "f":
                perturb_structure, perturb_features = False, True
                n_perturbation *= 10
            attacker = ntk.Nettack(adj_, feat_, label, w0, w1, node, verbose=True)
            attacker.reset()
            attacker.attack_surrogate(n_perturbation, perturb_structure, perturb_features) # direct attack
            adj_ = attacker.adj  # Update the adj & feat iteratively
            feat_ = attacker.X_obs
    graph_atk = dgl.from_scipy(adj_)  # Convert spmatrix to dgl graph (dgl 0.5.x)
    feat_atk = torch.FloatTensor(feat_.toarray())
    return graph_atk, feat_atk  # Return the attacked graph/feat matrix


if __name__ == "__main__":
    # ---Preprocessing---
    # Load dataset
    if args.pert_type == "clean":
        data = LoadDataset(args.data_name)
        graph = data.load_data()
    elif args.pert_type == "infoSparse":  # for denser graphs
        try:
            graph_name = 'G_isp_C{0}_{1}_{2}.pkl'.format(args.TARGET_CLASS, \
                                            args.data_name, args.model_name)
            graph, _, _ = read_pickle(dirs_attack+graph_name)
        except:
            print("Fail to load the infoSparse graph.")
            sys.exit()
    else:
        print("Please select the correct scenario: clean, infoSparse.")
        sys.exit()
    feat, label = graph.ndata['feat'], graph.ndata['label']
    w0, w1 = read_pickle(dirs_attack+'W_{0}_{1}.pkl'.format(args.data_name, args.model_name))
    # Preprocessing the graph
    adj_sp = graph.adjacency_matrix(scipy_fmt="csr")
    feat_sp = ss.csr_matrix(feat)
    # Generate noisy label
    Y_noisy = generate_random_noise_label(label, noisy_ratio=args.NOISE_RATIO, seed=0)
    Y_noisy_lst = Y_noisy.tolist() # array --> list
    # Randomly split the train, validation, test mask by given cut rate
    train_mask, val_mask, test_mask = split_masks(label, cut_rate=CUT_RATE)
    # Present the average degree of test nodes
    node_degrees = adj_sp.sum(axis=0).A1
    print("The average degree of test nodes: ", np.mean(node_degrees[test_mask]))
    # Setup the gpu if necessary
    if args.GPU < 0:
        print("Using CPU!")
        cuda = False
    else:
        print("Using GPU!")
        cuda = True
        torch.cuda.set_device(args.GPU)
        graph = graph.to('cuda')
        feat = feat.cuda()
        label = label.cuda()
        test_mask = test_mask.cuda()

    # ---Initialize the model---
    print("Initialize the model...")
    model = GNN(g=graph,
                in_feats=feat.shape[1],
                n_hidden=N_HIDDEN,
                n_classes=len(torch.unique(label)),
                n_layers=N_LAYERS,
                activation=F.relu,
                dropout=DROPOUT,
                model_name=args.model_name,
                aggregator_type=kwargs_dicts[args.model_name][0],
                n_filter=kwargs_dicts[args.model_name][1],
                n_heads=kwargs_dicts[args.model_name][2])
    if cuda:  # if gpu is available
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Path for saving the parameters
    path = 'runs/{0}_{1}/'.format(args.data_name, args.model_name) + 'model_best.pth.tar'

    if args.is_attack:
        # ---Attack on graph---
        # Implement node-level attack on the test graph (links, feats, and L&F)
        # If TARGET_CLASS = -1, non-target attack.
        print("Implement node-level attack on the test graph.")
        target_nodes_list, target_mask = \
            select_target_nodes(label, test_mask, \
                                args.SAMPLE_RATE, atk_class=args.TARGET_CLASS)
        # Filter the target nodes based on the degrees (within given ranges).
        target_nodes_list, target_mask = \
            nodes_filter(node_degrees, target_nodes_list, ranges=[0,10])
        print("Total degrees of target nodes: ", node_degrees[target_nodes_list])
        target_mask = target_mask.cuda() if cuda else target_mask
        print("Evaluation before attack on target nodes: ")
        evaluation(model, optimizer, path, graph, feat, label, target_mask, cuda)
        graph_atk, feat_atk = attack_graph(adj_sp, feat_sp, Y_noisy_lst, w0, w1, \
                                target_nodes_list, args.attack_type, args.NUM_PERT)
        graph_atk = graph_atk.to('cuda') if cuda else graph_atk
        feat_atk = feat_atk.cuda() if cuda else feat_atk
        graph_atk.ndata['feat'] = feat_atk # Update the attacked graph
        graph_atk.ndata['label'] = label
        print("Evaluation after attack on target nodes: ")
        evaluation(model, optimizer, path, graph_atk, feat_atk, label, target_mask, cuda)
        # Save the attacked graph
        if cuda:
            graph_atk, target_mask = graph_atk.cpu(), target_mask.cpu()
        dump_pickle(dirs_attack+'G_atk_C{0}_T{1}_{2}_{3}.pkl' \
                    .format(args.TARGET_CLASS, args.attack_type, \
                            args.data_name, args.model_name), \
                    [graph_atk, target_nodes_list, target_mask])
    else:
        # ---Generate predicted label after attack---
        # If not attack, reload the attacked graph and mask.
        graph_atk, target_nodes_list, target_mask = \
                read_pickle(dirs_attack+'G_atk_C{0}_T{1}_{2}_{3}.pkl' \
                            .format(args.TARGET_CLASS, args.attack_type, \
                                    args.data_name, args.model_name))
        feat_atk = graph_atk.ndata['feat']
        graph_atk = graph_atk.to('cuda') if cuda else graph_atk
        feat_atk = feat_atk.cuda() if cuda else feat_atk
        print("Generate predicted label after attack.")
        Y_pred, Y_pred_sm = prediction(model, optimizer, path, graph_atk, feat_atk)
        if cuda:
            Y_pred, Y_pred_sm = Y_pred.cpu(), Y_pred_sm.cpu()
        dump_pickle('../data/noisy_label/Y_preds_advAttack.pkl', [Y_pred, Y_pred_sm])
        print("Y_pred/Y_pred_sm.shape: ", Y_pred.shape, Y_pred_sm.shape)
        print("Evaluation after attack on target nodes: ")
        evaluation(model, optimizer, path, graph_atk, feat_atk, label, target_mask, cuda)
