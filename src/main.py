#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: Employ LInDT model to infer labels
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from load_data import LoadDataset
from train_GNN import prediction
from models_GNN import GNN
from models_LInDT import LInDT
from utils import read_yaml_file, read_pickle, gen_init_trans_matrix, \
                    select_target_nodes
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import entropy


# Define the arguments
parser = argparse.ArgumentParser(description="Read arguments for training.")
parser.add_argument("--config_file", type=str, default="model_lindt", help="The name of config files.")
parser.add_argument("--data_name", type=str, default="cora", help="The name of dataset.",
                    choices=["cora", "citeseer", "pubmed", "amazoncobuy", "coauthor", "reddit"])
parser.add_argument("--model_name", type=str, default="GCN", help="The name of GNNs.")  # GCN, SGC, GraphSAGE, GAT
parser.add_argument("--pert_type", type=str, default="rdmPert", help="The type of perturbations.",
                    choices=["clean", "rdmPert", "infoSparse", "advAttack"])
parser.add_argument("--TARGET_CLASS", type=int, default=-1, help="targeted attack: class_id; non-targeted attack: -1.")
parser.add_argument("--attack_type", type=str, default="lf", help="The type of attack for Nettack.",
                    choices=["l", "f", "lf"])
parser.add_argument("--SAMPLE_RATE", type=float, default=0.1, help="The percentage of test nodes being attacked.")
parser.add_argument("--GPU", type=int, default=-1, help="Input GPU device id or -1.")
parser.add_argument("--is_inferable", type=bool, default=True, help="Inference or not.")
parser.add_argument("--infer_type", type=str, default="pseudo", help="The type of labels used for inference.",
                    choices=["pseudo", "noise"])
parser.add_argument("--topo_type", type=str, default="major", help="The type of kernel for topo-sampling.",
                    choices=["", "random", "major", "degree"])
parser.add_argument("--NUM_EPS", type=int, default=10, help="The number of inference epochs.")
parser.add_argument("--NUM_RETRAIN", type=int, default=60, help="The number of retraining epochs.")
parser.add_argument("--WARMUP_STEP", type=int, default=40, help="The number of warm-up epochs.")
parser.add_argument("--is_alert", type=bool, default=False, help="Alert or Recover.")
parser.add_argument("--is_dynamic_phi", type=bool, default=True, help="Update the transition matrix or not.")
parser.add_argument("--is_dynamic_alpha", type=bool, default=True, help="Update the alpha vector or not.")
parser.add_argument("--ALPHA", type=float, default=0.1, help="The initial alpha value.")
parser.add_argument("--use_config_file", type=bool, default=False, help="Pass arguments from config files or not.")
args = parser.parse_known_args()[0]

# Read config files
config_file = read_yaml_file("../config", args.config_file)
if config_file and args.use_config_file:
    infer_config = config_file["inference"]
    args.data_name = infer_config["data_name"]
    args.model_name = infer_config["model_name"]
    args.pert_type = infer_config["pert_type"]
    args.TARGET_CLASS = infer_config["TARGET_CLASS"]
    args.attack_type = infer_config["attack_type"]
    args.SAMPLE_RATE = infer_config["SAMPLE_RATE"]
    args.GPU = infer_config["GPU"]
    args.is_inferable = infer_config["is_inferable"]
    args.infer_type = infer_config["infer_type"]
    args.topo_type = infer_config["topo_type"]
    args.NUM_EPS = infer_config["NUM_EPS"]
    args.NUM_RETRAIN = infer_config["NUM_RETRAIN"]
    args.WARMUP_STEP = infer_config["WARMUP_STEP"]
    args.is_alert = infer_config["is_alert"]
    args.is_dynamic_phi = infer_config["is_dynamic_phi"]
    args.is_dynamic_alpha = infer_config["is_dynamic_alpha"]
    args.ALPHA = infer_config["ALPHA"]
assert args.infer_type == "pseudo" or args.infer_type == "noise"

# The parameters for node classifier
LR = 0.001
N_LAYERS = 2
N_HIDDEN = 200
DROPOUT = 0
WEIGHT_DECAY = 0
kwargs_dicts = {"GCN": [None, None, None],  # aggregator_type, n_filter, n_heads.
                "GraphSAGE": ["gcn", None, None],
                "SGC": [None, 2, None],
                "GAT": [None, None, 3]}
topo_dicts = {"random": "random_kernel",
               "major": "majority_kernel",
               "degree": "degree_weighted_kernel"}
Kernel_Type = topo_dicts[args.topo_type] if args.topo_type in topo_dicts.keys() else ""

# ---Preprocessing---
# Reload the labels and masks
label, Y_noisy, train_mask, val_mask, test_mask = \
                      read_pickle('../data/noisy_label/Y_gt_noisy_masks.pkl')
print("Load the {0} dataset for {1}.".format(args.data_name, args.model_name))
if args.pert_type == "clean":
    data = LoadDataset(args.data_name)
    graph = data.load_data()
    feat = graph.ndata['feat']
    _, target_mask = select_target_nodes(label, test_mask, \
                                args.SAMPLE_RATE, atk_class=args.TARGET_CLASS)
    file_name = ''
else:
    if args.pert_type == "advAttack":
        graph_name = 'G_atk_C{0}_T{1}_{2}_{3}.pkl'\
                      .format(args.TARGET_CLASS, args.attack_type, \
                              args.data_name, args.model_name)
    elif args.pert_type == "rdmPert":
        graph_name = 'G_rdm_C{0}_{1}_{2}.pkl'\
                      .format(args.TARGET_CLASS, args.data_name, args.model_name)
    elif args.pert_type == "infoSparse":
        graph_name = 'G_isp_C{0}_{1}_{2}.pkl'\
                      .format(args.TARGET_CLASS, args.data_name, args.model_name)
    else:
        print("Please select the correct scenario: clean, advAttack, rdmPert, infoSparse.")
        sys.exit()
    dirs_attack = '../data/attacker_data/'
    graph, _, target_mask = read_pickle(dirs_attack+graph_name)
    feat = graph.ndata['feat']
    file_name = '_{0}'.format(args.pert_type)
# Reload the corresponding predicted labels
Y_pred, Y_pred_sm = read_pickle('../data/noisy_label/Y_preds{0}.pkl'.format(file_name))
# Reload the clean predicted labels for inference
Y_cpred, Y_cpred_sm = read_pickle('../data/noisy_label/Y_preds.pkl')
# Convert tensor to numpy array
Y_gt, Y_cpred, Y_noisy, Y_pred, Y_pred_sm, Y_cpred_sm = \
    label.numpy(), Y_cpred.numpy(), Y_noisy.numpy(), Y_pred.numpy(), \
    Y_pred_sm.detach().numpy(), Y_cpred_sm.detach().numpy()

# ---Initialize the initial warm-up transition matrix---
print("Initialize the warm-up transition matrix...")
Y_pred_sm_train = Y_cpred_sm[train_mask]  # predicted probability table (num_samples, num_classes)
Y_noisy_train = Y_noisy[train_mask]  # noisy label
NUM_CLASSES = len(set(label.numpy()))
print("NUM_CLASSES: ", NUM_CLASSES)
TM_warmup = gen_init_trans_matrix(Y_pred_sm_train, Y_noisy_train, NUM_CLASSES)
print("The shape of warm-up TM is: ", TM_warmup.shape)

# ---Setup the gpu if necessary---
if args.GPU < 0:
    print("Using CPU!")
    cuda = False
else:
    print("Using GPU!")
    cuda = True
    torch.cuda.set_device(args.GPU)
    graph = graph.to('cuda')
    feat = feat.cuda()

# ---Initialize the node classifier---
print("Initialize the node classifier...")
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
dirs = 'runs/{0}_{1}/'.format(args.data_name, args.model_name)
path = dirs + 'model_best.pth.tar'

# --- Label Inference jointly using Dirichlet and Topological sampling ---
if args.is_inferable:
    print("Infer the label...")
    Y_ssup_dict = {"pseudo": Y_cpred, "noise": Y_noisy}
    timer_0 = time.time()
    ss = LInDT(args.ALPHA)  # Y_noisy or Y_cpred
    Y_infer, C_new, _ = \
        ss.infer(model, optimizer, dirs, graph, feat, train_mask, val_mask, \
                 Y_pred, Y_pred_sm, Y_ssup_dict[args.infer_type], Y_gt, TM_warmup, \
                 args.NUM_RETRAIN, args.GPU, args.NUM_EPS, args.WARMUP_STEP, \
                 args.is_alert, args.is_dynamic_phi, Kernel_Type, args.is_dynamic_alpha)
    runtime = time.time() - timer_0
    print("\n Runtime: ", runtime)
    print("Y_inferred: \n {0} \n C_new: \n {1}".format(Y_infer, C_new))
print("Evaluation after inference:")
Y_infer, C_new, _ = read_pickle('../data/noisy_label/Y_C_RUN.pkl')  # array
_, cat_dist = prediction(model, optimizer, path, graph, feat)  # tensor
# Evaluation on test/target graph
mask_dict = {"test": test_mask, "target": target_mask}
assert len(mask_dict) > 0
for mask_name, mask in mask_dict.items():
    Y_gt_mask = label[mask].numpy()
    Y_infer_mask = Y_infer[mask]
    cat_dist_mask = cat_dist[mask].cpu() if cuda else cat_dist[mask]
    cat_dist_mask = cat_dist_mask.detach().numpy()
    acc_infer = accuracy_score(Y_gt_mask, Y_infer_mask)
    print("Accuracy of Y_infer on {} nodes: {:.2%}.".format(mask_name, acc_infer))
    f1_macro = f1_score(Y_gt_mask, Y_infer_mask, average='macro')
    f1_weighted = f1_score(Y_gt_mask, Y_infer_mask, average='weighted')
    print("F1 score (macro) of Y_infer on {} nodes: {:.2%}.".format(mask_name, f1_macro))
    print("F1 score (weighted) of Y_infer on {} nodes: {:.2%}.".format(mask_name, f1_weighted))
    total_entropy = entropy(cat_dist_mask, axis=1)
    normalized_entropy = total_entropy / np.log(cat_dist.shape[1])
    print("The normalized entropy on {} nodes: {:.2%}."\
          .format(mask_name, np.mean(normalized_entropy)))
    print("-"*30)
