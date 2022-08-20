#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: Utils modules
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

import os
import sys
import pickle
import yaml
import torch
import numpy as np
import dgl
import scipy.sparse as ss
from scipy.stats import entropy
from sklearn.metrics import f1_score
from zipfile import ZipFile


def read_pickle(file_name):
    """Load the dataset"""
    with open (file_name,'rb') as file:
        return pickle.load(file)

def dump_pickle(file_name, data):
    """Export the dataset"""
    with open (file_name,'wb') as file:
        pickle.dump(data, file)

def unzip_file(file_path, output_path):
    """Extract all the contents of the zip file"""
    if os.path.exists(file_path):  # check if the zip file exists
        with ZipFile(file_path, 'r') as zipObj:
            # extracted files will overwrite the existing files with the same name.
            zipObj.extractall(path=output_path)
            print('File is unzipped to "{0}".'.format(output_path))
    else:
        sys.exit("Model parameter file is not found.")

def read_yaml_file(path: str, file_name: str) -> dict:
    """
    @title: reads a .yaml file and returns its content as a dictionary
    @input: path (str): directory path; file_name (str): filename (without file extension).
    @returns: dict: contents of .yaml file
    @reference: https://github.com/stadlmax/Graph-Posterior-Network/tree/main/gpn/utils
    @example: config = read_yaml_file('./configs/xx', 'yaml_file_name')
    """
    file_name = file_name.lower()
    file_path = os.path.join(path, f'{file_name}.yaml')
    if not os.path.exists(file_path):  # check the file path
        raise AssertionError(f'"{file_name}"" file is not found in {path}!')
    with open(file_path) as file:  # open the file path
        yaml_file = yaml.safe_load(file)
    if yaml_file is None:
        yaml_file = {}
    return yaml_file

def preprocess_dgl_adj(graph):
    """
    @topic: Normalize adjacency matrix (dgl graph) for GCNs.
    @input: graph (dgl graph).
    @return: graph_normalized (dgl graph).
    """
    adj_csr = graph.adjacency_matrix(scipy_fmt="csr")  # convert dgl graph to csr matrix
    adj_csr += ss.eye(adj_csr.shape[0])  # add self-connection for each node
    rowsum = adj_csr.sum(1).A1  # sum up along the columns
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # eliminate the inf number
    d_mat_inv_sqrt = ss.diags(d_inv_sqrt)  # compute the inverse squared degree matrix
    adj_normalized = adj_csr.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt)  # note that both matrix should be sparse matrix.
    graph_normalized = dgl.from_scipy(adj_normalized)
    graph_normalized.ndata['feat'] = graph.ndata['feat']  # float32
    graph_normalized.ndata['label'] = graph.ndata['label']  # int64
    return graph_normalized

def split_masks(Y, cut_rate=[0.1, 0.2]):
    """
    @topic: Split the train/val/test masks
    @input: Y: real label; cut_rate: the cur ratio of train/validation mask.
    @return: label masks (train_mask, val_mask, test_mask).
    """

    def create_mask(shape):
        # Create a zero tensor for mask
        return torch.zeros([shape], dtype=torch.bool)

    # Create masks
    tensor_shape = Y.shape[0]
    train_mask, val_mask, test_mask = create_mask(tensor_shape), create_mask(tensor_shape), create_mask(tensor_shape)
    # Generate a random idx
    torch.manual_seed(42)
    idx = list(torch.utils.data.RandomSampler(range(0, Y.shape[0])))
    # Split the mask
    train_cut_pos, valid_cut_pos = cut_rate[0], cut_rate[0]+cut_rate[1]
    train_mask[idx[ : int(len(idx)*train_cut_pos)]] = True
    val_mask[idx[int(len(idx)*train_cut_pos) : int(len(idx)*valid_cut_pos)]] = True
    test_mask[idx[int(len(idx)*valid_cut_pos) : int(len(idx)*1)]] = True
    return train_mask, val_mask, test_mask

def generate_random_noise_label(label, noisy_ratio=0.3, seed=0):
    """
    @topic: Randomly generate noise label with given noisy_ratio.
    @input: lable(1D-array), noise_ratio(float), seed(int).
    @return: noisy label (1D-array).
    """
    np.random.seed(seed)
    label_ = np.random.randint(min(label), high=max(label), size=len(label))
    mask_idx = np.random.choice(len(label), int(noisy_ratio*len(label)), replace=False)
    label = np.array(label)
    label[mask_idx] = label_[mask_idx]
    return label

def select_target_nodes(label, test_mask, sample_rate=0.1, atk_class=-1):
    """
    @topic: Select target nodes for targeted/non-targeted perturbations.
    @input:
        label (int tensor): ground-truth label;
        test_mask (bool tensor): the mask for testing set;
        sample_rate (float): the ratio of sampling in the testing set;
        atk_class (int): the attacked target class.
    @return:
        target_nodes_list (array): the list of target nodes;
        target_mask (bool tensor): the mask for target nodes.
    """
    target_mask = torch.zeros([len(label)], dtype=torch.bool)
    test_id_list = [i for i in range(len(test_mask)) if test_mask[i] == True]
    target_size = int(len(label[test_mask])*sample_rate) # Decide the size of target nodes
    if int(atk_class) in torch.unique(label): # Select "atk_class" nodes from test graph
        target_idx = [l for l in range(len(label)) if label[l] == atk_class and l in test_id_list]
        target_nodes_list = [i for i in target_idx[:target_size]]
    else: # Random select "target_size" nodes if "atk_class" doesn't belong to any existing classes
        np.random.seed(abs(atk_class)) # Fix the random seed for reproduction.
        #test_id_list = test_id_list[:int(target_size*1.2)]
        target_nodes_list = np.random.choice(test_id_list, target_size, replace=False)
    target_mask[target_nodes_list] = True # Generate the target mask
    return target_nodes_list, target_mask

def nodes_filter(node_degrees, target_nodes, ranges=[0,10]):
    """Filter target nodes based on the degrees within the given range"""
    nd_target = node_degrees[target_nodes]
    idx_nd = np.where((nd_target>ranges[0]) & (nd_target<ranges[1]))[0] # index in target_nodes (not node_id)
    target_nodes_ft = target_nodes[idx_nd]  # node_id
    # Update the target_mask based on node_id
    target_mask = torch.zeros([len(node_degrees)], dtype=torch.bool)
    target_mask[target_nodes_ft] = True
    return target_nodes_ft, target_mask

def gen_init_trans_matrix(Y_pred_sm, Y_noisy, NUM_CLASSES):
    """
    @topic: Generate initial transition matrix
    @input:
        Y_pred_sm (2D-array: NUM_SAMPLES x NUM_CLASSES);
        Y_noisy (1D-array: NUM_SAMPLES x 1);
        NUM_CLASSES (int).
    @return: TM_init (2D-array: NUM_CLASSES x NUM_CLASSES).
    """
    unnorm_TM = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(len(Y_noisy)):
        label = Y_noisy[i]
        unnorm_TM[:,label] += Y_pred_sm[i]
    unnorm_TM_sum = np.sum(unnorm_TM, axis=1)
    TM_init = unnorm_TM / unnorm_TM_sum[:,None]
    return TM_init

def MinMaxScaler(data, low, high):
    """
    @topic: Rescale 2D matrix into given range
    @input: data (2d matrix), low/high (Scalar).
    @return: scaled data (2d matrix).
    """
    data_max, data_min = data.max(axis=0), data.min(axis=0)
    data_std = (data - data_min) / (data_max - data_min + 0.00001)
    data_scaled = data_std * (high - low) + low
    return data_scaled

def compute_accuracy(logits, labels, mask):
    """Compute the accuracy"""
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def compute_f1_score(logits, labels, mask, avg='macro'):
    """Compute the f1 score"""
    logits = logits[mask]
    _, y_pred = torch.max(logits, dim=1)
    y_true = labels[mask]
    return f1_score(y_true, y_pred, average=avg)

def compute_entropy(logits, mask):
    """Compute the normalized entropy"""
    logits = logits[mask]
    cat_dist = torch.nn.functional.softmax(logits, dim=1)  # could also use .topk(1, dim = 1) to find the max
    total_entropy = entropy(cat_dist.detach().numpy(), axis=1)  # [0, log(N)]
    normalized_entropy = total_entropy / np.log(cat_dist.shape[1])  # a.k.a. efficiency
    return np.mean(normalized_entropy), np.std(normalized_entropy)

def dist_pre_class(logits, labels, mask):
    """Return the distribution of each class"""
    logits = logits[mask]
    _, y_pred = torch.max(logits, dim=1)
    frq_pred = torch.bincount(y_pred)
    y_true = labels[mask]
    frq_true = torch.bincount(y_true)
    return frq_true.numpy(), frq_pred.numpy()

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    import shutil
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory+'model_best.pth.tar')

def load_checkpoint(checkpoint_fpath, model, optimizer):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['best_acc']
    return model, optimizer, checkpoint['epoch'], best_acc

def count_parameters(model):
    """Count the number of trianable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor2array(tensor, gpu):
    """Convert tensor to numpy array (Input must be Tensor)!"""
    return tensor.numpy() if gpu < 0 else tensor.cpu().numpy()

def bool2num(mask_bool, mask_type="tensor"):
    """Convert the boolean mask to the indices with integer numbers"""
    label_new = [i for i in range(len(mask_bool)) if mask_bool[i] == True]
    return torch.LongTensor(label_new) if mask_type == "tensor" else np.int64(label_new)

def spcsr2tscoo(CSR):
    """Convert scipy csr matrix to tensor coo matrix"""
    COO = CSR.tocoo() # convert to coo for getting the rows & cols
    indices = torch.LongTensor(np.vstack((COO.row, COO.col)))
    values = torch.LongTensor(COO.data)
    return torch.sparse_coo_tensor(indices, values, COO.shape, dtype=torch.float32)

def compute_edge_homo_ratio(graph, Y_gt):
    """Compute the edge homophily ratio of the given graph."""
    # ref: https://proceedings.neurips.cc/paper/2020/file/58ae23d878a47004366189884c2f8440-Paper.pdf
    num_homo_edge, num_edges = 0, 0
    adj = graph.adjacency_matrix(scipy_fmt="csr")
    adj = spcsr2tscoo(adj)
    for i, y_i in enumerate(Y_gt):
        node_idx = adj[i].coalesce().indices() # find the index of the current node (sparse mode)
        neighbors = node_idx[node_idx != i] # find the neighbors by removing self-connections
        Y_neighbors = Y_gt[neighbors] # the ground-truth labels of neighbors
        for y_n in Y_neighbors:
            if y_n == y_i:
                num_homo_edge += 1 # compute the number of homo edge
            num_edges += 1 # compute number of edge
    homo_ratio = num_homo_edge / num_edges # compute the edge homo ratio
    return homo_ratio
