#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: LInDT's Model
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

import os
import numpy as np
import torch
from train_GNN import train, prediction
from utils import dump_pickle, tensor2array, spcsr2tscoo
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from collections import defaultdict
from random import choice


# Label Inference jointly using Dirichlet and Topological sampling --------------------
class LInDT():
    def __init__(self, ALPHA=1.0):
        self.ALPHA = ALPHA  # a parameter for computing TM (float).

    def get_labels_dict(self, Y_pred, Y_ssup):
        """
        @topic: Convert labels as dict.
        @input: Y_pred/Y_ssup (1D array).
        @return: infer_dict/ssup_dict (dict).
        """
        infer_dict = dict()  # keys: idx; values: Y_pred.
        ssup_dict = dict()  # keys: idx; values: Y_ssup.s
        idx = np.array([i for i in range(len(Y_ssup))])
        for i in range(len(idx)):
            infer_dict[idx[i]] = Y_pred[i]
            ssup_dict[idx[i]] = Y_ssup[i]
        return infer_dict, ssup_dict


    def generate_counting_matrix(self, Y_pred, Y_ssup, NUM_CLASSES):
        """
        @topic: Generate counting matrix and testing labels.
        @input: Y_pred/Y_ssup (1D array); NUM_CLASSES (int).
        @return: C (2D array).
        """
        C_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for i in range(len(Y_ssup)):
            r = Y_pred[i]
            c = Y_ssup[i]
            C_matrix[r][c] += 1
        return C_matrix # (NUM_CLASSES, NUM_CLASSES)

    
    def isUncertain(self, y_infer_new, y_uct_old, y_ssup):
        """
        @topic: Check whether current inferred labels is uncertain.
        @input: y_infer_new/y_uct_old/y_ssup (int);
        @return: true/false (bool).
        """
        return y_infer_new != y_uct_old or y_infer_new != y_ssup


    def topo_sampler(self, node, nid, Y_ssup, node_degrees, kernel_type):
        """
        @topic: Sample a label with a given topology-based label sampler.
        @input:
            node: vector representation of the current node (1D sparse tensor: NUM_SAMPLES x 1);
            nid: node id (int);
            Y_ssup: self-supervised labels (a.k.a. auto-generated labels or noisy labels) (1D tensor: NUM_SAMPLES x 1);
            node_degrees: total degrees for all nodes (1D sparse tensor: NUM_SAMPLES x 1);
            kernel_type: the types of the topology-based label sampling kernel (str).
        @return:
            An inferred label of the current node (int) or None.
        """
        def random_kernel(Y_neighbors):
            """Randomly sample a label in neigbhors."""
            return choice(Y_neighbors)

        def majority_kernel(Y_neighbors):
            """Sample a label from majority class."""
            return torch.mode(Y_neighbors).values

        def degree_weighted_kernel(neighbors, Y_neighbors, node_degrees):
            """Sample a label from the maximum degree-weighted class."""        
            cd_neig_dict = defaultdict(int)  # accumulated degrees for each class in neighbors
            for l, label in enumerate(Y_neighbors):
                node_l, degree_l = int(neighbors[l].item()), 0
                if node_l in node_degrees.indices():  # check if node_l in the index of sparse tensors
                    idx_l = (node_degrees.indices().flatten()==node_l).nonzero().item()  # find the actual index of node_l 
                    degree_l = int(node_degrees.values()[idx_l].item())  # update the degree_l
                cd_neig_dict[label.item()] += degree_l
            return max(cd_neig_dict, key=cd_neig_dict.get) # find the key (class) with max values (degree) 

        node_idx = node.coalesce().indices()  # find the index of the current node (sparse mode)
        neighbors = node_idx[node_idx != nid]  # find the neighbors by removing self-connections
        Y_neighbors = Y_ssup[neighbors] # the labels of neighbors (in Y_ssup)
        kernel_args = {"random_kernel": [Y_neighbors], # update this dicts when adding new kernel
                       "majority_kernel": [Y_neighbors],
                       "degree_weighted_kernel": [neighbors, Y_neighbors, node_degrees]}
        args_generator = (args for args in kernel_args[kernel_type])
        return locals()[kernel_type](*args_generator) if len(Y_neighbors) else None


    def topo_sampling(self, Y_pred_sm, Y_ssup, TM, Y_uct_old, adj, node_degrees, kernel_type):
        """
        @topic: Apply the topology-based label sampling on uncertain nodes if required.
        @input:
            Y_pred_sm: categorical distribution (2D tensor: NUM_SAMPLES x NUM_CLASSES);
            Y_ssup: self-supervised labels (a.k.a. auto-generated labels or noisy labels) (1D tensor: NUM_SAMPLES x 1);
            TM: transition matrix (2D tensor: NUM_CLASSES x NUM_CLASSES);
            Y_uct_old: previous inferred labels with uncertain nodes (1D tensor: NUM_SAMPLES x 1);
            adj: adjacency matrix of the graph (2D sparse tensor: NUM_SAMPLES x NUM_SAMPLES);
            node_degrees: total degrees for all nodes (1D sparse tensor: NUM_SAMPLES x 1);
            kernel_type: the types of the topology-based label sampling kernel (str).
        @return:
            Y_infer/Y_uct: inferred labels after/before topo-sampling (1D tensor: NUM_SAMPLES x 1);
            uncertain_rate: the ratio of uncertainty for Y_uct (float).
        """
        # torch.index_select(TM.T, 0, Y): sample the row of TM.T based on the class in Y.
        # ref: https://pytorch.org/docs/stable/generated/torch.index_select.html
        unnorm_probs = Y_pred_sm * torch.index_select(torch.transpose(TM,0,1), 0, Y_ssup)
        probs = unnorm_probs / torch.sum(unnorm_probs, axis=1, keepdims=True)
        Y_infer = torch.max(probs, dim=1)[1]
        Y_uct = torch.clone(Y_infer)  # inferred labels with uncertain nodes
        n_uncertain_nodes = 0
        for i in range(len(Y_infer)):
            # update the Y_infer based on topo-info if we meet uncertain nodes
            if self.isUncertain(int(Y_infer[i]), int(Y_uct_old[i]), int(Y_ssup[i])):
                n_uncertain_nodes += 1
                if kernel_type:
                    y_infer_i = self.topo_sampler(adj[i], i, Y_ssup, node_degrees, kernel_type)
                    if y_infer_i:
                        Y_infer[i] = int(y_infer_i)  # update after topo-sampling
        uncertain_rate = n_uncertain_nodes/len(Y_infer)
        return Y_infer, Y_uct, uncertain_rate
        

    def init_alpha(self, alpha0, NUM_CLASSES, label, isEqual):
        """
        @topic: Initialize the alpha vector.
        @input:
            alpha0: initial alpha value (float);
            NUM_CLASSES: the number of classes (int);
            label: labels (1D array: the size of labels x 1);
            isEqual: whether the alpha is equal for each class (bool).
        @return:
            ALPHA: alpha vector (1D array: NUM_CLASSES x 1).
        """
        if isEqual:  # all initial alpha0 value are equal.
            ALPHA = [alpha0 for _ in range(NUM_CLASSES)]  # add by row
        else:  # set alpha0 based on the class distribution in the train set
            ALPHA = [0 for _ in range(NUM_CLASSES)]  # add by row
            for c, frq in enumerate(np.bincount(label)):
                ALPHA[c] = frq / len(label)
                # ALPHA[c] = frq / len(label) + alpha0
        ALPHA = np.array(ALPHA).reshape(-1, 1)  # add by columns
        return ALPHA


    def update_alpha(self, Y_uct_cur, Y_uct_old, Alpha_vec):
        """
        @topic: Dynamically update the alpha vector.
        @input:
            Y_uct_cur: current inferred labels with uncertain nodes (1D tensor: NUM_SAMPLES x 1);
            Y_uct_old: previous inferred labels with uncertain nodes (1D tensor: NUM_SAMPLES x 1);
            Alpha_vec: alpha vector (1D tensor: NUM_CLASSES x 1).
        """
        Yi_cur_dist = torch.bincount(Y_uct_cur)
        Yi_old_dist = torch.bincount(Y_uct_old)
        if Yi_cur_dist.size(dim=0) != Yi_old_dist.size(dim=0):
            #TODO: Need to match the size of both distributions (pseudocode)
            #Yi_dist_new = torch.zeros_like(Yi_dist_longer)
            #for i in range(Yi_dist_shorter.size(dim=0))):
            #    Yi_dist_new[i] += Yi_dist_shorter[i]
            print("Fail to update the alpha vector!")
        else:
            Yi_dist_delta = Yi_cur_dist - Yi_old_dist
            for k, k_delta in enumerate(Yi_dist_delta):
                if Yi_old_dist[k].item() != 0:  # skip if #samples=0
                    Alpha_vec[k] *= 1 + k_delta.item() / Yi_old_dist[k].item()  # alpha_k*(1+delta/#samples)
                else:  # fill with mean value when #samples=0
                    Alpha_vec[k] = Alpha_vec.mean()


    def update_phi(self, Y_infer, C, z_dict, y_dict):
        """
        @topic: Dynamically update the transition matrix phi.
        @input:
            Y_infer: inferred labels (1D tensor: NUM_SAMPLES x 1);
            C: counting matrix (2D tensor: NUM_CLASSES x NUM_CLASSES);
            z_dict / y_dict (dict).
        """
        for i, y_if in enumerate(Y_infer):
            C[z_dict[i]][y_dict[i]] -= 1  # count down the numbers in C
            assert C[z_dict[i]][y_dict[i]] >= 0  # ensure all elements in C>=0
            z_dict[i] = int(y_if)  # update the inferred value i
            C[z_dict[i]][y_dict[i]] += 1  # update the numbers in C


    def infer(self, model, optimizer, dirs, graph, feat, train_mask, val_mask, \
                    Y_pred, Y_pred_sm, Y_ssup, Y_gt, TM_warmup, \
                    NUM_RETRAIN=60, GPU=-1, NUM_EPS=100, WARMUP_STEP=40, \
                    Alert=False, Dynamic_Phi=True, Kernel_Type="", Dynamic_Alpha=True):
        """
        @topic: Label inference using self-supervised labels (a.k.a. auto-generated labels or noisy labels).
        @input:
            Y_pred/Y_ssup/Y_gt: predicted/self-supervised/groundtruth labels (1D array: NUM_SAMPLES x 1);
            Y_pred_sm: categorical distribution (2D array: NUM_SAMPLES x NUM_CLASSES);
            TM_warmup: warming-up transition matrix (2D array: NUM_CLASSES x NUM_CLASSES);
            NUM_RETRAIN: the number of epochs for retraining the node classifier (int);
            GPU: the ID of GPU device (int);
            NUM_EPS: the number of epochs for each inference (int);
            WARMUP_STEP: using TM_warmup if step < WARMUP_STEP (int);
            Alert: Alert mode or not (bool);
            Dynamic_Phi: dynamically update the transition matrix or not (bool);
            Kernel_Type: the types of the topology-based label sampling kernel (str);
            Dynamic_Alpha: dynamically update the alpha vector or not (bool).
        @return:
            Y_inferred: new inferred labels (1D array: NUM_SAMPLES x 1);
            C: counting matrix (2D array: NUM_CLASSES x NUM_CLASSES).
        """
        # Get the adjacency matrix of the graph
        adj = graph.adjacency_matrix(scipy_fmt="csr")
        # Get Y_pred/Y_ssup dict
        z_dict, y_dict = self.get_labels_dict(Y_pred, Y_ssup)
        # Generate counting matrix
        NUM_CLASSES = int(Y_pred_sm.shape[1])
        C = self.generate_counting_matrix(Y_pred, Y_ssup, NUM_CLASSES)
        # Initialize the alpha (vector)
        # the initial value of alpha will be based on the #samples in train set if not isEqual.
        ALPHA = self.init_alpha(self.ALPHA, NUM_CLASSES, Y_ssup[train_mask], isEqual=True)
        # Convert to pytorch tensor
        adj = spcsr2tscoo(adj)
        node_degrees = torch.sparse.sum(adj, dim=0) if Kernel_Type == "degree_weighted_kernel" else None
        C = torch.FloatTensor(C)
        ALPHA = torch.FloatTensor(ALPHA)
        Y_pred_sm = torch.FloatTensor(Y_pred_sm)
        Y_ssup = torch.LongTensor(Y_ssup)
        TM_warmup = torch.FloatTensor(TM_warmup)
        TM_i = torch.clone(TM_warmup)

        # Setup the GPU
        if GPU >= 0:
            adj = adj.cuda()
            C = C.cuda()
            ALPHA = ALPHA.cuda()
            Y_pred_sm = Y_pred_sm.cuda()
            Y_ssup = Y_ssup.cuda()
            TM_warmup = TM_warmup.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
        # Setup the interval
        interval = int(WARMUP_STEP//100) if WARMUP_STEP >= 1000 else 10
        # Record the data if necessary
        writer = SummaryWriter(log_dir=os.path.join("runs", 'Logs_LInDT'))
        path = dirs + 'model_best.pth.tar' # The path of model parameters

        ent_best, acc_best = float('inf'), 0
        Y_uct_old = torch.clone(Y_ssup) 
        unct_rate_list = []  # the ratio of uncertain nodes in each epoch of inference
        for step in range(NUM_EPS):
            # Infer Z by topo-sampling based on corresponding TMs
            TM = TM_warmup if step < WARMUP_STEP else TM_i
            Y_infer, Y_uct, unct_rate = self.topo_sampling(Y_pred_sm, Y_ssup, TM, \
                                            Y_uct_old, adj, node_degrees, Kernel_Type)
            if GPU >= 0:
                Y_infer = Y_infer.cuda()
                Y_uct = Y_uct.cuda()
            if step > 1:
                unct_rate_list.append(np.round(unct_rate*100, 2))
            # Update the alpha vector
            if Dynamic_Alpha and step > 1:
                self.update_alpha(Y_uct, Y_uct_old, ALPHA)
            # Update the transition matrix
            if Dynamic_Phi:
                self.update_phi(Y_infer, C, z_dict, y_dict)
            TM_i = (C + ALPHA) / torch.sum(C + ALPHA, axis=1, keepdims=True)
            TM_i = TM_i.cuda() if GPU >= 0 else TM_i
            Y_uct_old = Y_uct
            # Tensorboard --logdir=./runs/Logs_LInDT --port 8999            
            if step % interval == 0:
                Y_infer_i = np.array([v for v in z_dict.values()])
                # Evaluate the inference by accuracy
                val_mask_cpu = val_mask.cpu() if GPU >= 0 else val_mask
                acc_i = accuracy_score(Y_gt[val_mask_cpu], Y_infer_i[val_mask_cpu])
                writer.add_scalar('Accuracy_Y_infer', acc_i, step)
                # Evaluate the prediction by entropy
                Y_pred_sm_val = Y_pred_sm[val_mask].cpu() if GPU >= 0 else Y_pred_sm[val_mask]
                total_entropy = entropy(Y_pred_sm_val.detach().numpy(), axis=1)
                normalized_entropy = total_entropy / np.log(Y_pred_sm.shape[1])
                nm_ent_avg = np.mean(normalized_entropy)
                writer.add_scalar('Entropy_Y_pred', nm_ent_avg, step)
                # Update the node classifier based on current z in given interval. 
                if (nm_ent_avg < ent_best or acc_i > acc_best) and not Alert:
                    print("Update the model in step {0}:".format(step))
                    acc_best = acc_i
                    ent_best = nm_ent_avg
                    Y_infer_i = torch.LongTensor(Y_infer_i)
                    Y_infer_i = Y_infer_i.cuda() if GPU >= 0 else Y_infer_i
                    train(model, optimizer, dirs, feat, Y_infer_i, \
                          train_mask, val_mask, NUM_RETRAIN)
                    _, Y_pred_sm = prediction(model, optimizer, path, graph, feat)
                print("Accuracy of inferred labels: ", acc_i)
                print("The normalized entropy: {:.2%}.".format(nm_ent_avg))

        # Get new infer label z
        Y_inferred = np.array([v for v in z_dict.values()])
        # Store the parameters
        C = tensor2array(C, GPU)  # array
        dump_pickle('../data/noisy_label/Y_C_RUN.pkl', [Y_inferred, C, unct_rate_list])
        writer.close()
        return Y_inferred, C, unct_rate_list
