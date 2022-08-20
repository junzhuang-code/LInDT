#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation
@topic: Train the node classifier on the train graph
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

import os
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from models_GNN import GNN
from load_data import LoadDataset
from utils import read_yaml_file, dump_pickle, split_masks, generate_random_noise_label, \
                    load_checkpoint, save_checkpoint, count_parameters, \
                    compute_accuracy, compute_f1_score, dist_pre_class, compute_entropy, \
                    compute_edge_homo_ratio
from torch.utils.tensorboard import SummaryWriter

# Define the arguments
parser = argparse.ArgumentParser(description="Read arguments for training.")
parser.add_argument("--config_file", type=str, default="model_lindt", help="The name of config files.")
parser.add_argument("--data_name", type=str, default="cora", help="The name of dataset.",
                    choices=["cora", "citeseer", "pubmed", "amazoncobuy", "coauthor", "reddit"])
parser.add_argument("--model_name", type=str, default="GCN", help="The name of GNNs.")  # GCN, SGC, GraphSAGE, GAT
parser.add_argument("--NOISE_RATIO", type=float, default=0.1, help="Noise ratio.")
parser.add_argument("--NUM_EPOCHS", type=int, default=200, help="The number of training epochs.")
parser.add_argument("--is_trainable", type=bool, default=True, help="Train the model or not.")
parser.add_argument("--GPU", type=int, default=-1, help="Input GPU device id or -1.")
parser.add_argument("--use_config_file", type=bool, default=True, help="Pass arguments from config files or not.")
args = parser.parse_known_args()[0]

# Read config files
config_file = read_yaml_file("../config", args.config_file)
if config_file and args.use_config_file:
    train_gnn_config = config_file["train_gnn"]
    args.data_name = train_gnn_config["data_name"]
    args.model_name = train_gnn_config["model_name"]
    args.NOISE_RATIO = train_gnn_config["NOISE_RATIO"]
    args.NUM_EPOCHS = train_gnn_config["NUM_EPOCHS"]
    args.is_trainable = train_gnn_config["is_trainable"]
    args.GPU = train_gnn_config["GPU"]

# Initialize the parameters
CUT_RATE = [0.1, 0.2]  # the split ratio of train/validation mask.
LR = 0.001
N_LAYERS = 2
N_HIDDEN = 200
DROPOUT = 0
WEIGHT_DECAY = 0
kwargs_dicts = {"GCN": [None, None, None],  # aggregator_type, n_filter, n_heads.
                "GraphSAGE": ["gcn", None, None],
                "SGC": [None, 2, None],
                "GAT": [None, None, 3]}


# Fitting the model --------------------
def train(model, optimizer, dirs, feat, label, train_mask, val_mask, n_epochs):
    """
    @topic: Fitting the GCNs
    @input: feature matrix, label, train/val masks, and #epochs.
    @return: train and save the model parameters.
    """
    loss_fn = torch.nn.CrossEntropyLoss()  # Define the loss function

    # Load checkpoint
    try:
        model, optimizer, start_epoch, best_acc = \
            load_checkpoint(dirs+'model_best.pth.tar', model, optimizer)
    except:
        print("Model parameter is not found.")
        start_epoch = 1
    # Train additional epochs if input #epochs < trained #epochs.
    if n_epochs <= start_epoch:
        n_epochs += start_epoch
    writer = SummaryWriter(log_dir=dirs, 
            comment="_time%s"%(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), purge_step=start_epoch)

    dur = []
    best_acc = 0
    for epoch in range(start_epoch, n_epochs):
        model.train()

        if epoch >= 3:
            t0 = time.time()

        # forward
        logits = model(feat)
        loss = loss_fn(logits[train_mask], label[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        # Compute the loss for evaluation
        with torch.no_grad():
            acc_train = compute_accuracy(logits, label, train_mask)
            loss_train = loss_fn(logits[train_mask], label[train_mask])
            acc_val = compute_accuracy(logits, label, val_mask)
            loss_val = loss_fn(logits[val_mask], label[val_mask])

        # Define the file name
        FileName = "Epoch{0}.pth.tar".format(epoch)
        # Delete previous existing parameter file (optional)
        if os.path.exists(dirs+"Epoch{0}.pth.tar".format(epoch-1)):
            os.remove(dirs+"Epoch{0}.pth.tar".format(epoch-1))
        # Update the parameters
        if acc_val > best_acc:
            best_acc = acc_val
            is_best = True
            # Save checkpoint
            save_checkpoint(
                    state = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_acc': best_acc,
                            'optimizer': optimizer.state_dict(),
                            }, \
                    is_best = is_best, \
                    directory = dirs, \
                    filename = FileName
                            )

        # Output the result
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} "
                      "| Train Accuracy {:.4f} | Val Accuracy {:.4f} "\
        .format(epoch, np.mean(dur), loss_train.item(), loss_val.item(), acc_train, acc_val))

        # Update SummaryWriter
        writer.add_scalar('Loss/train', loss_train.item(), epoch)
        writer.add_scalar('Loss/cross', loss_val.item(), epoch)
        writer.add_scalar('Accuracy/train', acc_train, epoch)
        writer.add_scalar('Accuracy/cross', acc_val, epoch)
        writer.flush()

    writer.close()


# Evaluation --------------------
def evaluation(model, optimizer, path, graph, feat, label, test_mask, cuda):
    """
    @topic: Evaluation on the given model
    @input: graph, feature matrix, label and its mask.
    @return: print out the test acc/f1/distributions/ent.
    """
    try:
        if not graph.number_of_nodes() == len(feat) == len(label) == len(test_mask):
            return "The length of adj/feat/label/test_mask is not equal!"
        model.eval()
        model, optimizer, start_epoch, best_acc = load_checkpoint(path, model, optimizer)
        model.g = graph  # update the graph
        logits = model(feat)
        if cuda:
            logits = logits.cpu()
            label = label.cpu()
            test_mask = test_mask.cpu()
        acc = compute_accuracy(logits, label, test_mask)  # the higher the better
        print("Best Testing Accuracy: {:.2%}".format(acc))
        f1 = compute_f1_score(logits, label, test_mask, avg='weighted')
        print("Test F1 Score: {:.2%}".format(f1))
        dist_true, dist_pred = dist_pre_class(logits, label, test_mask)
        print("The distributions of groundtruth classes: \n {0}".format(dist_true))
        print("The distributions of predicted classes: \n {0}".format(dist_pred))
        ent_mean, ent_std = compute_entropy(logits, test_mask)  # the lower the better
        print("The normalized entropy: {:.2%} (±{:.2%})".format(ent_mean, ent_std))
    except:
        return "Model parameter is not found."


# Prediction --------------------
def prediction(model, optimizer, path, graph, feat):
    """
    @topic: Generate predicted label with well-trained GCN model
    @input: graph, feature matrix.
    @return: predicted label (1D Tensor), probabilistic matrix (2D Tensor).
    """
    try:
        if graph.number_of_nodes() != len(feat):
            return "The length of adj/feat is not equal!"
        model.eval()
        model, optimizer, _, _ = load_checkpoint(path, model, optimizer)
        model.g = graph
        Y_pred_2d = model(feat)  # predicted label (2d)
        Y_pred_2d_softmax = torch.nn.functional.softmax(Y_pred_2d, dim=1)  # Normalize each row to sum=1
        Y_pred = torch.max(Y_pred_2d_softmax, dim=1)[1]  # predicted label (1d)
        return Y_pred, Y_pred_2d_softmax
    except:
        return "Model parameter is not found."


if __name__ == "__main__":
    # Load dataset
    data = LoadDataset(args.data_name)
    graph = data.load_data()
    feat, label = graph.ndata['feat'], graph.ndata['label']
    print("Class ID: ", set(label.numpy()))
    # Randomly split the train, validation, test mask by given cut rate
    train_mask, val_mask, test_mask = split_masks(label, cut_rate=CUT_RATE)
    # Generate noisy label
    Y_noisy = generate_random_noise_label(label, noisy_ratio=args.NOISE_RATIO, seed=0)
    Y_noisy = torch.LongTensor(Y_noisy)
    dump_pickle('../data/noisy_label/Y_gt_noisy_masks.pkl', \
                [label, Y_noisy, train_mask, val_mask, test_mask])
    # Display the variables
    print("""-------Data statistics-------'
          # Nodes: {0}
          # Edges: {1}
          # Features: {2}
          # Classes: {3}
          # Train samples: {4}
          # Val samples: {5}
          # Test samples: {6}
          # The edge homophily ratio: {7}%
          """.format(graph.number_of_nodes(), graph.number_of_edges(),\
                     feat.shape[1], len(torch.unique(label)), \
                      train_mask.int().sum().item(), \
                      val_mask.int().sum().item(), \
                      test_mask.int().sum().item(), \
                      np.round(compute_edge_homo_ratio(graph, label)*100, 2)
                      ))

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
        Y_noisy = Y_noisy.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # Initialize the node classifier
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
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters.')
    if cuda:  # if gpu is available
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Path for saving the parameters
    dirs = 'runs/{0}_{1}/'.format(args.data_name, args.model_name)
    path = dirs + 'model_best.pth.tar'

    # Training the model
    if args.is_trainable:
        train(model, optimizer, dirs, feat, Y_noisy, train_mask, val_mask, args.NUM_EPOCHS)
        # Save the weights of current model
        W0, W1 = model.w0.cpu(), model.w1.cpu()
        dump_pickle('../data/attacker_data/W_{0}_{1}.pkl'\
                    .format(args.data_name, args.model_name),
                    [W0.detach().numpy(), W1.detach().numpy()])

    # Evaluation
    print("Evaluation on stationary graphs!")
    evaluation(model, optimizer, path, graph, feat, label, test_mask, cuda)
    # Generate and save predicted labels
    Y_pred, Y_pred_sm = prediction(model, optimizer, path, graph, feat)  # prediction on all nodes
    if cuda:
        Y_pred, Y_pred_sm = Y_pred.cpu(), Y_pred_sm.cpu()
    dump_pickle('../data/noisy_label/Y_preds.pkl', [Y_pred, Y_pred_sm])
    print("Y_pred/Y_pred_sm.shape: ", Y_pred.shape, Y_pred_sm.shape)
