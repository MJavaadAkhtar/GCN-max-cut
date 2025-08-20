import torch
import random
import os
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import matplotlib
from collections import OrderedDict, defaultdict
from networkx.algorithms.flow import shortest_augmenting_path
from dgl.nn.pytorch import GraphConv
from itertools import chain, islice, combinations
from networkx.algorithms.approximation.clique import maximum_independent_set as mis
from time import time
from networkx.algorithms.approximation.maxcut import one_exchange
from itertools import permutations
import matplotlib.pyplot as plt
import dgl
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def open_file(filename):
    '''
    Example usage "with open('./datasetItem_exp2.pkl', 'rb') as inp"
    :param obj:
    :param filename:
    :return:
    '''
    with open(filename, 'rb') as inp:
        dst = pickle.load(inp)

    return dst

def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = len(nx_G.nodes)

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat

def gen_adj_matrix(nx_G):
    adj_dict = defaultdict(int)

    for(u,v) in nx_G.edges:
        adj_dict[(u, v)] = nx_G[u][v]['weight']
        adj_dict[(v, u)] = nx_G[u][v]['weight']

    for u in nx_G.nodes:
        for i in nx_G.nodes:
            if not adj_dict[(u, i)]:
                adj_dict[(u, i)] = 0

    return adj_dict


# class GCNSoftmax(nn.Module):
#     def __init__(self, in_feats, hidden_size, num_classes, dropout, device):
#         super(GCNSoftmax, self).__init__()
#         self.dropout_frac = dropout
#         self.conv1 = GraphConv(in_feats, hidden_size).to(device)
#         self.conv2 = GraphConv(hidden_size, num_classes).to(device)
#
#     def forward(self, g, inputs):
#         # Basic forward pass
#         h = self.conv1(g, inputs)
#         h = F.relu(h)
#         h = F.dropout(h, p=self.dropout_frac, training=self.training)
#         h = self.conv2(g, h)
#         h = F.softmax(h, dim=1)  # Apply softmax over the classes dimension
#         # h = F.sigmoid(h)

        # return h

def get_gnn(n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Generate GNN instance with specified structure. Creates GNN, retrieves embedding layer,
    and instantiates ADAM optimizer given those.

    Input:
        n_nodes: Problem size (number of nodes in graph)
        gnn_hypers: Hyperparameters relevant to GNN structure
        opt_params: Hyperparameters relevant to ADAM optimizer
        torch_device: Whether to load pytorch variables onto CPU or GPU
        torch_dtype: Datatype to use for pytorch variables
    Output:
        net: GNN instance
        embed: Embedding layer to use as input to GNN
        optimizer: ADAM optimizer instance
    """
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']

    # instantiate the GNN
    net = GCNSoftmax(dim_embedding, hidden_dim, number_classes, dropout, torch_device)
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, embed, optimizer

def LoadNeuralModel(model, gnn_hypers, torch_device, model_location):
    checkpoint = torch.load(model_location)
    #model = checkpoint['model']
    # instantiate the GNN
    # model.load_state_dict(torch.load(model_location))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, checkpoint['inputs']

def LoadNeuralModelRL(model, gnn_hypers, torch_device, model_location, policy_mod):
    checkpoint = torch.load(model_location)
    #model = checkpoint['model']
    # instantiate the GNN
    # model.load_state_dict(torch.load(model_location))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    policyNet.load_state_dict(checkpoint['model_policy'])
    policyNet.eval()

    return model, checkpoint['inputs'], policyNet