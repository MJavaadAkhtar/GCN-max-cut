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

def calculateMinCut(adj_matrix, output, terminal1 = 0, terminal2 = 4):
    #output = (output.detach() >= 0.5) * 1

    # if output[terminal1] == output[terminal2]:
    #     return float("inf")
    #print(adj_matrix, output)
    loss = 0
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if (output[i] > 0.5 and output[j] < 0.5) or (output[i] < 0.5 and output[j] > 0.5) :
                loss+=adj_matrix[i][j]

    return loss

def partition_weight(adj, s):
    """
    Calculates the sum of weights of edges that are in different partitions.

    :param adj: Adjacency matrix of the graph.
    :param s: List indicating the partition of each edge (0 or 1).
    :return: Sum of weights of edges in different partitions.
    """
    s = np.array(s)
    partition_matrix = np.not_equal.outer(s, s).astype(int)
    weight = (adj * partition_matrix).sum() / 2
    return weight

def expected_partition_weight(adj, s):
    """
    Calculates the expected sum of weights of edges that are in different partitions,
    based on the probabilities in s.

    :param adj: Adjacency matrix of the graph.
    :param s: List indicating the probability of each edge being in a certain partition.
    :return: Expected sum of weights of edges in different partitions.
    """
    s = np.array(s)
    partition_matrix = np.outer(s, 1 - s) + np.outer(1 - s, s)
    expected_weight = (adj * partition_matrix).sum() / 2
    return expected_weight

def hyperParameters_old(n = 100, d = 3, p = None, graph_type = 'reg', number_epochs = int(1e5),
                    learning_rate = 1e-4, PROB_THRESHOLD = 0.5, tol = 1e-4, patience = 100):
    dim_embedding = int(np.sqrt(n))    # e.g. 10
    hidden_dim = int(dim_embedding/2)

    return n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim

def hyperParameters(n = 100, d = 3, p = None, graph_type = 'reg', number_epochs = int(1e5),
                    learning_rate = 1e-4, PROB_THRESHOLD = 0.5, tol = 1e-4, patience = 100):
    dim_embedding = int(np.sqrt(4096))    # e.g. 10
    hidden_dim = int(dim_embedding/2)

    return n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim


def calculateAllCut(q_torch, s):
    '''

    :param q_torch: The adjacent matrix of the graph
    :param s: The binary output from the neural network. s will be in form of [[prob1, prob2, ..., prob n], ...]
    :return: The calculated cut loss value
    '''
    if len(s) > 0:
        totalCuts = len(s[0])
        CutValue = 0
        for i in range(totalCuts):
            CutValue += partition_weight(q_torch, s[:,i])
        return CutValue/2
    return 0
def calculateCut(q_torch, s):
    return (q_torch * (1 - np.outer(s, s)) / 2).sum()

def assignTerminals(terminalNodes, bitstrings, probs):
    for node, partition in terminalNodes.items():
        bitstrings[node] = torch.zeros_like(probs[node])  # Reset probabilities
        bitstrings[node][partition] = 1  # Set probability to 1 for the designated partition

    return bitstrings

def partition_weight(adj, s):
    """
    Calculates the sum of weights of edges that are in different partitions.

    :param adj: Adjacency matrix of the graph.
    :param s: List indicating the partition of each edge (0 or 1).
    :return: Sum of weights of edges in different partitions.
    """
    s = np.array(s)
    partition_matrix = np.not_equal.outer(s, s).astype(int)
    weight = (adj * partition_matrix).sum() / 2
    return weight





def generate_terminal_nodes(graph, num_terminals, total_classes):
    """
    Randomly generate terminal node configurations for a given graph.
    """
    num_nodes = graph.number_of_nodes()
    terminal_nodes = {}

    # Randomly pick unique nodes to be terminals
    terminal_indices = random.sample(range(num_nodes), num_terminals)

    # Randomly assign a partition to each terminal node
    for node in terminal_indices:
        terminal_nodes[node] = random.randint(0, total_classes - 1)

    return terminal_nodes