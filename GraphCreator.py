
import dgl
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
from dgl.nn.pytorch import GraphConv
from itertools import chain, islice, combinations
from networkx.algorithms.approximation.clique import maximum_independent_set as mis
from time import time
from networkx.algorithms.approximation.maxcut import one_exchange

# MacOS can have issues with MKL. For more details, see
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# fix seed to ensure consistent results
seed_value = 1
random.seed(seed_value)        # seed python RNG
np.random.seed(seed_value)     # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')


def CreateDummyFunction():
    test_graph = nx.Graph()
    test_graph.add_edges_from([(0,1, {"weight": 8, "capacity":8}),
                               (0,2, {"weight": 1, "capacity":1}),
                               (1,3, {"weight": 1, "capacity":1}),
                               (2,3, {"weight": 1, "capacity":1}),
                               (2,4, {"weight": 1, "capacity":1}),
                               (3,4, {"weight": 1, "capacity":1})])
    test_graph.order()
    return test_graph



def CreateDummyFunction2():
    test_graph = nx.Graph()
    test_graph.add_edges_from([(0,1, {"weight": 1, "capacity":5}),
                               (0,2, {"weight": 2, "capacity":2}),
                               (1,3, {"weight": 2, "capacity":2}),
                               (2,3, {"weight": 2, "capacity":2}),
                               (0,4, {"weight": 5, "capacity":5}),
                               (2,5, {"weight": 4, "capacity":4})])
    test_graph.order()
    return test_graph

def CreateGraph(n):
    # Create a new graph
    G = nx.Graph()

    # Add 100 nodes
    G.add_nodes_from(range(n))

    # Add random edges with weights
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.9:  # 90% chance to create an edge
                weight = random.randint(1, 100) #place 0 - 10
                G.add_edge(i, j, weight=weight, capacity=weight)

    # Example: Print the number of edges and some sample edges
    print("Number of edges:", G.number_of_edges())
    print("Sample edges:", list(G.edges(data=True))[:5])
    return G

def CreateGraph_random(n, random_value):
    # Create a new graph
    G = nx.Graph()

    # Add 100 nodes
    G.add_nodes_from(range(n))

    # Add random edges with weights
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < round(random.uniform(0.1,1),1):  # 50% chance to create an edge
                weight = random.randint(1, random_value)
                G.add_edge(i, j, weight=weight, capacity=weight)

    # Example: Print the number of edges and some sample edges
    print("Number of edges:", G.number_of_edges())
    print("Sample edges:", list(G.edges(data=True))[:5])
    return G


def CreateGraphEdgeChance(n,edgeNum = 0.5):
    # Create a new graph
    G = nx.Graph()

    # Add 100 nodes
    G.add_nodes_from(range(n))

    # Add random edges with weights
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edgeNum:  # 50% chance to create an edge
                weight = random.randint(0, 10)
                G.add_edge(i, j, weight=weight, capacity=weight)

    # Example: Print the number of edges and some sample edges
    print("Number of edges:", G.number_of_edges())
    print("Sample edges:", list(G.edges(data=True))[:5])
    return G

def FIndAC(graph):
    max_degree = max(dict(graph.degree()).values())
    A_initial = max_degree + 1  # A is set to be one more than the maximum degree
    C_initial = max_degree / 2  # C is set to half the maximum degree

    return A_initial, C_initial

def DrawGraph(graph):
    pos = nx.spring_layout(graph,seed=1)

    # Visualize graph
    options = {
        "font_size": 16,
        "node_size": 800,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 5,
        "width": 5,
    }
    nx.draw(graph, pos, with_labels=True, **options)

    labels = nx.get_edge_attributes(graph,'weight')
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)

def CreateDummyFunction(edges):
    test_graph = nx.Graph()
    test_graph.add_edges_from(edges)
    test_graph.order()
    return test_graph

#est_graph = nx.OrderedGraph([(0,1),(0,2),(1,3),(2,3),(2,4),(3,4)])
test_graph = CreateDummyFunction2()
pos = nx.spring_layout(test_graph,seed=1)

# Visualize graph
options = {
    "font_size": 16,
    "node_size": 800,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 5,
    "width": 5,
}
nx.draw(test_graph, pos, with_labels=True, **options)

labels = nx.get_edge_attributes(test_graph,'weight')
nx.draw_networkx_edge_labels(test_graph,pos,edge_labels=labels)
#%%
