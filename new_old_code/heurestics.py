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



# def find3WayCut(graph, terminal_lst):
#
#     cut_value_AB, partition_AB = nx.minimum_cut(graph, terminal_lst[0], terminal_lst[1])
#     if terminal_lst[2] in partition_AB[0]:
#         # here we are still using the whle graph, cut the graph and used that to find the cut
#         cut_value_BC, partition_BC = nx.minimum_cut(graph, terminal_lst[0], terminal_lst[2])
#
#         return cut_value_BC+cut_value_AB
#     else:
#         cut_value_BC, partition_BC = nx.minimum_cut(graph, terminal_lst[1], terminal_lst[2])
#         return cut_value_BC+cut_value_AB

