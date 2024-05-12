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

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
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

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

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
def hyperParameters(n = 100, d = 3, p = None, graph_type = 'reg', number_epochs = int(1e5),
                    learning_rate = 1e-4, PROB_THRESHOLD = 0.5, tol = 1e-4, patience = 100):
    dim_embedding = int(np.sqrt(4096))    # e.g. 10
    hidden_dim = int(dim_embedding/2)

    return n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim
def FIndAC(graph):
    max_degree = max(dict(graph.degree()).values())
    A_initial = max_degree + 1  # A is set to be one more than the maximum degree
    C_initial = max_degree / 2  # C is set to half the maximum degree

    return A_initial, C_initial

def LoadData(fileName = './datasetItem_exp2.pkl', mode = 'rb'):
    with open(fileName, mode) as inp:
        dst = pickle.load(inp)
    return dst


class GCNSoftmax(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout, device):
        super(GCNSoftmax, self).__init__()
        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, num_classes).to(device)

    def forward(self, g, inputs, terminals=None):
        # Basic forward pass
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_frac, training=self.training)
        h = self.conv2(g, h)
        h = F.softmax(h, dim=1)  # Apply softmax over the classes dimension

        # Handling terminals if provided
        if terminals is not None:
            # Clone h to avoid in-place operations
            h_clone = h.clone()
            # Ensure terminals is a dictionary with node index as keys and target class as values
            for node_index, class_index in terminals.items():
                # Create a one-hot encoded vector for the terminal node's class
                terminal_vector = torch.zeros(h.shape[1], device=h.device)
                terminal_vector[class_index] = 1
                # Use indexing that does not modify the original tensor in-place
                h_clone[node_index] = terminal_vector
            h = h_clone

        return h




def run_gnn_training_5way(dataset, net, optimizer, number_epochs, tol, patience, loss_func, dim_embedding, total_classes=3, save_directory=None, torch_dtype = TORCH_DTYPE, torch_device = TORCH_DEVICE):
    """
    Train a GCN model with early stopping.
    """
    # loss for a whole epoch
    prev_loss = float('inf')  # Set initial loss to infinity for comparison
    count = 0  # Patience counter
    best_loss = float('inf')  # Initialize best loss to infinity
    best_model_state = None  # Placeholder for the best model state
    loss_list = []
    epochList = []

    t_gnn_start = time()

    # contains information regarding all terminal nodes for the dataset
    terminal_configs = {}
    epochCount = 0

    for epoch in range(number_epochs):
        cumulative_loss = 0.0  # Reset cumulative loss for each epoch

        for key, (dgl_graph, adjacency_matrix,graph, A, C) in dataset.items():
            epochCount +=1
            # create random terminal node for each dataset graph
            if key not in terminal_configs:
                terminal_configs[key] = generate_terminal_nodes(dgl_graph, total_classes, total_classes)
            terminal_nodes = terminal_configs[key]

            # test why we are using randn?
            # inputs = torch.randn(graph.number_of_nodes(), dim_embedding, device=torch_device, dtype=torch_dtype)

            embed = nn.Embedding(graph.number_of_nodes(), dim_embedding)
            embed = embed.type(torch_dtype).to(torch_device)
            inputs = embed.weight

            # Ensure model is in training mode
            net.train()

            # Pass the graph and the input features to the model
            logits = net(dgl_graph, inputs, terminal_nodes)

            # Compute the loss
            loss = loss_func(logits, adjacency_matrix, A, C)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update cumulative loss
            cumulative_loss += loss.item()



            # Check for early stopping
            if epoch > 0 and (cumulative_loss > prev_loss or abs(prev_loss - cumulative_loss) <= tol):
                count += 1
                if count >= patience: # play around with patience value, try lower one
                    print(f'Stopping early at epoch {epoch}')
                    break
            else:
                count = 0  # Reset patience counter if loss decreases

            # Update best model
            if cumulative_loss < best_loss:
                best_loss = cumulative_loss
                best_model_state = net.state_dict()  # Save the best model state

        loss_list.append(loss)

        # Early stopping break from the outer loop
        if count >= patience:
            break

        prev_loss = cumulative_loss  # Update previous loss

        if epoch % 100 == 0:  # Adjust printing frequency as needed
            print(f'Epoch: {epoch}, Cumulative Loss: {cumulative_loss}')

            if save_directory != None:
                checkpoint = {
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'inputs':inputs}
                torch.save(checkpoint, save_directory)

    t_gnn = time() - t_gnn_start

    # Load the best model state
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    print(f'GNN training took {round(t_gnn, 3)} seconds.')
    print(f'Best cumulative loss: {best_loss}')

    if save_directory != None:
        checkpoint = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'inputs':inputs}
        torch.save(checkpoint, save_directory)

    return net, best_loss, epoch, inputs, loss_list

# Construct graph to learn on
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

def run_gnn_training(dataset, net, optimizer, number_epochs, tol, patience, loss_func, dim_embedding, total_classes=3, save_directory=None, torch_dtype = TORCH_DTYPE, torch_device = TORCH_DEVICE):
    """
    Train a GCN model with early stopping.
    """
    # loss for a whole epoch
    prev_loss = float('inf')  # Set initial loss to infinity for comparison
    prev_cummulative_loss = float('inf')
    cummulativeCount = 0
    count = 0  # Patience counter
    best_loss = float('inf')  # Initialize best loss to infinity
    best_model_state = None  # Placeholder for the best model state
    loss_list = []
    epochList = []

    t_gnn_start = time()

    # contains information regarding all terminal nodes for the dataset
    terminal_configs = {}
    epochCount = 0
    cumulative_loss = 0

    A = nn.Parameter(torch.tensor([65.0]))
    C = nn.Parameter(torch.tensor([32.5]))

    for epoch in range(number_epochs):


        cumulative_loss = 0.0  # Reset cumulative loss for each epoch

        for key, (dgl_graph, adjacency_matrix,graph, _, _) in dataset.items():
            epochCount +=1
            # create random terminal node for each dataset graph
            if key not in terminal_configs:
                terminal_configs[key] = generate_terminal_nodes(dgl_graph, total_classes, total_classes)
            terminal_nodes = terminal_configs[key]

            # test why we are using randn?
            # inputs = torch.randn(graph.number_of_nodes(), dim_embedding, device=torch_device, dtype=torch_dtype)

            embed = nn.Embedding(graph.number_of_nodes(), dim_embedding)
            embed = embed.type(torch_dtype).to(torch_device)
            inputs = embed.weight

            # Ensure model is in training mode
            net.train()

            # Pass the graph and the input features to the model
            logits = net(dgl_graph, inputs, terminal_nodes)

            # Compute the loss
            loss = loss_func(logits, adjacency_matrix, A, C)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update cumulative loss
            cumulative_loss += loss.item()



            # Check for early stopping
            if epoch > 0 and (cumulative_loss > prev_loss or abs(prev_loss - cumulative_loss) <= tol):
                count += 1
                if count >= patience: # play around with patience value, try lower one
                    print(f'Stopping early at epoch {epoch}')
                    break
            else:
                count = 0  # Reset patience counter if loss decreases

            # Update best model
            if cumulative_loss < best_loss:
                best_loss = cumulative_loss
                best_model_state = net.state_dict()  # Save the best model state

        loss_list.append(loss)

        # Early stopping break from the outer loop
        if count >= patience:
            break

        prev_loss = cumulative_loss  # Update previous loss

        if epoch % 100 == 0:  # Adjust printing frequency as needed
            print(f'Epoch: {epoch}, Cumulative Loss: {cumulative_loss}')

            if save_directory != None:
                checkpoint = {
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lossList':loss_list,
                    'inputs':inputs}
                torch.save(checkpoint, './epoch'+str(epoch)+'loss'+str(cumulative_loss)+ save_directory)




    t_gnn = time() - t_gnn_start

    # Load the best model state
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    print(f'GNN training took {round(t_gnn, 3)} seconds.')
    print(f'Best cumulative loss: {best_loss}')

    if save_directory != None:
        checkpoint = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossList':loss_list,
            'inputs':inputs}
        torch.save(checkpoint, './final_'+save_directory)

    return net, best_loss, epoch, inputs, loss_list

def printCombo(orig):
    # Original dictionary
    input_dict = orig

    # Generate all permutations of the dictionary values
    value_permutations = list(permutations(input_dict.values()))

    # Create a list of dictionaries from the permutations
    permuted_dicts = [{key: value for key, value in zip(input_dict.keys(), perm)} for perm in value_permutations]

    return permuted_dicts

def GetOptimalNetValue(net, dgl_graph, inp, q_torch, terminal_dict):
    net.eval()
    best_loss = float('inf')

    if (dgl_graph.number_of_nodes() < 30):
        inp = torch.ones((dgl_graph.number_of_nodes(), 30))

    # find all potential combination of terminal nodes with respective indices

    perm_items = printCombo(terminal_dict)
    for i in perm_items:
        probs = net(dgl_graph, inp, i)
        binary_partitions = (probs >= 0.5).float()
        cut_value_item = calculateAllCut(q_torch, binary_partitions)
        if cut_value_item < best_loss:
            best_loss = cut_value_item
    return best_loss

def calculate_HA_vectorized(s):
    """
    Vectorized calculation of HA.
    :param s: A binary matrix of size |V| x |K| where s[i][j] is 1 if vertex i is in partition j.
    :return: The HA value.
    """
    # HA = ∑v∈V(∑k∈K(sv,k)−1)^2
    HA = torch.sum((torch.sum(s, axis=1) - 1) ** 2)
    return HA

def calculate_HC_vectorized_old(s, adjacency_matrix):
    """
    Vectorized calculation of HC.
    :param s: A binary matrix of size |V| x |K|.
    :param adjacency_matrix: A matrix representing the graph where the value at [i][j] is the weight of the edge between i and j.
    :return: The HC value.
    """
    # HC = ∑(u,v)∈E(1−∑k∈K(su,k*sv,k))*adjacency_matrix[u,v]
    K = s.shape[1]
    # Outer product to find pairs of vertices in the same partition and then weight by the adjacency matrix
    prod = adjacency_matrix * (1 - s @ s.T)
    HC = torch.sum(prod)
    return HC

def calculate_HC_vectorized(s, adjacency_matrix):
    """
    Vectorized calculation of HC for soft partitioning.
    :param s: A probability matrix of size |V| x |K| where s[i][j] is the probability of vertex i being in partition j.
    :param adjacency_matrix: A matrix representing the graph where the value at [i][j] is the weight of the edge between i and j.
    :return: The HC value.
    """
    # Initialize HC to 0
    HC = 0

    # Iterate over each partition to calculate its contribution to HC
    for k in range(s.shape[1]):
        # Compute the probability matrix for partition k
        partition_prob_matrix = s[:, k].unsqueeze(1) * s[:, k].unsqueeze(0)

        # Compute the contribution to HC for partition k
        HC_k =adjacency_matrix * (1 - partition_prob_matrix)
        # Sum up the contributions for partition k
        HC += torch.sum(HC_k, dim=(0, 1))

    # Since we've summed up the partition contributions twice (due to symmetry), divide by 2
    HC = HC / 2

    return HC

def Loss(s, adjacency_matrix,  A=1, C=1):
    HA = calculate_HA_vectorized(s)
    HC = calculate_HC_vectorized(s, adjacency_matrix)

    return A * HA + C * HC
def GenerateGraphData(numberOfNodes, numberOfGraph):
    datasetItem = {}
    for i in range(numberOfGraph): #1000

        graph = CreateGraph(numberOfNodes) #30
        graph_dgl = dgl.from_networkx(nx_graph=graph)
        graph_dgl = graph_dgl.to(TORCH_DEVICE)
        q_torch = qubo_dict_to_torch(graph, gen_adj_matrix(graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

        datasetItem[i] = [graph_dgl, q_torch, graph]

    datasetItem_all = {}
    for key, (dgl_graph, adjacency_matrix,graph) in datasetItem.items():
        A, C = FIndAC(graph)
    datasetItem_all[key] = [dgl_graph, adjacency_matrix, graph, A, C]

    return datasetItem_all

def GenerateGraphDataRandom(startNodes, endNodes, numberOfGraph):
    datasetItem = {}
    for i in range(numberOfGraph): #1000

        graph = CreateGraph(random.randint(startNodes, endNodes)) #30
        graph_dgl = dgl.from_networkx(nx_graph=graph)
        graph_dgl = graph_dgl.to(TORCH_DEVICE)
        q_torch = qubo_dict_to_torch(graph, gen_adj_matrix(graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)
        print("Graph Created for index:"+ str(i))
        datasetItem[i] = [graph_dgl, q_torch, graph]

    datasetItem_all = {}
    for key, (dgl_graph, adjacency_matrix,graph) in datasetItem.items():
        A, C = FIndAC(graph)
        datasetItem_all[key] = [dgl_graph, adjacency_matrix, graph, A, C]

    return datasetItem_all

def train1():
    n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim = hyperParameters(n=4096,patience=15)

    # Establish pytorch GNN + optimizer
    opt_params = {'lr': learning_rate}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.0,
        'number_classes': 5,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience,
        'nodes':n
    }
    datasetItem = LoadData('./TrainingSet_Random_30_100_graph')
    #print(datasetItem)
    # datasetItem_all = {}
    # for key, (dgl_graph, adjacency_matrix,graph) in datasetItem.items():
    #     A, C = FIndAC(graph)
    #     datasetItem_all[key] = [dgl_graph, adjacency_matrix, graph, A, C]

    print(len(datasetItem), datasetItem[0][3])

    net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)


    # print(datasetItem[1][2].nodes)
    # # Visualize graph
    # pos = nx.kamada_kawai_layout(datasetItem[1][2])
    # nx.draw(datasetItem[1][2], pos, with_labels=True, node_color=[[.7, .7, .7]])

    trained_net, bestLost, epoch, inp, lossList=  run_gnn_training(
        datasetItem, net, optimizer, int(500),
        gnn_hypers['tolerance'], gnn_hypers['patience'], Loss,gnn_hypers['dim_embedding'], gnn_hypers['number_classes'], '_small_exp_5way.pth')

    return trained_net, bestLost, epoch, inp, lossList

# datasets = GenerateGraphDataRandom(2000, 3000, 100)
# save_object(datasets, 'Random_2000_3000_graph')


trained_net, bestLost, epoch, inp, lossList = train1()

#%%

#%%
