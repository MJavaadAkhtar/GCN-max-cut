
"""
DEPRECATED: This file is deprecated and will be removed in a future version.
Please use TrainingNeural.py instead, which provides the same functionality 
with a cleaner, more modular design.

Migration guide:
- Replace get_gnn() calls with setup_model_and_optimizer() using TrainingConfig
- Replace train1() with train_from_pickle() or train_legacy_wrapper()  
- Replace run_gnn_training2() with train_model()
- Use TrainingConfig dataclass instead of hyperParameters()
- Use find_ac_parameters() instead of FIndAC()

All functions from this file are available as legacy wrappers in TrainingNeural.py
"""

import warnings
warnings.warn(
    "TrainingNeural_load.py is deprecated. Use TrainingNeural.py instead. "
    "See deprecation notice at the top of this file for migration guidance.",
    DeprecationWarning,
    stacklevel=2
)

from python.commons import *
from dgl.nn.pytorch import GATConv, EdgeConv

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
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

import torch

def partition_weight2(adj, s):
    """
    Calculates the sum of weights of edges that are in different partitions.

    :param adj: Adjacency matrix of the graph as a PyTorch tensor.
    :param s: Tensor indicating the partition of each node (0 or 1).
    :return: Sum of weights of edges in different partitions.
    """
    # Ensure s is a tensor
    # s = torch.tensor(s, dtype=torch.float32)

    # Compute outer difference to create partition matrix
    s = s.unsqueeze(0)  # Convert s to a row vector
    t = s.t()           # Transpose s to a column vector
    partition_matrix = (s != t).float()  # Compute outer product and convert boolean to float

    # Calculate the weight of edges between different partitions
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
            CutValue += partition_weight2(q_torch, s[:,i])
        return CutValue/2
    return 0

def hyperParameters(n = 80, d = 3, p = None, graph_type = 'reg', number_epochs = int(1e5),
                    learning_rate = 1e-4, PROB_THRESHOLD = 0.5, tol = 1e-4, patience = 100):
    dim_embedding = n #int(np.sqrt(4096))    # e.g. 10, used to be the one before # used to be n
    hidden_dim = int(dim_embedding/2)

    return n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim
def FIndAC(graph):
    max_degree = max(dict(graph.degree()).values())
    A_initial = max_degree + 1  # A is set to be one more than the maximum degree
    C_initial = max_degree / 2  # C is set to half the maximum degree

    return A_initial, C_initial

def generate_unique_random_numbers(N):
    if N < 2:
        raise ValueError("N must be at least 2 to generate 3 unique random numbers.")

    # Generate 3 unique random numbers
    random_numbers = random.sample(range(N + 1), 3)

    return random_numbers

N = 10  # Set the value of N
unique_random_numbers = generate_unique_random_numbers(N)
print("Three unique random numbers between 0 and", N, ":", unique_random_numbers)

def extend_matrix_to_N(matrix, N):
    original_size = matrix.shape[0]

    if N < original_size:
        raise ValueError("N should be greater than or equal to the original matrix size.")

    extended_matrix = np.zeros((N, N))
    extended_matrix[:original_size, :original_size] = matrix

    return extended_matrix





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

def terminal_independence_penalty(s, terminal_nodes):
    """
    Calculate a penalty that enforces each terminal node to be in a distinct partition.
    :param s: A probability matrix of size |V| x |K| where s[i][j] is the probability of vertex i being in partition j.
    :param terminal_nodes: A list of indices for terminal nodes.
    :return: The penalty term.
    """
    penalty = 0
    num_terminals = len(terminal_nodes)
    # Compare each pair of terminal nodes
    for i in range(num_terminals):
        for j in range(i + 1, num_terminals):
            # Calculate the dot product of the probability vectors for the two terminals
            dot_product = torch.dot(s[terminal_nodes[i]], s[terminal_nodes[j]])
            # Penalize the similarity in their partition assignments (dot product should be close to 0)
            penalty += dot_product
    return penalty

def calculate_HA_vectorized(s):
    """
    Vectorized calculation of HA.
    :param s: A binary matrix of size |V| x |K| where s[i][j] is 1 if vertex i is in partition j.
    :return: The HA value.
    """
    # HA = ∑v∈V(∑k∈K(sv,k)−1)^2
    HA = torch.sum((torch.sum(s, axis=1) - 1) ** 2)
    return HA

def calculate_HC_min_cut_intra_inter(s, adjacency_matrix):
    """
    Vectorized calculation of HC to minimize cut size.
    :param s: A probability matrix of size |V| x |K| where s[i][j] is the probability of vertex i being in partition j.
    :param adjacency_matrix: A matrix representing the graph where the value at [i][j] is the weight of the edge between i and j.
    :return: The HC value focusing on minimizing edge weights between partitions.
    """
    HC = 0
    K = s.shape[1]
    for k in range(K):
        for l in range(k + 1, K):
            partition_k = s[:, k].unsqueeze(1) * s[:, k].unsqueeze(0)  # Probability node pair both in partition k
            partition_l = s[:, l].unsqueeze(1) * s[:, l].unsqueeze(0)  # Probability node pair both in partition l
            # Edges between partitions k and l
            inter_partition_edges = adjacency_matrix * (partition_k + partition_l)
            HC += torch.sum(inter_partition_edges)

    return HC

def calculate_HC_min_cut_intra_inter2(s, adjacency_matrix):
    """
    Vectorized calculation of HC to minimize cut size.
    :param s: A probability matrix of size |V| x |K| where s[i][j] is the probability of vertex i being in partition j.
    :param adjacency_matrix: A matrix representing the graph where the value at [i][j] is the weight of the edge between i and j.
    :return: The HC value focusing on minimizing edge weights between partitions.
    """
    HC = 0
    K = s.shape[1]
    for k in range(K):
        for l in range(k + 1, K):
            partition_k = s[:, k].unsqueeze(1) * s[:, k].unsqueeze(0)  # Probability node pair both in partition k
            partition_l = s[:, l].unsqueeze(1) * s[:, l].unsqueeze(0)  # Probability node pair both in partition l
            # Edges between partitions k and l
            inter_partition_edges = adjacency_matrix * (partition_k + partition_l)
            HC += torch.sum(inter_partition_edges)

    return HC

def calculate_HC_min_cut_new(s, adjacency_matrix):
    """
    Differentiable calculation of HC for minimizing edge weights between different partitions.
    :param s: A probability matrix of size |V| x |K| where s[i][j] is the probability of vertex i being in partition j.
    :param adjacency_matrix: A matrix representing the graph where the value at [i][j] is the weight of the edge between i and j.
    :return: The HC value, focusing on minimizing edge weights between partitions.
    """
    K = s.shape[1]
    V = s.shape[0]

    # Create a full partition matrix indicating the likelihood of each node pair being in the same partition
    partition_matrix = torch.matmul(s, s.T)

    # Calculate the complement matrix, which indicates the likelihood of node pairs being in different partitions
    complement_matrix = 1 - partition_matrix

    # Apply adjacency matrix to only consider actual edges and their weights
    inter_partition_edges = adjacency_matrix * complement_matrix

    # Summing up all contributions for edges between different partitions
    HC = torch.sum(inter_partition_edges)

    return HC

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
import torch

def min_cut_loss(s, adjacency_matrix):
    """
    Compute a differentiable min-cut loss for a graph given node partition probabilities.

    :param s: A probability matrix of size |V| x |K| where s[i][j] is the probability of vertex i being in partition j.
    :param adjacency_matrix: A matrix representing the graph where the value at [i][j] is the weight of the edge between i and j.
    :return: The expected min-cut value, computed as a differentiable loss.
    """
    V = s.size(0)  # Number of nodes
    K = s.size(1)  # Number of partitions

    # Ensure the partition matrix s sums to 1 over partitions
    s = torch.softmax(s, dim=1)

    # Compute the expected weight of edges within each partition
    intra_partition_cut = torch.zeros((K, K), dtype=torch.float32)
    for k in range(K):
        for l in range(k + 1, K):
            # Probability that a node pair (i, j) is split between partitions k and l
            partition_k = s[:, k].unsqueeze(1)  # Shape: V x 1
            partition_l = s[:, l].unsqueeze(0)  # Shape: 1 x V

            # Compute the expected weight of the cut edges between partitions k and l
            cut_weight = adjacency_matrix * (partition_k @ partition_l)
            intra_partition_cut[k, l] = torch.sum(cut_weight)

    # Sum up all contributions to get the total expected min-cut value
    total_cut_weight = torch.sum(intra_partition_cut)

    return total_cut_weight

import torch

# def min_cut_loss(s, adjacency_matrix):
#     """
#     Compute a differentiable min-cut loss for a graph given node partition probabilities.
#
#     :param s: A probability matrix of size |V| x |K| where s[i][j] is the probability of vertex i being in partition j.
#     :param adjacency_matrix: A matrix representing the graph where the value at [i][j] is the weight of the edge between i and j.
#     :return: The expected min-cut value, computed as a differentiable loss.
#     """
#     V = s.size(0)  # Number of nodes
#     K = s.size(1)  # Number of partitions
#
#     # Ensure the partition matrix s sums to 1 over partitions
#     # s = torch.softmax(s, dim=1)
#
#     # Compute the expected weight of cut edges between each pair of partitions
#     total_cut_weight = 0
#     for k in range(K):
#         for l in range(k + 1, K):
#             # Probability that a node pair (i, j) is split between partitions k and l
#             partition_k = s[:, k].unsqueeze(1)  # Shape: V x 1
#             partition_l = s[:, l].unsqueeze(0)  # Shape: 1 x V
#
#             # Compute the expected weight of the cut edges between partitions k and l
#             cut_weight = adjacency_matrix * (partition_k @ partition_l)
#             total_cut_weight += torch.sum(cut_weight)
#
#     return total_cut_weight


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

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
def swap_graph_nodes(graph, mapping):
    """
    Swap the nodes of a NetworkX graph according to a given mapping dictionary, ensuring mutual swaps.

    :param graph: NetworkX graph
    :param mapping: dictionary where keys and values represent nodes to be swapped
    :return: None; the graph is modified in place
    """
    # Create a temporary mapping to intermediate labels
    temp_mapping = {old_label: max(graph.nodes) + 1 + i for i, old_label in enumerate(mapping)}

    # First step: move all nodes to temporary labels
    nx.relabel_nodes(graph, mapping=temp_mapping, copy=False)

    # Create the reverse of the initial mapping to complete the swap
    reverse_mapping = {value: key for key, value in mapping.items()}

    # Second step: from temporary labels to final labels
    nx.relabel_nodes(graph, mapping={temp_mapping[old]: reverse_mapping[old] for old in mapping}, copy=False)

def swap_all_terminal_nodes(graph_list):
    graph_listV2 = {}
    for key, (dgl_graph, adjacency_matrix,graph, terminals) in graph_list.items():
        graph_listV2[key] = (dgl_graph, adjacency_matrix,graph, terminals)

    return graph_listV2

def extend_matrix(matrix, N):
    original_size = matrix.shape[0]

    if N < original_size:
        raise ValueError("N should be greater than or equal to the original matrix size.")

    extended_matrix = np.zeros((N, N))
    extended_matrix[:original_size, :original_size] = matrix

    return extended_matrix

def extend_matrix_torch(matrix, N, torch_dtype=None, torch_device=None):
    original_size = matrix.shape[0]

    if N < original_size:
        raise ValueError("N should be greater than or equal to the original matrix size.")

    extended_matrix = torch.zeros(original_size, N)
    extended_matrix[:original_size, :original_size] = matrix

    if torch_dtype is not None:
        extended_matrix = extended_matrix.type(torch_dtype)

    if torch_device is not None:
        extended_matrix = extended_matrix.to(torch_device)

    return extended_matrix

def extend_matrix_torch_2(matrix, N, torch_dtype=None, torch_device=None):
    original_size = matrix.shape[0]

    if N < original_size:
        raise ValueError("N should be greater than or equal to the original matrix size.")

    # Use the dtype and device from the original matrix if not provided
    torch_dtype = torch_dtype or matrix.dtype
    torch_device = torch_device or matrix.device

    # Initialize an empty matrix with the specified size, dtype, and device
    extended_matrix = torch.empty((original_size, N), dtype=torch_dtype, device=torch_device)

    # Copy the original matrix into the extended matrix
    extended_matrix[:, :original_size] = matrix

    # Zero out the remaining columns
    if N > original_size:
        extended_matrix[:, original_size:] = 0

    return extended_matrix

def createGraphFromFolder(all_graphs, all_terminals, max_nodes):

    # Example usage
    # directory =  dst # Replace this with your directory path
    # all_graphs, all_terminals = process_all_files(directory)

    datasetItem = {}
    i = 0
    skipped = 0
    try:
        # Print out some details about the graphs (optional)
        for filename, graph in all_graphs.items():
            # print(f"Graph for {filename}: Nodes = {graph.nodes()}, Edges = {graph.edges(data=True)}")
            # print(f"Terminals for {filename}: {all_terminals[filename]}")
            terminals = all_terminals[filename]
            if 0 not in terminals and 1 not in terminals and 2 not in terminals:
                swap_graph_nodes(graph, {terminals[0]:0, terminals[1]:1, terminals[2]:2, 0:terminals[0], 1:terminals[1], 2:terminals[2]})
            elif 0 not in terminals and 1 not in terminals and 2 in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {terminals[1]:0, terminals[2]:1,  0:terminals[1], 1:terminals[2]})
            elif 0 not in terminals and 1  in terminals and 2 not in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {terminals[1]:0, terminals[2]:2,  0:terminals[1], 2:terminals[2]})
            elif 0  in terminals and 1 not in terminals and 2 not in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {terminals[1]:1, terminals[2]:2,  1:terminals[1], 2:terminals[2]})
            else:
                skipped +=1
                continue
            graph_dgl = dgl.from_networkx(nx_graph=graph)
            graph_dgl = graph_dgl.to(TORCH_DEVICE)

            q_torch = qubo_dict_to_torch(graph, gen_adj_matrix(graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

            # full_matrix = extend_matrix_torch(q_torch, max_nodes, torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

            # datasetItem[i] = [graph_dgl, q_torch, graph, all_terminals[filename]]
            datasetItem[i] = [graph_dgl, q_torch, graph, [0,1,2]]
            i+=1

    except:
        print(i, filename, terminals, graph.edges())

    print(skipped)
    return datasetItem

def createGraphFromFolder_full(all_graphs, all_terminals, max_nodes):

    # Example usage
    # directory =  dst # Replace this with your directory path
    # all_graphs, all_terminals = process_all_files(directory)

    datasetItem = {}
    i = 0
    skipped = 0
    try:
        # Print out some details about the graphs (optional)
        for filename, graph in all_graphs.items():
            # print(f"Graph for {filename}: Nodes = {graph.nodes()}, Edges = {graph.edges(data=True)}")
            # print(f"Terminals for {filename}: {all_terminals[filename]}")
            terminals = all_terminals[filename]
            if 0 not in terminals and 1 not in terminals and 2 not in terminals:
                swap_graph_nodes(graph, {terminals[0]:0, terminals[1]:1, terminals[2]:2, 0:terminals[0], 1:terminals[1], 2:terminals[2]})
            elif 0 not in terminals and 1 not in terminals and 2 in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {terminals[1]:0, terminals[2]:1,  0:terminals[1], 1:terminals[2]})
            elif 0 not in terminals and 1  in terminals and 2 not in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {terminals[1]:0, terminals[2]:2,  0:terminals[1], 2:terminals[2]})
            elif 0  in terminals and 1 not in terminals and 2 not in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {terminals[1]:1, terminals[2]:2,  1:terminals[1], 2:terminals[2]})
            else:
                skipped +=1
                continue
            print("Terminal swapped ", i)
            graph_dgl = dgl.from_networkx(nx_graph=graph)
            graph_dgl = graph_dgl.to(TORCH_DEVICE)

            q_torch = qubo_dict_to_torch(graph, gen_adj_matrix(graph), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

            full_matrix = extend_matrix_torch_2(q_torch, max_nodes, torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

            # datasetItem[i] = [graph_dgl, q_torch, graph, all_terminals[filename]]
            datasetItem[i] = [graph_dgl, full_matrix, graph, [0,1,2]]
            i+=1

            print("graph finished: ", i)

    except:
        print("Exception Occured ", i, filename, terminals)

    print("skipped Items:", skipped)
    return datasetItem

def train1(modelName, filename = './testData/nx_generated_graph_n80_d3_t200.pkl', n = 80):
    n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim = hyperParameters(learning_rate=0.001, n=n,patience=20)

    # Establish pytorch GNN + optimizer
    opt_params = {'lr': learning_rate}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.0,
        'number_classes': 3,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience,
        'nodes':n
    }
    datasetItem = open_file(filename)

    net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)
    net, inputs =LoadNeuralModel(net, gnn_hypers, TORCH_DEVICE, './epoch100loss-468174.0_10000MaxwayCut_LossExp8_loss.pth')

    trained_net, bestLost, epoch, inp, lossList= run_gnn_training2(
        datasetItem, net, optimizer, int(1000),
        gnn_hypers['tolerance'], gnn_hypers['patience'], loss_terminal,gnn_hypers['dim_embedding'], gnn_hypers['number_classes'], modelName,  TORCH_DTYPE,  TORCH_DEVICE)

    return trained_net, bestLost, epoch, inp, lossList

def train_2wayNeural(modelName, filename='./testData/prepareDS.pkl'):
    n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim = hyperParameters(learning_rate=0.001, n=4096,patience=20)

    # Establish pytorch GNN + optimizer
    opt_params = {'lr': learning_rate}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.0,
        'number_classes': 2,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience,
        'nodes':n
    }
    datasetItem = open_file(filename)

    net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)


    trained_net, bestLost, epoch, inp, lossList= run_gnn_training2(
        datasetItem, net, optimizer, int(500),
        gnn_hypers['tolerance'], gnn_hypers['patience'], loss_terminal,gnn_hypers['dim_embedding'], gnn_hypers['number_classes'], modelName,  TORCH_DTYPE,  TORCH_DEVICE)

    return trained_net, bestLost, epoch, inp, lossList

import torch
import torch.nn.functional as F

def max_to_one_hot(tensor):
    # Find the index of the maximum value
    max_index = torch.argmax(tensor)

    # Create a one-hot encoded tensor
    one_hot_tensor = torch.zeros_like(tensor)
    one_hot_tensor[max_index] = 1.0

    one_hot_tensor = one_hot_tensor + tensor - tensor.detach()

    return one_hot_tensor

def apply_max_to_one_hot(output):
    return torch.stack([max_to_one_hot(output[i]) for i in range(output.size(0))])


def run_gnn_training2(dataset, net, optimizer, number_epochs, tol, patience, loss_func, dim_embedding, total_classes=3, save_directory=None, torch_dtype = TORCH_DTYPE, torch_device = TORCH_DEVICE, labels=None):
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
    cumulative_loss = 0

    t_gnn_start = time()

    # contains information regarding all terminal nodes for the dataset
    terminal_configs = {}
    epochCount = 0
    criterion = nn.BCELoss()
    A = nn.Parameter(torch.tensor([65.0]))
    C = nn.Parameter(torch.tensor([32.5]))

    embed = nn.Embedding(1000, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)
    inputs = embed.weight

    ds_list = ['../Jobs/nx_test_generated_graph_n800_4000_d8_12_t500_200.pkl',
               '../Jobs/nx_test_generated_graph_n800_4000_d8_12_t500_400.pkl',
               '../Jobs/nx_test_generated_graph_n800_4000_d8_12_t500.pkl']

    for epoch in range(number_epochs):

        cumulative_loss = 0.0  # Reset cumulative loss for each epoch

        for i in ds_list:
            dataset = open_file(i)
            for key, (dgl_graph, adjacency_matrix,graph, terminals) in dataset.items():
                epochCount +=1


                # Ensure model is in training mode
                net.train()

                # Pass the graph and the input features to the model
                logits = net(dgl_graph, adjacency_matrix)
                logits = override_fixed_nodes(logits)
                # Apply max to one-hot encoding
                one_hot_output = apply_max_to_one_hot(logits)
                # Compute the loss
                # loss = loss_func(criterion, logits, labels, terminals[0], terminals[1])

                loss = loss_func( one_hot_output, adjacency_matrix)


                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update cumulative loss
                cumulative_loss += loss.item()



                # # Check for early stopping
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

        # # Early stopping break from the outer loop
        # if count >= patience:
        #     count=0

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

            if (prev_cummulative_loss == cummulativeCount):
                cummulativeCount+=1

                if cummulativeCount > 4:
                    break
            else:
                prev_cummulative_loss = cumulative_loss


    t_gnn = time() - t_gnn_start

    # Load the best model state
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    print(f'GNN training took {round(t_gnn, 3)} seconds.')
    print(f'Best cumulative loss: {best_loss}')
    loss = loss_func(logits, adjacency_matrix)
    if save_directory != None:
        checkpoint = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossList':loss_list,
            'inputs':inputs}
        torch.save(checkpoint, './final_'+save_directory)

    return net, best_loss, epoch, inputs, loss_list

class GCNSoftmax(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout, device):
        super(GCNSoftmax, self).__init__()
        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, num_classes).to(device)

    def forward(self, g, inputs):
        # Basic forward pass
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_frac, training=self.training)
        h = self.conv2(g, h)
        h = F.softmax(h, dim=1)  # Apply softmax over the classes dimension
        # h = F.sigmoid(h)
        # h = override_fixed_nodes(h)

        return h

def override_fixed_nodes(h):
    output = h.clone()
    # Set the output for node 0 to [1, 0, 0]
    output[0] = torch.tensor([1.0, 0.0, 0.0],requires_grad=True) + h[0] - h[0].detach()
    # Set the output for node 1 to [0, 1, 0]
    output[1] = torch.tensor([0.0, 1.0, 0.0],requires_grad=True)+ h[1] - h[1].detach()
    # Set the output for node 2 to [0, 0, 1]
    output[2] = torch.tensor([0.0, 0.0, 1.0],requires_grad=True)+ h[2] - h[2].detach()
    return output

def calculate_HC_vectorized(s, adjacency_matrix):
    """
    Compute the minimum cut loss, which is the total weight of edges cut between partitions using vectorized operations.

    Parameters:
    s (torch.Tensor): Binary partition matrix of shape (num_nodes, num_partitions)
    adjacency_matrix (torch.Tensor): Adjacency matrix of the graph of shape (num_nodes, num_nodes)

    Returns:
    torch.Tensor: Scalar loss value representing the total weight of edges cut
    """
    num_nodes, num_partitions = s.shape

    # Compute the partition probability matrix for all partitions
    partition_prob_matrix = s @ s.T

    # Compute the cut value by summing weights of edges that connect nodes in different partitions
    cut_value = adjacency_matrix * (1 - extend_matrix_torch(partition_prob_matrix, 10000))

    # Sum up the contributions for all edges
    loss = torch.sum(cut_value) / 2  # Divide by 2 to correct for double-counting

    return loss

def Loss(s, adjacency_matrix,  A=1, C=1):
    HC = -1*calculate_HC_vectorized(s, adjacency_matrix)
    return C * HC


def loss_terminal(s, adjacency_matrix,  A=0, C=1, penalty=1000):
    loss = Loss(s, adjacency_matrix, A, C)
    # loss += penalty* terminal_independence_penalty(s, [0,1,2])
    return loss

trained_net, bestLost, epoch, inp, lossList = train1('_10000MaxwayCut_LossExp8_loss.pth', '../Jobs/nx_test_generated_graph_n800_4000_d8_12_t500_200.pkl', 10000)

save_object([trained_net, bestLost, epoch, inp, lossList], 'trained_data.pkl')