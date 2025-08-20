
from python.commons import *

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

def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'reg_random':
        print(f'Generating d-regular random graph with n={n}, d={d}')
        nx_temp = nx.random_regular_graph(d=d, n=n)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    nx_graph.order()
    return nx_graph
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
            terminals = all_terminals[i]
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

# def graph_with_max_edges(graph_list):
#     # Initialize variables to keep track of the graph with the max number of edges
#     max_nodes = -1
#     max_graph = None
#
#     # Iterate over all graphs in the list
#     for key, graph in graph_list.items():
#         num_nodes = graph.number_of_nodes()
#         if num_nodes > max_nodes:
#             max_nodes = num_nodes
#             max_graph = graph
#
#     return max_graph, max_nodes

# N = 250  # Desired size of the extended matrix
# original_matrix = np.random.randint(1, 10, size=(200, 200))  # Creating a random 200 x 200 matrix
#
# extended_matrix = extend_matrix(original_matrix, N)
# print("Original Matrix:")
# print(original_matrix)
# print("\nExtended Matrix:")
# print(extended_matrix)

nx_generated_graph = {}
terminals = {}

for i in range (0, 500, 1):
    nodes = random.randint(200,500)
    degree = random.randint(8,12)
    if (nodes * degree) % 2 != 0:
        i-=1
        continue
    nx_graph = generate_graph(n=nodes, d=degree, p=None, graph_type='reg', random_seed=i)

    for u, v, d in nx_graph.edges(data=True):
        d['weight'] = 1
        d['capacity'] = 1

    unique_random_numbers = generate_unique_random_numbers(nodes)

    nx_generated_graph[i] = nx_graph
    terminals[i] = unique_random_numbers
# #
# nx_generated_graph = {}
# terminals = {}

# for i in range (501, 700, 1):
#     nodes = random.randint(1000,1002)
#     degree = random.randint(8,12)
#     if (nodes * degree) % 2 != 0:
#         i-=1
#         continue
#     nx_graph = generate_graph(n=nodes, d=degree, p=None, graph_type='reg', random_seed=i)
#
#     for u, v, d in nx_graph.edges(data=True):
#         d['weight'] = 1
#         d['capacity'] = 1
#
#     unique_random_numbers = generate_unique_random_numbers(nodes)
#
#     nx_generated_graph[i] = nx_graph
#     terminals[i] = unique_random_numbers
# # #

save_object(nx_generated_graph, 'nx_generated_graph_nfull_20.pkl')
save_object(terminals, 'terminals_nx_generated_graph_nfull_20.pkl')

# ds = createGraphFromFolder_full(nx_generated_graph, terminals, 10000)
# save_object(ds, 'nx_test_generated_graph_n800_4000_d8_12_t500.pkl')