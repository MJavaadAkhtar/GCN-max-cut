from python.commons import *
import time

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32

# Define a scoring function (example: sum of assigned partition probabilities for simplicity)
# def score_partition_assignment(partition_assignment, node_probs):
#     return sum(node_probs[i][partition.index(1)] for i, partition in enumerate(partition_assignment))
#
# def assign_partitions(node_probs):
#     # print(len(node_probs))
#     partition_assignment = []
#     # partition_assignment.append([1,0,0])
#     # partition_assignment.append([0,1,0])
#     # partition_assignment.append([0,0,1])
#     node_probs[0] =  torch.Tensor([1,0,0])
#     node_probs[1] =  torch.Tensor([0,1,0])
#     node_probs[2] =  torch.Tensor([0,0,1])
#     for probs in node_probs[3:]:
#         rand_val = np.random.rand()  # Generate a random number between 0 and 1
#         cumulative_prob = 0
#         partition_vector = [0, 0, 0]  # Initialize a vector with zeros for each partition
#
#         for i, prob in enumerate(probs):
#             cumulative_prob += prob
#             probs[i] = 0
#             if rand_val < cumulative_prob:
#                 partition_vector[i] = 1  # Set 1 for the selected partition
#                 probs[i] = 1
#                 for k in probs:
#                     if probs[k] != 1:
#                         probs[k] = 0
#                 break
#         partition_assignment.append(partition_vector)
#     # print(len(partition_assignment))
#     return probs

def postProcessing(node_probabilities, graph, range_item=200):
    best_partition_assignment = None
    best_score = -float('inf')
    for _ in range(range_item):
        current_partition_assignment = assign_partitions(node_probabilities)
        # print(node_probabilities)
        # print(current_partition_assignment)
        current_score =obj_maxcut_3way(current_partition_assignment, graph) #score_partition_assignment(current_partition_assignment, node_probabilities)

        # Keep track of the best assignment
        if current_score > best_score:
            best_score = current_score
            best_partition_assignment = current_partition_assignment
    return  best_partition_assignment, best_score
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

def calculate_HC_vectorized(s, adjacency_matrix, n):
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
    cut_value = adjacency_matrix * (1 - extend_matrix_torch(partition_prob_matrix, n))

    # Sum up the contributions for all edges
    loss = torch.sum(cut_value) / 2  # Divide by 2 to correct for double-counting

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
    weight = (adj * extend_matrix_torch(partition_matrix, 1000)).sum() / 2
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
        #print(q_torch.size())
        # print(s[:,i].size())
        for i in range(totalCuts):
            CutValue += partition_weight(q_torch, s[:,i])
        return int(CutValue/2)
    return 0
def extend_matrix_torch(matrix, N, torch_dtype=None, torch_device=None):
    original_size = matrix.shape[0]

    if N < original_size:
        raise ValueError("N should be greater than or equal to the original matrix size.")

    extended_matrix = torch.zeros(original_size, N)
    extended_matrix[:original_size, :original_size] = torch.from_numpy(matrix)

    if torch_dtype is not None:
        extended_matrix = extended_matrix.type(torch_dtype)

    if torch_device is not None:
        extended_matrix = extended_matrix.to(torch_device)

    return extended_matrix.numpy()
def hyperParameters(n = 100, d = 3, p = None, graph_type = 'reg', number_epochs = int(1e5),
                    learning_rate = 1e-4, PROB_THRESHOLD = 0.5, tol = 1e-4, patience = 100):
    dim_embedding = 80    # e.g. 10
    hidden_dim = int(dim_embedding/2)
    return n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim

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


def printCombo(orig):
    # Original dictionary
    input_dict = orig

    # Generate all permutations of the dictionary values
    value_permutations = list(permutations(input_dict.values()))

    # Create a list of dictionaries from the permutations
    permuted_dicts = [{key: value for key, value in zip(input_dict.keys(), perm)} for perm in value_permutations]

    return permuted_dicts

def GetOptimal(net, dgl_graph, inp, q_torch, terminal = None):

    probs = net(dgl_graph, inp, terminal)
    binary_partitions = (probs >= 0.5).float()

    for i in range(len(binary_partitions)-1):
        if torch.sum(binary_partitions[i]) != 1:
            binary_partitions[i] = torch.tensor([0,1,0])

    cut_value_item = calculateAllCut(q_torch, binary_partitions)

    return cut_value_item, binary_partitions

def GetOptimalNetValue(net, dgl_graph, inp, q_torch, terminal_dict):
    net.eval()
    best_loss = float('inf')
    best_binary = []
    # if (dgl_graph.number_of_nodes() < 30):
    #     inp = torch.ones((dgl_graph.number_of_nodes(), 30))

    # find all potential combination of terminal nodes with respective indices

    perm_items = printCombo(terminal_dict)
    for i in perm_items:
        probs = net(dgl_graph, inp, i)
        binary_partitions = (probs >= 0.5).float()
        # print([m for m in binary_partitions if sum(m)>1 or sum(m)==0])
        # print(binary_partitions, q_torch)
        cut_value_item = calculateAllCut(q_torch, binary_partitions)
        if cut_value_item < best_loss:
            best_loss = cut_value_item
            best_binary = binary_partitions
    return best_loss, best_binary

def obj_maxcut_3way(solution, graph):
    cut_value = 0
    for u, v, data in graph.edges(data=True):
        if solution[u] != solution[v]:
            cut_value += data.get('weight', 1)  # Assuming weight attribute is correct
    return cut_value


def openFiles( lists):
    all_ds = {}
    for (start, end)  in lists:
        filename = f'../nx_test_generated_graph_n{start}_{end}_d8_12_t500.pkl'
        ds = open_file(filename)
        all_ds[start] = ds
    return all_ds
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

def override_fixed_nodes(h):
    output = h.clone()
    # Set the output for node 0 to [1, 0, 0]
    output[0] = torch.tensor([1.0, 0.0, 0.0],requires_grad=True) + h[0] - h[0].detach()
    # Set the output for node 1 to [0, 1, 0]
    output[1] = torch.tensor([0.0, 1.0, 0.0],requires_grad=True)+ h[1] - h[1].detach()
    # Set the output for node 2 to [0, 0, 1]
    output[2] = torch.tensor([0.0, 0.0, 1.0],requires_grad=True)+ h[2] - h[2].detach()
    return output
def hyperParameters(n = 80, d = 3, p = None, graph_type = 'reg', number_epochs = int(1e5),
                    learning_rate = 1e-4, PROB_THRESHOLD = 0.5, tol = 1e-4, patience = 100):
    dim_embedding = 1000 #int(np.sqrt(4096))    # e.g. 10, used to be the one before
    hidden_dim = int(dim_embedding/2)

    return n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim

def score_partition_assignment(partition_assignment, node_probs):
    return sum(node_probs[i][partition] for i, partition in enumerate(partition_assignment))

def assign_partitions(node_probs):
    partition_assignment = [0,1,2]
    for probs in node_probs[3:]:
        rand_val = np.random.rand()  # Generate a random number between 0 and 1
        cumulative_prob = 0
        for i, prob in enumerate(probs):
            cumulative_prob += prob
            if rand_val < cumulative_prob:
                partition_assignment.append(i)
                break
    return partition_assignment
def neuralNet(dgl_graph, adjacency_matrix, nn_location = './testing_new/final__1000MaxwayCut_loss_100.pth'):

    n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim = hyperParameters(n=1000,patience=40)
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

    net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)
    model, inputs =LoadNeuralModel(net, gnn_hypers, TORCH_DEVICE, nn_location)
    model.eval()

    logits = net(dgl_graph, adjacency_matrix)
    # logits = override_fixed_nodes(logits)
    binary_partitions = (logits >= 0.5).float()

    for i in range(len(binary_partitions)):
        if sum(binary_partitions[i]) != 1:
            binary_partitions[i] = torch.Tensor([1,0,0])
    binary_partitions[0] =  torch.Tensor([1,0,0])
    binary_partitions[1] =  torch.Tensor([0,1,0])
    binary_partitions[2] =  torch.Tensor([0,0,1])

    # print( np.sum(binary_partitions[:, 1] == 1))
    # print( np.sum(binary_partitions[:, 2] == 1))

    cut = calculateAllCut(adjacency_matrix, binary_partitions)
    totSum = np.sum(binary_partitions.numpy(), axis=0)

    return binary_partitions, cut

def neuralNet_post(dgl_graph, adjacency_matrix, graph, nn_location = './testing_new/final__1000MaxwayCut_loss_100.pth'):

    n, d, p, graph_type, number_epochs, learning_rate, PROB_THRESHOLD, tol, patience, dim_embedding, hidden_dim = hyperParameters(n=1000,patience=40)
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

    net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)
    model, inputs =LoadNeuralModel(net, gnn_hypers, TORCH_DEVICE, nn_location)
    model.eval()

    logits = net(dgl_graph, adjacency_matrix)
    # logits = override_fixed_nodes(logits)
    # binary_partitions = (logits >= 0.5).float()
    #
    # for i in range(len(binary_partitions)):
    #     if sum(binary_partitions[i]) != 1:
    #         binary_partitions[i] = torch.Tensor([1,0,0])

    partitionItem, score = postProcessing(logits, graph)
    # binary_partitions[0] =  torch.Tensor([1,0,0])
    # binary_partitions[1] =  torch.Tensor([0,1,0])
    # binary_partitions[2] =  torch.Tensor([0,0,1])
    # cut = calculateAllCut(adjacency_matrix, binary_partitions)
    # totSum = np.sum(binary_partitions.numpy(), axis=0)
    return partitionItem, score

# def findAvg():
#     ds = openFiles([[2000,3000], [3001, 4000], [4001, 5000], [5001, 6000], [6001, 7000]])
#     best_partition_item = {}
#     for key, (graphItems) in ds.items():
#         temp_partition_cut_val = []
#         start_time = time.time()
#
#         for key2, (dgl_graph, adjacency_matrix,graph, terminals) in graphItems.items():
#             best_partition, best_cut_value = neuralNet(dgl_graph, adjacency_matrix)
#             temp_partition_cut_val.append(best_cut_value)
#
#         end_time = time.time()  # End the timer
#         elapsed_time = end_time - start_time  # Calculate the total time taken
#         temp_partition_cut_val = np.array(temp_partition_cut_val)
#         best_partition_item[key] = [np.mean(temp_partition_cut_val), elapsed_time]
#
#         print(f'Graph Node: {key}, Graph Average :{best_partition_item[key][0]}, time: {best_partition_item[key][1]}')
#     save_object(best_partition_item, 'nx_neural_avg.pkl')
#
# findAvg()
# G = generate_graph(2734, 12, p=None, graph_type='reg', random_seed=0)
# best_partition, best_cut_value = random_partition(G, 100)
# print("Best Partition:", best_partition)
# print("Best Cut Value:", best_cut_value)

# ds = open_file('nx_test_generated_graph_n40_100_d8_12_t500.pkl')
# lst = []
# totalTime = []
# temp_partition_cut_val = []
# for key, (dgl_graph, adjacency_matrix,graph, terminals) in ds.items():
#     start_time = time.time()
#     best_partition, best_cut_value = neuralNet(dgl_graph, adjacency_matrix)
#     temp_partition_cut_val.append(best_cut_value)
#     end_time = time.time()  # End the timer
#     elapsed_time = end_time - start_time  # Calculate the total time taken
#     totalTime.append(elapsed_time)
#     # lst.append(cplex_solver(graph))
#
# print(temp_partition_cut_val)
# print(totalTime)
# print(np.average(totalTime))
# print(np.average(temp_partition_cut_val))

''''
[269, 243, 179, 200, 307, 192, 191, 241, 241]
[0.22798609733581543, 0.18748021125793457, 0.12467002868652344, 0.17549705505371094, 0.1165618896484375, 0.21324801445007324, 0.08788394927978516, 0.054058074951171875, 0.06474781036376953]
0.13912590344746908
229.22222222222223
'''

def test_n100(filename = 'nx_test_generated_graph_n40_100_d8_12_t500.pkl'):
    ds = open_file(filename)
    lst = []
    totalTime = []
    temp_partition_cut_val = []
    for key, (dgl_graph, adjacency_matrix,graph, terminals) in ds.items():
        start_time = time.time()
        best_partition, best_cut_value = neuralNet(dgl_graph, adjacency_matrix)
        temp_partition_cut_val.append(best_cut_value)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the total time taken
        totalTime.append(elapsed_time)
        # lst.append(cplex_solver(graph))

    print(temp_partition_cut_val)
    print(totalTime)
    print(np.average(totalTime))
    print(np.average(temp_partition_cut_val))

''''
[278, 255, 183, 221, 320, 205, 204, 257, 265]
[0.485184907913208, 0.504971981048584, 0.26601171493530273, 0.4156370162963867, 0.4751899242401123, 0.3337130546569824, 0.32118988037109375, 0.27427005767822266, 0.33888792991638184]
0.3794507185618083
243.11111111111111
'''
def test_n200(filename = 'nx_test_generated_graph_n40_100_d8_12_t500.pkl'):
    ds = open_file(filename)
    lst = []
    totalTime = []
    temp_partition_cut_val = []
    for key, (dgl_graph, adjacency_matrix,graph, terminals) in ds.items():
        start_time = time.time()
        best_partition, best_cut_value = neuralNet(dgl_graph, adjacency_matrix, './testing_new/balancedGraph/final__1000MaxwayCut_loss_200.pth')
        temp_partition_cut_val.append(best_cut_value)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the total time taken
        totalTime.append(elapsed_time)
        # lst.append(cplex_solver(graph))

    print(temp_partition_cut_val)
    print(totalTime)
    print(np.average(totalTime))
    print(np.average(temp_partition_cut_val))

'''
[310, 372, 184, 236, 371, 201, 276, 265, 319]
[0.4084160327911377, 0.4309549331665039, 0.22517991065979004, 0.2300260066986084, 0.4305238723754883, 0.3344559669494629, 0.3296236991882324, 0.29529690742492676, 0.33498191833496094]
0.3354954719543457
281.55555555555554
'''


def test_n300(filename = 'nx_test_generated_graph_n40_100_d8_12_t500.pkl'):
    ds = open_file(filename)
    lst = []
    totalTime = []
    temp_partition_cut_val = []
    for key, (dgl_graph, adjacency_matrix,graph, terminals) in ds.items():
        start_time = time.time()
        best_partition, best_cut_value = neuralNet(dgl_graph, adjacency_matrix,  './testing_new/balancedGraph/final__1000MaxwayCut_loss_300.pth')
        temp_partition_cut_val.append(best_cut_value)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the total time taken
        totalTime.append(elapsed_time)
        # lst.append(cplex_solver(graph))

    print(temp_partition_cut_val)
    print(totalTime)
    print(np.average(totalTime))
    print(np.average(temp_partition_cut_val))


# node_probabilities = [
#     [0.2, 0.7, 0.1],
#     [0.5, 0.3, 0.2],
#     [0.1, 0.6, 0.3],
#     # Add more nodes as needed
# ]
# print(postProcessing(node_probabilities))


# test_n100("nx_test_generated_graph_n40_100_d8_12_t500.pkl")
'''
[278, 264, 182, 225, 321, 204, 206, 257, 263]
[0.4964737892150879, 0.4908430576324463, 0.3741466999053955, 0.34170007705688477, 0.4687690734863281, 0.32533907890319824, 0.39397382736206055, 0.28897595405578613, 0.39260005950927734]
0.3969801796807183
244.44444444444446
'''
# test_n200("nx_test_generated_graph_n40_100_d8_12_t500.pkl")
'''
[307, 373, 184, 236, 373, 201, 276, 265, 322]
[0.47101521492004395, 0.4803040027618408, 0.3161017894744873, 0.24278903007507324, 0.4998002052307129, 0.26612210273742676, 0.3087890148162842, 0.3157970905303955, 0.30446410179138184]
0.3561313947041829
281.8888888888889
'''
# test_n300("nx_test_generated_graph_n40_100_d8_12_t500.pkl")
'''
[279, 314, 137, 188, 333, 198, 247, 219, 282]
[0.5304839611053467, 0.4963200092315674, 0.3741457462310791, 0.3768010139465332, 0.534991979598999, 0.35170507431030273, 0.39162492752075195, 0.32675790786743164, 0.35293126106262207]
0.41508465343051487
294.11111111111111
'''


# test_n100("nx_test_generated_graph_n101_200_d8_12_t500.pkl")
'''
[772, 481, 421, 580, 573, 813, 570, 534, 497, 535, 293]
[0.8980097770690918, 0.6594500541687012, 0.477916955947876, 0.689857006072998, 0.6040968894958496, 0.7956221103668213, 0.5321118831634521, 0.5196669101715088, 0.5710220336914062, 0.5589308738708496, 0.4578828811645508]
0.6149606704711914
551.7272727272727
'''
# test_n200("nx_test_generated_graph_n101_200_d8_12_t500.pkl")
'''
[808, 482, 499, 586, 661, 866, 634, 529, 563, 534, 315]
[0.6782310009002686, 0.5218088626861572, 0.44451403617858887, 0.6408908367156982, 0.5430152416229248, 0.7872650623321533, 0.523453950881958, 0.5174250602722168, 0.5240168571472168, 0.5371291637420654, 0.44750475883483887]
0.5604777119376443
588.8181818181819
'''
# test_n300("nx_test_generated_graph_n101_200_d8_12_t500.pkl")
'''
[833, 469, 416, 637, 672, 840, 584, 571, 478, 510, 303]
[0.659538984298706, 0.5740959644317627, 0.4753689765930176, 0.6715519428253174, 0.5603311061859131, 0.7970349788665771, 0.581671953201294, 0.5212101936340332, 0.5652060508728027, 0.5614829063415527, 0.4585130214691162]
0.5841823707927357
573.9090909090909
'''


# test_n100("nx_test_generated_graph_n200_300_d8_12_t500.pkl")
'''
[703, 946, 944, 1175, 694, 1014, 885, 944, 1149, 728, 907, 765]
[0.9690828323364258, 1.0226781368255615, 0.9714090824127197, 1.0768482685089111, 0.8207738399505615, 0.9495668411254883, 0.9138219356536865, 0.9384989738464355, 0.9822449684143066, 0.8094699382781982, 1.1109800338745117, 1.0069999694824219]
0.9643645683924357
904.5
'''
# test_n200("nx_test_generated_graph_n200_300_d8_12_t500.pkl")
'''
[726, 878, 980, 1229, 675, 990, 915, 986, 1118, 814, 950, 833]
[1.024824857711792, 1.1634559631347656, 1.0989930629730225, 1.1095571517944336, 0.8819150924682617, 0.9993908405303955, 0.9387438297271729, 0.9444360733032227, 1.0122270584106445, 0.7896029949188232, 1.109281063079834, 1.075674295425415]
1.012341856956482
924.5
'''
test_n300("nx_test_generated_graph_n200_300_d8_12_t500.pkl")
'''
[715, 966, 971, 1243, 733, 977, 888, 1017, 1166, 781, 881, 847]
[0.9130082130432129, 0.9467549324035645, 0.9251630306243896, 1.076101303100586, 0.8507471084594727, 0.9597220420837402, 0.8473680019378662, 0.9116058349609375, 0.9468920230865479, 0.7943549156188965, 1.1034431457519531, 0.9904749393463135]
0.93880295753479
932.0833333333334
'''


# test_n100("nx_test_generated_graph_n700_800_d8_12_t500.pkl")
'''
[2141, 2991, 2337, 2740, 2705, 3088, 2236]
[2.544975757598877, 2.592775821685791, 2.531911849975586, 2.4958841800689697, 2.727475881576538, 2.6058270931243896, 2.514227867126465]
2.573296921593802
2605.4285714285716
'''
# test_n200("nx_test_generated_graph_n700_800_d8_12_t500.pkl")
'''
[2261, 3114, 2528, 2803, 2852, 3089, 2319]
[2.5159108638763428, 2.7034900188446045, 2.5916547775268555, 2.5424838066101074, 3.0694546699523926, 2.6679389476776123, 2.6380879878997803]
2.6755744389125278
2709.4285714285716
'''
# test_n300("nx_test_generated_graph_n700_800_d8_12_t500.pkl")
'''
[2311, 3275, 2459, 2890, 2845, 3187, 2458]
[2.3721258640289307, 2.677150011062622, 2.5534660816192627, 2.569889783859253, 2.791317939758301, 2.5484650135040283, 2.48685884475708]
2.571324791227068
2775.0
'''


# test_n100("nx_test_generated_graph_n800_900_d8_12_t500.pkl")
'''
[3191, 2614, 3048, 3476, 3130, 2561, 3412, 3068, 3171]
[3.063774824142456, 2.973912000656128, 3.3335659503936768, 3.4179298877716064, 3.297959089279175, 3.0611190795898438, 3.122404098510742, 2.9045238494873047, 2.8308002948760986]
3.1117765638563366
3074.5555555555557
'''
# test_n200("nx_test_generated_graph_n800_900_d8_12_t500.pkl")
'''
[3283, 2696, 3003, 3595, 3234, 2702, 3410, 2988, 3118]
[3.2811970710754395, 2.974708080291748, 2.9476420879364014, 3.3803060054779053, 3.2143030166625977, 3.0256950855255127, 3.0734329223632812, 2.979706048965454, 2.832458019256592]
3.078827593061659
3114.3333333333335
'''
# print('---')
# test_n300("nx_test_generated_graph_n800_900_d8_12_t500.pkl")
'''[3205, 2708, 2965, 3584, 3271, 2657, 3510, 3145, 3274]
[3.1211178302764893, 2.929764986038208, 3.019868850708008, 3.1786279678344727, 3.1859052181243896, 3.057681083679199, 3.0771639347076416, 2.908369779586792, 2.8593637943267822]
3.037540382809109
3146.5555555555557
'''


# test_n100("nx_test_generated_graph_n900_999_d8_12_t500.pkl")
'''
[2884, 3162, 3250, 2627, 2759, 3939, 4075, 2739, 3427, 2670]
[3.3770830631256104, 3.3182179927825928, 3.368360996246338, 3.1852619647979736, 3.4445881843566895, 3.4244558811187744, 3.409281015396118, 3.0384671688079834, 3.221513271331787, 3.296776294708252]
3.308400583267212
3153.2
'''
# print('---')
# test_n200("nx_test_generated_graph_n900_999_d8_12_t500.pkl")
'''
[2901, 3197, 3362, 2635, 2795, 3853, 3970, 2894, 3368, 2906]
[3.47615385055542, 3.271285057067871, 3.501173973083496, 3.2646307945251465, 3.2010672092437744, 3.303877830505371, 3.3621129989624023, 3.098536968231201, 3.2488200664520264, 3.353416919708252]
3.3081075668334963
3188.1
'''
# print('---')
# test_n300("nx_test_generated_graph_n900_999_d8_12_t500.pkl")
'''
[3012, 3309, 3373, 2786, 2850, 3947, 4166, 2923, 3409, 2985]
[3.4391369819641113, 3.1702189445495605, 3.4660208225250244, 3.166607141494751, 3.4106719493865967, 3.3623878955841064, 3.4655261039733887, 3.081348180770874, 3.3526771068573, 3.3668789863586426]
3.3281474113464355
3276.0
'''

# test_n100("nx_test_generated_graph_n900_999_d8_12_t500.pkl")
# print('---')
#
# filenames = []
# for i in filenames:
#     test_n300(i)

# ---------- Balanced cut algo
# test_n100("nx_test_generated_graph_n40_100_d8_12_t500.pkl")
# filenames = ['nx_test_generated_graph_n101_200_d8_12_t500.pkl', 'nx_test_generated_graph_n200_300_d8_12_t500.pkl',
#              'nx_test_generated_graph_n700_800_d8_12_t500.pkl', 'nx_test_generated_graph_n800_900_d8_12_t500.pkl', 'nx_test_generated_graph_n900_999_d8_12_t500.pkl']
# for i in filenames:
#     test_n300(i)
#     print('-------')
# ---------

#---- Balanced Cut Algo 100
'''
[667, 388, 389, 597, 541, 844, 550, 505, 459, 469, 304]
[0.5284018516540527, 0.06730985641479492, 0.0766611099243164, 0.06521391868591309, 0.11421680450439453, 0.05752992630004883, 0.14762425422668457, 0.18261122703552246, 0.05159711837768555, 0.22075915336608887, 0.06016993522644043]
0.14291774142872204
519.3636363636364
-------
[725, 879, 983, 1141, 699, 936, 867, 1006, 1037, 750, 920, 779]
[0.05666303634643555, 0.16030097007751465, 0.2450709342956543, 0.175079345703125, 0.06143832206726074, 0.23361515998840332, 0.17019200325012207, 0.30704474449157715, 0.1928110122680664, 0.10090374946594238, 0.18172001838684082, 0.10517191886901855]
0.16583426793416342
893.5
-------
[2130, 3025, 2401, 2738, 2800, 3042, 2339]
[0.14819574356079102, 0.09202790260314941, 0.34189701080322266, 0.2014772891998291, 0.3266112804412842, 0.23296904563903809, 0.2590620517730713]
0.2288914748600551
2639.285714285714
-------
[3196, 2694, 3047, 3549, 3039, 2538, 3470, 2889, 3091]
[0.20038104057312012, 0.29057979583740234, 0.28357505798339844, 0.3881227970123291, 0.3276989459991455, 0.3137192726135254, 0.258267879486084, 0.25260019302368164, 0.3726170063018799]
0.29861799875895184
3057.0
-------
[2786, 3166, 3315, 2553, 2778, 3888, 3973, 2813, 3357, 2909]
[0.16372895240783691, 0.31044483184814453, 0.36020898818969727, 0.11845517158508301, 0.33171892166137695, 0.11697888374328613, 0.36632299423217773, 0.25942492485046387, 0.43593907356262207, 0.2827799320220947]
0.2746002674102783
3153.8
-------
'''
#----

#---- Balanced Cut Algo 200
'''
[783, 436, 416, 602, 669, 809, 589, 593, 471, 500, 347]
[0.1573350429534912, 0.1105959415435791, 0.041892051696777344, 0.08062314987182617, 0.06653475761413574, 0.07746410369873047, 0.14879512786865234, 0.14183902740478516, 0.16373682022094727, 0.10997319221496582, 0.05702090263366699]
0.10507364706559615
565.0
-------
[737, 862, 996, 1172, 643, 987, 939, 1005, 1082, 747, 890, 797]
[0.07956099510192871, 0.051763296127319336, 0.07026004791259766, 0.07294011116027832, 0.10890579223632812, 0.06637692451477051, 0.09633898735046387, 0.07151222229003906, 0.09394097328186035, 0.13411998748779297, 0.09044003486633301, 0.07139992713928223]
0.08396327495574951
904.75
-------
[2328, 3162, 2512, 2809, 2743, 3171, 2418]
[0.1097571849822998, 0.08463788032531738, 0.07711625099182129, 0.07105588912963867, 0.07178092002868652, 0.08849000930786133, 0.0998530387878418]
0.08609873907906669
2734.714285714286
-------
[3305, 2618, 2964, 3526, 3251, 2547, 3436, 3058, 3169]
[0.10009527206420898, 0.10876727104187012, 0.09367227554321289, 0.09101200103759766, 0.10870504379272461, 0.12549972534179688, 0.10892415046691895, 0.07989907264709473, 0.08018016815185547]
0.09963944223192003
3097.1111111111113
-------
[2896, 3242, 3355, 2587, 2758, 4012, 4005, 2952, 3320, 2929]
[0.09075212478637695, 0.08050704002380371, 0.13160181045532227, 0.09236502647399902, 0.08696174621582031, 0.10555791854858398, 0.09068417549133301, 0.10085296630859375, 0.1581559181213379, 0.09952378273010254]
0.10369625091552734
3205.6

'''
#----

#---- Balanced Cut Algo 300
'''
[771, 463, 452, 619, 639, 839, 550, 517, 470, 541, 293]
[0.15668416023254395, 0.1495990753173828, 0.035034894943237305, 0.04812002182006836, 0.039431095123291016, 0.0532841682434082, 0.05129885673522949, 0.03825783729553223, 0.16341280937194824, 0.14350199699401855, 0.07995963096618652]
0.08714404973116788
559.4545454545455
-------
[719, 919, 996, 1134, 686, 986, 944, 964, 1129, 737, 900, 814]
[0.12160682678222656, 0.1710679531097412, 0.30823826789855957, 0.19286203384399414, 0.20696592330932617, 0.4243929386138916, 0.5241677761077881, 0.3620002269744873, 0.4202003479003906, 0.22053027153015137, 0.3681321144104004, 0.46247196197509766]
0.3152197202046712
910.6666666666666
-------
[2176, 3227, 2501, 2878, 2802, 3243, 2406]
[0.22875404357910156, 0.5079810619354248, 0.3490748405456543, 0.26707005500793457, 0.23696112632751465, 0.40662217140197754, 0.3649883270263672]
0.3373502322605678
2747.5714285714284
-------
[3204, 2748, 2983, 3643, 3241, 2570, 3423, 3016, 3221]
[0.14701199531555176, 0.11285996437072754, 0.0855722427368164, 0.09116506576538086, 0.11861014366149902, 0.09506893157958984, 0.07852292060852051, 0.06972599029541016, 0.16607213020324707]
0.10717882050408258
3116.5555555555557
-------
[2953, 3271, 3508, 2627, 2847, 4013, 4072, 2932, 3497, 2921]
[0.13009285926818848, 0.10985898971557617, 0.10045480728149414, 0.09439992904663086, 0.09348797798156738, 0.09374284744262695, 0.10872197151184082, 0.09068799018859863, 0.07819581031799316, 0.20853495597839355]
0.11081781387329101
3264.1
-------
'''
#----