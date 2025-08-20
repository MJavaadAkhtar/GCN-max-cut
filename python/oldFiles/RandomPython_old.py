from python.commons import *
import time


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


def random_partition(graph, X=100):
    nodes = list(graph.nodes)
    best_cut_value = float('-inf')
    best_partition = None

    # Ensure nodes 0, 1, and 2 are in partitions 0, 1, and 2 respectively.
    fixed_nodes = {0: 0, 1: 1, 2: 2, 3:3, 4:4}
    remaining_nodes = [n for n in nodes if n not in fixed_nodes]

    for _ in range(X):
        # Start with the fixed nodes
        partition = fixed_nodes.copy()

        # Randomly assign the remaining nodes to partitions 0, 1, or 2
        for node in remaining_nodes:
            partition[node] = random.choice([0, 1, 2, 3, 4])

        # Calculate the cut value for this partition
        cut_value = obj_maxcut_3way(partition, graph)

        # Check if this is the best cut value so far
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_partition = partition.copy()

    return best_partition, best_cut_value

def findAvg():
    ds = openFiles([[2000,3000], [3001, 4000], [4001, 5000], [5001, 6000], [6001, 7000]])
    best_partition_item = {}
    for key, (graphItems) in ds.items():
        temp_partition_cut_val = []
        start_time = time.time()

        for key2, (dgl_graph, adjacency_matrix,graph, terminals) in graphItems.items():
            best_partition, best_cut_value = random_partition(graph, 30)
            temp_partition_cut_val.append(best_cut_value)

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the total time taken
        temp_partition_cut_val = np.array(temp_partition_cut_val)
        best_partition_item[key] = [np.mean(temp_partition_cut_val), elapsed_time]

        print(f'Graph Node: {key}, Graph Average :{best_partition_item[key][0]}, time: {best_partition_item[key][1]}')
        save_object(best_partition_item, 'nx_random_avg.pkl')

# findAvg()
G = generate_graph(500, 12, p=None, graph_type='reg', random_seed=0)
for u, v, d in G.edges(data=True):
    d['weight'] = 1
    d['capacity'] = 1
start_time = time.time()
best_partition, best_cut_value = random_partition(G, 100)
end_time = time.time()  # End the timer
elapsed_time = end_time - start_time  # Calculate the total time taken

print("Best Partition:", best_partition)
print("Best Cut Value:", best_cut_value)
print("Best time:", elapsed_time)

#
# with open(f'./test_graph_6284.csv',  'w') as f:
#     f.write('i,j,weight\n')
#     for u, v, data in G.edges(data=True):
#         f.write(f'{u},{v},{data["weight"]}\n')