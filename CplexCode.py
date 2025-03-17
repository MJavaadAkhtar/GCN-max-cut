from fileinput import filename

import numpy as np
from sympy.stats.sampling.sample_numpy import numpy

from commons import *
import time
import pandas as pd
from docplex.cp.model import CpoModel, CpoParameters
from docplex.cp.expression import integer_var_list
from docplex.cp.config import context
from dgl.nn.pytorch import GATConv, EdgeConv


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


def cplex_solver(graph, time = 300):

    print("--------")
    context.solver.agent = 'local'
    context.solver.local.execfile = '/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer'
    # Define model
    model = CpoModel(name="multiMax")
    # model.set_time_limit(20)

    # Parameters
    n = graph.number_of_nodes()
    r = range(n)

    # Read edges data from CSV
    edges = [(u, v, data.get('weight', 1.0)) for u, v, data in graph.edges(data=True)]

    # print("populating edges")
    # edges = [(int(row["i"]), int(row["j"]), float(row["weight"])) for _, row in edges_df[1:].iterrows()]

    # Initialize the weight matrix
    w = [[0.0 for _ in r] for _ in r]

    # print("populating weight data")
    # Populate the weight matrix based on the loaded data
    for edge in edges:
        i, j, weight = edge
        w[i][j] = weight
        w[j][i] = weight  # Assuming the graph is undirected

    print("setting up the model")
    # Define decision variables
    x = integer_var_list(n, 0, 3, name="x")

    print("setting up the OBJ")
    print("model", model)
    # Define the objective function

    # obj = model.sum(w[i][j] * (x[i] != x[j]) for i in r for j in r if i < j)


    print("finished setting up the OBJ")
    # model.set_parameter("TimeLimit", 10)
    # model.parameters.timelimit=10;
    # model.set_time_limit(10) #The same
    # print("time limit = ",model.parameters.timelimit.get())
    # print("time limit = ",model.get_time_limit()) #The same
    param=CpoParameters();
    param.set_TimeLimit(time)
    param.TimeLimit=time;
    param.set_SolutionLimit(100)

    model.set_parameters(param)
    # Set objective to maximize
    # model.add(model.maximize(sum(w[i][j] * (x[i] != x[j]) for i in r for j in r if i < j)))

    model.add(model.maximize(sum(w[i][j] * (x[i] != x[j]) for i in r for j in r )))
    # model.set_log_stream(None)
    # model.set_error_stream(None)
    # model.set_warning_stream(None)
    # model.set_results_stream(None)
    # model.write_information(None)


    # Add constraints to break symmetry
    model.add(x[1] == 1)
    model.add(x[0] == 0)
    model.add(x[2] == 2)

    model.add(x[3] == 3)

    print("Solving")

    # Solve model
    solution = model.solve()  # Time limit of 1800 seconds (30 minutes)

    # Retrieve solution details
    if solution:
        print(f"Objective value: {solution.get_objective_value()/2}")
        # x_values = solution.get_value(x)
        #
        # # Get nodes in each partition
        # x0 = [i for i in r if x_values[i] == 0]
        # x1 = [i for i in r if x_values[i] == 1]
        # x2 = [i for i in r if x_values[i] == 2]

        return solution.get_objective_value()/2

        # print(f"x set to 0: {x0}")
        # print(f"x set to 1: {x1}")
        # print(f"x set to 2: {x2}")
        # print(f"Size of set x0: {len(x0)}")
        # print(f"Size of set x1: {len(x1)}")
        # print(f"Size of set x2: {len(x2)}")
    else:
        print("No solution found")
        return 0

def cplex_solver_balanced(graph, time = 150):

    print("--------")
    context.solver.agent = 'local'
    context.solver.local.execfile = '/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer'
    # Define model
    model = CpoModel(name="multiMax")
    # model.set_time_limit(20)

    # Parameters
    n = graph.number_of_nodes()
    r = range(n)

    # Read edges data from CSV
    edges = [(u, v, data.get('weight', 1.0)) for u, v, data in graph.edges(data=True)]

    # print("populating edges")
    # edges = [(int(row["i"]), int(row["j"]), float(row["weight"])) for _, row in edges_df[1:].iterrows()]

    # Initialize the weight matrix
    w = [[0.0 for _ in r] for _ in r]

    # print("populating weight data")
    # Populate the weight matrix based on the loaded data
    for edge in edges:
        i, j, weight = edge
        w[i][j] = weight
        w[j][i] = weight  # Assuming the graph is undirected

    print("setting up the model")
    # Define decision variables
    x = integer_var_list(n, 0, 2, name="x")

    print("setting up the OBJ")
    print("model", model)
    # Define the objective function

    # obj = model.sum(w[i][j] * (x[i] != x[j]) for i in r for j in r if i < j)


    print("finished setting up the OBJ")
    # model.set_parameter("TimeLimit", 10)
    # model.parameters.timelimit=10;
    # model.set_time_limit(10) #The same
    # print("time limit = ",model.parameters.timelimit.get())
    # print("time limit = ",model.get_time_limit()) #The same
    param=CpoParameters();
    param.set_TimeLimit(time)
    param.TimeLimit=time;
    param.set_SolutionLimit(30)

    model.set_parameters(param)
    # Set objective to maximize
    # model.add(model.maximize(sum(w[i][j] * (x[i] != x[j]) for i in r for j in r if i < j)))

    model.add(model.maximize(sum(w[i][j] * (x[i] != x[j]) for i in r for j in r )))
    # model.set_log_stream(None)
    # model.set_error_stream(None)
    # model.set_warning_stream(None)
    # model.set_results_stream(None)
    # model.write_information(None)


    # Add constraints to break symmetry
    model.add(x[1] == 1)
    model.add(x[0] == 0)
    model.add(x[2] == 2)

    # Enforce balanced partitions
    size0 = n // 3
    size1 = (n + 1) // 3
    size2 = n - size0 - size1  # Ensure that sizes sum up to n

    print(size0, size1, size2)
    # Add constraints to balance partitions using the 'count' function
    model.add(sum((x[i] == 0) for i in r) == size0)
    model.add(sum((x[i] == 1) for i in r) == size1)
    model.add(sum((x[i] == 2) for i in r) == size2)

    print("Solving")

    # Solve model
    solution = model.solve()  # Time limit of 1800 seconds (30 minutes)

    # Retrieve solution details
    if solution:
        print(f"Objective value: {solution.get_objective_value()/2}")
        # x_values = solution.get_value(x)
        #
        # # Get nodes in each partition
        # x0 = [i for i in r if x_values[i] == 0]
        # x1 = [i for i in r if x_values[i] == 1]
        # # x2 = [i for i in r if x_values[i] == 2]
        # x_values = solution.get_value(x)
        # count_partition_0 = sum(1 for value in x_values if value == 0)
        # count_partition_1 = sum(1 for value in x_values if value == 1)
        # count_partition_2 = sum(1 for value in x_values if value == 2)
        #
        #
        # print("Partition counts:")
        # print(f"Partition 0: {count_partition_0} nodes")
        # print(f"Partition 1: {count_partition_1} nodes")
        # print(f"Partition 2: {count_partition_2} nodes")
        # print("-----")
        return solution.get_objective_value()/2

        # print(f"x set to 0: {x0}")
        # print(f"x set to 1: {x1}")
        # print(f"x set to 2: {x2}")
        # print(f"Size of set x0: {len(x0)}")
        # print(f"Size of set x1: {len(x1)}")
        # print(f"Size of set x2: {len(x2)}")
    else:
        print("No solution found")
        return 0


# def findAvg():
#     ds = openFiles([[2000,3000], [3001, 4000], [4001, 5000], [5001, 6000], [6001, 7000]])
#     best_partition_item = {}
#     best_partition_item_perItem = {}
#     for key, (graphItems) in ds.items():
#         temp_partition_cut_val = []
#         start_time = time.time()
#
#         for key2, (dgl_graph, adjacency_matrix,graph, terminals) in graphItems.items():
#             best_partition, best_cut_value = cplex_solver(graph)
#             if (best_cut_value!= 0):
#                 temp_partition_cut_val.append(best_cut_value)
#                 best_partition_item_perItem[key2] = best_partition
#
#         end_time = time.time()  # End the timer
#         elapsed_time = end_time - start_time  # Calculate the total time taken
#         temp_partition_cut_val = np.array(temp_partition_cut_val)
#         best_partition_item[key] = [np.mean(temp_partition_cut_val), elapsed_time]
#
#         print(f'Graph Node: {key}, Graph Average :{best_partition_item[key][0]}, time: {best_partition_item[key][1]}')
#     save_object(best_partition_item, 'nx_cplex_avg.pkl')
#     save_object(best_partition_item_perItem, 'nx_allPartition_avg.pkl')

# findAvg()
# G = generate_graph(6284, 12, p=None, graph_type='reg', random_seed=0)
# best_partition, best_cut_value = cplex_solver(G, 1800)
# print("Best Partition:", best_partition)
# print("Best Cut Value:", best_cut_value)

context.solver.local.execfile = "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"

def test(fileName = 'nx_test_generated_graph_n40_100_d8_12_t500.pkl'):
    ds = open_file(fileName)
    lst = []
    totalTime = []
    for key, (dgl_graph, adjacency_matrix,graph, terminals) in ds.items():
        start_time = time.time()
        lst.append(cplex_solver(graph))
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the total time taken
        totalTime.append(elapsed_time)

    print(len(ds))
    print(lst)
    print(totalTime)
    print(numpy.average(totalTime))
    print(numpy.average(lst))

def test_balanced(fileName = 'nx_test_generated_graph_n40_100_d8_12_t500.pkl'):
    ds = open_file(fileName)
    lst = []
    totalTime = []
    for key, (dgl_graph, adjacency_matrix,graph, terminals) in ds.items():
        start_time = time.time()
        lst.append(cplex_solver_balanced(graph))
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the total time taken
        totalTime.append(elapsed_time)

    print(len(ds))
    print(lst)
    print(totalTime)
    print(numpy.average(totalTime))
    print(numpy.average(lst))
# fileNameLst = []
# for i in fileNameLst:
#     test(i)

'''
fileName = 'nx_test_generated_graph_n40_100_d8_12_t500.pkl'
[341.0, 378.0, 209.0, 283.0, 423.0, 221.0, 347.0, 325.0, 361.0]
[1.5376088619232178, 1.2024531364440918, 1.0976030826568604, 14.042061805725098, 2.0971519947052, 1.0634591579437256, 1.3958826065063477, 2.2576708793640137, 1.973902702331543]
2.9630882475111218
320.8888888888889
'''

'''
test('nx_test_generated_graph_n101_200_d8_12_t500.pkl')
[811.0, 515.0, 501.0, 625.0, 655.0, 868.0, 622.0, 608.0, 561.0, 543.0, 377.0]
[2.151216983795166, 1.7118909358978271, 1.4152400493621826, 1.9154181480407715, 1.5739829540252686, 2.0973358154296875, 1.5242371559143066, 1.5846691131591797, 1.8768959045410156, 1.708371877670288, 1.8372979164123535]
3.763323350386186
607.8181818181819
'''

# test('nx_test_generated_graph_n200_300_d8_12_t500.pkl')
'''
[739.0, 978.0, 1049.0, 1180.0, 708.0, 1012.0, 892.0, 1056.0, 1155.0, 801.0, 967.0, 885.0]
[3.8403799533843994, 3.6361069679260254, 3.4469151496887207, 3.495950937271118, 3.4399421215057373, 3.836113214492798, 3.250499963760376, 4.101799964904785, 3.3421640396118164, 2.454317092895508, 5.900020122528076, 5.1659300327301025]
3.825844963391622
951.8333333333334
'''

# test('nx_test_generated_graph_n700_800_d8_12_t500.pkl')
'''
[2413.0, 3201.0, 2552.0, 2960.0, 2977.0, 3130.0, 2588.0]
[20.314695835113525, 19.61050796508789, 23.557710886001587, 21.676140069961548, 28.897568941116333, 22.615626096725464, 28.462023735046387]
23.59061050415039
2831.5714285714284
'''

# test('nx_test_generated_graph_n800_900_d8_12_t500.pkl')
'''
[3375.0, 2814.0, 3229.0, 3694.0, 3376.0, 2691.0, 3703.0, 3239.0, 3203.0]
[37.59176540374756, 29.256783962249756, 33.07463192939758, 40.33008885383606, 38.348684787750244, 33.84363794326782, 34.587623834609985, 37.06819701194763, 25.28944993019104]
34.3767626285553
3258.222222222222
'''

# test('nx_test_generated_graph_n900_999_d8_12_t500.pkl')
'''
[3203.0, 3415.0, 3593.0, 2838.0, 3032.0, 3904.0, 4286.0, 3122.0, 3666.0, 3040.0]
[54.738774061203, 39.25505709648132, 46.903695821762085, 52.35745906829834, 50.204627990722656, 34.57691407203674, 50.37816023826599, 36.017343044281006, 66.72213506698608, 45.8635139465332]
47.70176804065704
3409.9
'''
filenames = ['nx_test_generated_graph_n101_200_d8_12_t500.pkl', 'nx_test_generated_graph_n200_300_d8_12_t500.pkl',
             'nx_test_generated_graph_n700_800_d8_12_t500.pkl', 'nx_test_generated_graph_n800_900_d8_12_t500.pkl', 'nx_test_generated_graph_n900_999_d8_12_t500.pkl']

# for i in filenames:
#     test_balanced(i)
#     print('solulu -------')
# ds = open_file('nx_test_generated_graph_n800_900_d8_12_t500.pkl')
# print(cplex_solver_balanced(ds[0][2]))

G = generate_graph(500, 12, p=None, graph_type='reg', random_seed=0)
best_cut_value = cplex_solver(G)
# print("Best Partition:", best_partition)
print("Best Cut Value:", best_cut_value)