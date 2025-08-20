from python.commons import *
import time
import random
import networkx as nx
from typing import Dict, Tuple, Optional, List


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

def calculate_cut_value(graph, partition: Dict[int, int]) -> int:
    """
    Calculate the total cut value for a given partition of a graph.
    
    :param graph: NetworkX graph with weighted edges
    :param partition: Dictionary mapping node -> partition number
    :return: Total cut value (sum of weights of edges crossing partitions)
    """
    cut_value = 0
    for u, v, data in graph.edges(data=True):
        if partition[u] != partition[v]:
            cut_value += data.get('weight', 1)
    return cut_value


def create_random_regular_graph(n: int, degree: int = 8, random_seed: Optional[int] = None) -> nx.Graph:
    """
    Creates a random d-regular graph with n nodes using NetworkX.
    Assigns weight=1 to each edge.

    :param n: Number of nodes in the graph
    :param degree: The regular degree (each node has exactly 'degree' neighbors)
    :param random_seed: Optional seed for reproducibility
    :return: A NetworkX Graph object with weighted edges
    """
    # Generate a random d-regular graph
    if random_seed is not None:
        G = nx.random_regular_graph(d=degree, n=n, seed=random_seed)
    else:
        G = nx.random_regular_graph(d=degree, n=n)

    # Assign weight=1 to every edge
    for (u, v) in G.edges():
        G[u][v]['weight'] = 1

    return G


def randomized_k_way_maxcut(
    graph, 
    k: int = 3, 
    max_iterations: int = 1000, 
    threshold: int = 0, 
    patience: int = 10,
    fixed_terminals: Optional[Dict[int, int]] = None,
    random_seed: Optional[int] = None
) -> Tuple[int, Dict[int, int]]:
    """
    A randomized k-way Max-Cut algorithm with early stopping.

    :param graph: A NetworkX graph with weighted edges (default weight=1)
    :param k: The number of partitions for the k-way Max-Cut
    :param max_iterations: The maximum number of random assignments to try
    :param threshold: Minimum improvement needed to reset the patience counter
    :param patience: If no improvement above threshold happens for 'patience' consecutive iterations, stop early
    :param fixed_terminals: Optional dict mapping specific nodes to specific partitions
    :param random_seed: Optional seed for reproducibility
    :return: (best_cut_value, best_partition) where best_partition is a dict node->partition (0..k-1)
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    best_cut_value = 0
    best_partition = None
    nodes = list(graph.nodes())
    
    # Track iterations since improvement for early stopping
    iterations_since_improvement = 0

    for i in range(max_iterations):
        # Start with fixed terminals if provided
        if fixed_terminals:
            partition = fixed_terminals.copy()
            remaining_nodes = [n for n in nodes if n not in fixed_terminals]
        else:
            partition = {}
            remaining_nodes = nodes

        # Randomly assign remaining nodes to partitions
        for node in remaining_nodes:
            partition[node] = random.randint(0, k - 1)

        # Calculate the cut value for this partition
        cut_value = calculate_cut_value(graph, partition)

        # Check improvement
        if cut_value > best_cut_value + threshold:
            best_cut_value = cut_value
            best_partition = partition.copy()
            iterations_since_improvement = 0  # reset counter
        else:
            iterations_since_improvement += 1

        # Early stopping check
        if iterations_since_improvement >= patience:
            break

    return best_cut_value, best_partition

def evaluate_algorithm_on_graphs(
    graphs: List, 
    k: int = 3, 
    max_iterations: int = 1000,
    threshold: int = 0,
    patience: int = 10,
    fixed_terminals: Optional[Dict[int, int]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate the randomized k-way max-cut algorithm on a list of graphs.
    
    :param graphs: List of NetworkX graphs to evaluate
    :param k: Number of partitions
    :param max_iterations: Maximum iterations per graph
    :param threshold: Early stopping threshold
    :param patience: Early stopping patience
    :param fixed_terminals: Optional fixed terminal assignments
    :return: Dictionary with statistics (mean_cut_value, total_time)
    """
    cut_values = []
    start_time = time.time()
    
    for graph in graphs:
        best_cut_value, _ = randomized_k_way_maxcut(
            graph, k, max_iterations, threshold, patience, fixed_terminals
        )
        cut_values.append(best_cut_value)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return {
        'mean_cut_value': np.mean(cut_values),
        'total_time': elapsed_time,
        'cut_values': cut_values
    }

def main():
    """
    Example usage of the randomized k-way max-cut algorithm.
    """
    # Create a test graph
    G = create_random_regular_graph(n=500, degree=12, random_seed=0)
    
    # Run the algorithm
    start_time = time.time()
    best_cut_value, best_partition = randomized_k_way_maxcut(
        G, k=3, max_iterations=1000, threshold=1, patience=50
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Best Cut Value: {best_cut_value}")
    print(f"Runtime: {elapsed_time:.2f} seconds")
    print(f"Partition example (first 10 nodes): {dict(list(best_partition.items())[:10])}")

if __name__ == "__main__":
    main()