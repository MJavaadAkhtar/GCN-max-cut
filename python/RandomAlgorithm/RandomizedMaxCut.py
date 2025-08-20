"""
Randomized k-way Max-Cut Algorithm

This module provides a randomized algorithm for solving the k-way max-cut problem
on graphs. It includes utilities for graph generation, cut value calculation,
benchmarking, and evaluation.

Key Features:
- Randomized k-way max-cut algorithm with early stopping
- Support for fixed terminal nodes (useful for neural network comparison)
- Comprehensive benchmarking and evaluation functions
- Regular graph generation utilities

Author: Research Team
"""

import networkx as nx
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List


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


def benchmark_algorithm(
    node_sizes: List[int] = [1000, 2000, 3000, 4000, 5000],
    partition_sizes: List[int] = [3, 4, 5, 10],
    degree: int = 8,
    max_iterations: int = 10000,
    threshold: int = 1,
    patience: int = 50
) -> Dict:
    """
    Comprehensive benchmarking of the randomized k-way max-cut algorithm.
    Tests different partition sizes and graph sizes.
    
    :param node_sizes: List of node counts to test
    :param partition_sizes: List of k values (number of partitions) to test
    :param degree: Regular graph degree
    :param max_iterations: Maximum iterations per graph
    :param threshold: Early stopping threshold
    :param patience: Early stopping patience
    :return: Dictionary containing benchmark results
    """
    results = {}
    
    for k in partition_sizes:
        results[k] = []
        print(f"Testing k={k} partitions:")
        
        for n in node_sizes:
            # Create test graph
            G = create_random_regular_graph(n, degree)
            
            # Run algorithm and measure time
            start_time = time.time()
            best_cut_value, best_partition = randomized_k_way_maxcut(
                G, k=k, max_iterations=max_iterations, threshold=threshold, patience=patience
            )
            end_time = time.time()
            
            runtime = end_time - start_time
            results[k].append({
                'n': n,
                'best_cut_value': best_cut_value,
                'runtime': runtime
            })
            
            print(f"  n={n}: cut_value={best_cut_value}, runtime={runtime:.2f}s")
        
        print("-------")
    
    return results


def analyze_results(results: Dict):
    """
    Analyze and visualize benchmark results.
    
    :param results: Results dictionary from benchmark_algorithm()
    """
    # Extract data for plotting
    partition_sizes = list(results.keys())
    
    # Plot runtime vs graph size for different k values
    plt.figure(figsize=(15, 5))
    
    # Runtime plot
    plt.subplot(1, 3, 1)
    for k in partition_sizes:
        n_values = [result['n'] for result in results[k]]
        runtimes = [result['runtime'] for result in results[k]]
        plt.plot(n_values, runtimes, 'o-', label=f'k={k}')
    
    plt.xlabel('Graph Size (n)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Graph Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cut value plot
    plt.subplot(1, 3, 2)
    for k in partition_sizes:
        n_values = [result['n'] for result in results[k]]
        cut_values = [result['best_cut_value'] for result in results[k]]
        plt.plot(n_values, cut_values, 's-', label=f'k={k}')
    
    plt.xlabel('Graph Size (n)')
    plt.ylabel('Best Cut Value')
    plt.title('Cut Value vs Graph Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cut value normalized by number of edges
    plt.subplot(1, 3, 3)
    for k in partition_sizes:
        n_values = [result['n'] for result in results[k]]
        # For d-regular graph with degree 8, number of edges = n*d/2 = n*4
        normalized_cuts = [result['best_cut_value'] / (result['n'] * 4) for result in results[k]]
        plt.plot(n_values, normalized_cuts, '^-', label=f'k={k}')
    
    plt.xlabel('Graph Size (n)')
    plt.ylabel('Cut Value / Number of Edges')
    plt.title('Normalized Cut Value vs Graph Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for k in partition_sizes:
        runtimes = [result['runtime'] for result in results[k]]
        cut_values = [result['best_cut_value'] for result in results[k]]
        
        print(f"k={k} partitions:")
        print(f"  Average runtime: {np.mean(runtimes):.2f}s")
        print(f"  Average cut value: {np.mean(cut_values):.0f}")
        print(f"  Cut value range: {min(cut_values):.0f} - {max(cut_values):.0f}")
        print()


def test_fixed_terminals(
    n: int = 1000, 
    degree: int = 8, 
    k: int = 3,
    max_iterations: int = 1000,
    threshold: int = 1,
    patience: int = 50,
    random_seed: int = 42
):
    """
    Test the algorithm with fixed terminal nodes (useful for neural network comparison).
    
    :param n: Number of nodes in test graph
    :param degree: Regular graph degree
    :param k: Number of partitions
    :param max_iterations: Maximum iterations
    :param threshold: Early stopping threshold
    :param patience: Early stopping patience
    :param random_seed: Random seed for reproducibility
    """
    print("Testing with fixed terminals...")
    
    # Create a test graph
    G = create_random_regular_graph(n=n, degree=degree, random_seed=random_seed)
    
    # Test without fixed terminals
    start_time = time.time()
    cut_value_free, partition_free = randomized_k_way_maxcut(
        G, k=k, max_iterations=max_iterations, threshold=threshold, patience=patience
    )
    time_free = time.time() - start_time
    
    # Test with fixed terminals (nodes 0, 1, 2 fixed to partitions 0, 1, 2)
    fixed_terminals = {i: i for i in range(k)}
    start_time = time.time()
    cut_value_fixed, partition_fixed = randomized_k_way_maxcut(
        G, k=k, max_iterations=max_iterations, threshold=threshold, 
        patience=patience, fixed_terminals=fixed_terminals
    )
    time_fixed = time.time() - start_time
    
    print(f"Without fixed terminals:")
    print(f"  Cut value: {cut_value_free}")
    print(f"  Runtime: {time_free:.2f}s")
    terminal_assignments_free = {i: partition_free[i] for i in range(k)}
    print(f"  Terminal assignments: {terminal_assignments_free}")
    
    print(f"\nWith fixed terminals:")
    print(f"  Cut value: {cut_value_fixed}")
    print(f"  Runtime: {time_fixed:.2f}s")
    terminal_assignments_fixed = {i: partition_fixed[i] for i in range(k)}
    print(f"  Terminal assignments: {terminal_assignments_fixed}")
    
    print(f"\nPerformance impact of constraints:")
    if cut_value_free > 0:
        percentage_diff = ((cut_value_free - cut_value_fixed) / cut_value_free * 100)
        print(f"  Cut value difference: {cut_value_free - cut_value_fixed} ({percentage_diff:.1f}%)")
    else:
        print(f"  Cut value difference: {cut_value_free - cut_value_fixed}")
    print(f"  Runtime difference: {time_fixed - time_free:.2f}s")


def quick_demo():
    """
    Quick demonstration of the randomized max-cut algorithm.
    """
    print("Quick Demo: Randomized k-way Max-Cut Algorithm")
    print("=" * 50)
    
    # Create a small test graph
    G = create_random_regular_graph(n=500, degree=8, random_seed=42)
    
    # Run the algorithm
    start_time = time.time()
    best_cut_value, best_partition = randomized_k_way_maxcut(
        G, k=3, max_iterations=1000, threshold=1, patience=50, random_seed=42
    )
    end_time = time.time()
    
    print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print(f"Best cut value: {best_cut_value}")
    print(f"Runtime: {end_time - start_time:.2f} seconds")
    
    # Show partition distribution
    from collections import Counter
    partition_counts = Counter(best_partition.values())
    print(f"Partition distribution: {dict(partition_counts)}")
    
    print(f"First 10 node assignments: {dict(list(best_partition.items())[:10])}")


if __name__ == "__main__":
    # Run quick demo
    quick_demo()