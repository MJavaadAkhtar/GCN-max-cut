
import networkx as nx
import random
import pickle
from typing import Dict, List, Tuple, Optional, Union

def save_object(obj, filename: str) -> None:
    """
    Save an object to a file using pickle.
    
    Args:
        obj: The object to save
        filename: Path where to save the object
    """
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename: str):
    """
    Load an object from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        The loaded object
    """
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def generate_graph(n: int, 
                d: Optional[int] = None, 
                p: Optional[float] = None, 
                graph_type: str = 'reg', 
                random_seed: int = 0,
                edge_weight: int = 1,
                edge_capacity: int = 1) -> nx.Graph:
    """
    Generate a NetworkX graph of specified type with given parameters.

    Args:
        n: Number of nodes in the graph
        d: Degree of each node (required for regular graphs)
        p: Probability of edge between nodes (required for probabilistic/erdos graphs)
        graph_type: Type of graph to generate ('reg', 'reg_random', 'prob', 'erdos')
        random_seed: Seed for random number generation
        edge_weight: Weight to assign to each edge
        edge_capacity: Capacity to assign to each edge

    Returns:
        nx.Graph: NetworkX graph with specified properties

    Raises:
        ValueError: If invalid parameters are provided
        NotImplementedError: If unsupported graph type is specified
    """
    # Validate input parameters
    if graph_type in ['reg', 'reg_random'] and d is None:
        raise ValueError("Degree 'd' must be provided for regular graphs")
    if graph_type in ['prob', 'erdos'] and p is None:
        raise ValueError("Probability 'p' must be provided for probabilistic graphs")
    if n < 1:
        raise ValueError("Number of nodes must be positive")
    if d is not None and d >= n:
        raise ValueError("Degree must be less than number of nodes")
    if p is not None and (p < 0 or p > 1):
        raise ValueError("Probability must be between 0 and 1")

    # Generate the appropriate type of graph
    if graph_type == 'reg':
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'reg_random':
        nx_temp = nx.random_regular_graph(d=d, n=n)
    elif graph_type == 'prob':
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'Graph type {graph_type} not supported')

    # Ensure consistent node ordering
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)

    # Add edge properties
    for u, v in nx_graph.edges():
        nx_graph[u][v]['weight'] = edge_weight
        nx_graph[u][v]['capacity'] = edge_capacity

    return nx_graph
def generate_unique_terminals(n: int, num_terminals: int = 3) -> List[int]:
    """
    Generate a specified number of unique terminal nodes for a graph.

    Args:
        n: Number of nodes in the graph
        num_terminals: Number of terminal nodes to generate

    Returns:
        List[int]: List of unique terminal node indices

    Raises:
        ValueError: If n is too small for the requested number of terminals
    """
    if n < num_terminals:
        raise ValueError(f"Graph size ({n}) must be >= number of terminals ({num_terminals})")
    return random.sample(range(n), num_terminals)


def generate_graph_dataset(
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    min_degree: int,
    max_degree: int,
    graph_type: str = 'reg',
    num_terminals: int = 3,
    edge_weight: int = 1,
    edge_capacity: int = 1
) -> Tuple[Dict[int, nx.Graph], Dict[int, List[int]]]:
    """
    Generate multiple graphs with random properties and their terminal nodes.

    Args:
        num_graphs: Number of graphs to generate
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        min_degree: Minimum degree for regular graphs
        max_degree: Maximum degree for regular graphs
        graph_type: Type of graph to generate
        num_terminals: Number of terminal nodes per graph
        edge_weight: Weight to assign to edges
        edge_capacity: Capacity to assign to edges

    Returns:
        Tuple[Dict[int, nx.Graph], Dict[int, List[int]]]: 
            - Dictionary mapping index to generated graph
            - Dictionary mapping index to terminal nodes for each graph
    """
    graphs = {}
    terminals = {}
    i = 0
    attempts = 0
    max_attempts = num_graphs * 2  # Allow some failed attempts

    while i < num_graphs and attempts < max_attempts:
        try:
            # Generate random number of nodes
            nodes = random.randint(min_nodes, max_nodes)
            degree = random.randint(min_degree, max_degree)

            # For regular graphs, n * d must be even
            if graph_type in ['reg', 'reg_random'] and (nodes * degree) % 2 != 0:
                attempts += 1
                continue

            # Generate graph
            graph = generate_graph(
                n=nodes,
                d=degree,
                graph_type=graph_type,
                random_seed=i,
                edge_weight=edge_weight,
                edge_capacity=edge_capacity
            )

            # Generate terminal nodes
            graph_terminals = generate_unique_terminals(nodes, num_terminals)

            graphs[i] = graph
            terminals[i] = graph_terminals
            i += 1

        except (ValueError, nx.NetworkXError) as e:
            attempts += 1
            continue

    if i < num_graphs:
        print(f"Warning: Only generated {i} valid graphs out of {num_graphs} requested")

    return graphs, terminals

def save_graphs_to_pickle(graphs: Dict[int, nx.Graph], filename: str) -> None:
    """
    Save graphs dictionary to a pickle file.
    
    Args:
        graphs: Dictionary mapping indices to NetworkX graphs
        filename: Path where to save the graphs
    """
    save_object(graphs, filename)

def save_terminals_to_pickle(terminals: Dict[int, List[int]], filename: str) -> None:
    """
    Save terminals dictionary to a pickle file.
    
    Args:
        terminals: Dictionary mapping indices to terminal node lists
        filename: Path where to save the terminals
    """
    save_object(terminals, filename)

if __name__ == "__main__":
    # Example usage
    
    # 1. Generate a single graph
    graph = generate_graph(
        n=100,  # 100 nodes
        d=3,    # 3-regular graph
        graph_type='reg',
        random_seed=42
    )
    print(f"Generated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # 2. Generate terminal nodes for the graph
    terminals = generate_unique_terminals(100, num_terminals=3)
    print(f"Generated terminal nodes: {terminals}")
    
    # 3. Generate multiple graphs with random properties
    graphs, graph_terminals = generate_graph_dataset(
        num_graphs=5,
        min_nodes=50,
        max_nodes=100,
        min_degree=3,
        max_degree=5,
        graph_type='reg',
        num_terminals=3
    )
    
    print(f"\nGenerated {len(graphs)} graphs with the following properties:")
    for i, graph in graphs.items():
        print(f"Graph {i}:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        print(f"  Terminal nodes: {graph_terminals[i]}")
    
    # 4. Save graphs and terminals to pickle files
    save_graphs_to_pickle(graphs, 'example_graphs.pkl')
    save_terminals_to_pickle(graph_terminals, 'example_terminals.pkl')
    print(f"\nSaved graphs to 'example_graphs.pkl' and terminals to 'example_terminals.pkl'")