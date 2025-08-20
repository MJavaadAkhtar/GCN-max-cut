from python.commons import *
import traceback
from typing import Dict, List, Tuple, Optional

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

def process_graphs_from_folder(all_graphs: Dict, all_terminals: Dict, max_nodes: int, 
                             save_batch_size: Optional[int] = None, 
                             output_filename_prefix: str = "processed_graphs") -> Dict:
    """
    Process graphs from a folder/dictionary, normalize terminal nodes, and prepare them for training.
    
    :param all_graphs: Dictionary of NetworkX graphs
    :param all_terminals: Dictionary of terminal nodes for each graph
    :param max_nodes: Maximum number of nodes to extend matrices to
    :param save_batch_size: If provided, save results in batches of this size
    :param output_filename_prefix: Prefix for output filenames when saving in batches
    :return: Dictionary containing processed graph data
    """
    datasetItem = {}
    i = 0
    skipped = 0
    
    try:
        for filename, graph in all_graphs.items():
            terminals = all_terminals[filename]
            
            # Normalize terminal nodes to ensure they include [0, 1, 2]
            if 0 not in terminals and 1 not in terminals and 2 not in terminals:
                swap_graph_nodes(graph, {
                    terminals[0]: 0, terminals[1]: 1, terminals[2]: 2, 
                    0: terminals[0], 1: terminals[1], 2: terminals[2]
                })
            elif 0 not in terminals and 1 not in terminals and 2 in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {
                    terminals[1]: 0, terminals[2]: 1,  
                    0: terminals[1], 1: terminals[2]
                })
            elif 0 not in terminals and 1 in terminals and 2 not in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {
                    terminals[1]: 0, terminals[2]: 2,  
                    0: terminals[1], 2: terminals[2]
                })
            elif 0 in terminals and 1 not in terminals and 2 not in terminals:
                terminals.sort()
                swap_graph_nodes(graph, {
                    terminals[1]: 1, terminals[2]: 2,  
                    1: terminals[1], 2: terminals[2]
                })
            else:
                skipped += 1
                continue
            
            print(f"Terminal swapped {i}")
            
            # Convert to DGL graph
            graph_dgl = dgl.from_networkx(nx_graph=graph)
            graph_dgl = graph_dgl.to(TORCH_DEVICE)

            # Create adjacency matrix
            q_torch = qubo_dict_to_torch(graph, gen_adj_matrix(graph), 
                                       torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

            # Extend matrix to max_nodes size
            full_matrix = extend_matrix_torch_2(q_torch, max_nodes, 
                                              torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)

            # Store processed graph data
            datasetItem[i] = [graph_dgl, full_matrix, graph, [0, 1, 2]]
            i += 1

            # Save in batches if specified
            if save_batch_size and (i % save_batch_size == 0):
                batch_filename = f'{output_filename_prefix}_{i}.pkl'
                save_object(datasetItem, batch_filename)
                print(f"Saved batch to {batch_filename}")
                datasetItem = {}

            print(f"Graph finished: {i}")

    except Exception as e:
        print(f"Exception occurred at graph {i}, filename {filename}, terminals {terminals}")
        print(f"Graph nodes: {graph.number_of_nodes()}")
        print(traceback.format_exc())

    print(f"Skipped items: {skipped}")
    return datasetItem

def load_and_process_graphs(graphs_filename: str, terminals_filename: str, 
                          max_nodes: int, output_filename: str,
                          save_batch_size: Optional[int] = None) -> None:
    """
    Load graphs and terminals from pickle files, process them, and save the result.
    
    :param graphs_filename: Path to pickle file containing graphs dictionary
    :param terminals_filename: Path to pickle file containing terminals dictionary  
    :param max_nodes: Maximum number of nodes to extend matrices to
    :param output_filename: Output filename for the processed dataset
    :param save_batch_size: If provided, save results in batches of this size
    """
    print(f"Loading graphs from {graphs_filename}")
    all_graphs = open_file(graphs_filename)
    
    print(f"Loading terminals from {terminals_filename}")
    all_terminals = open_file(terminals_filename)
    
    print(f"Processing {len(all_graphs)} graphs with max_nodes={max_nodes}")
    processed_data = process_graphs_from_folder(
        all_graphs, all_terminals, max_nodes, 
        save_batch_size=save_batch_size,
        output_filename_prefix=output_filename.replace('.pkl', '')
    )
    
    if processed_data:  # Save remaining data if not saved in batches
        print(f"Saving final dataset to {output_filename}")
        save_object(processed_data, output_filename)

def save_processed_graphs(processed_data: Dict, filename: str) -> None:
    """
    Save processed graph data to a pickle file.
    
    :param processed_data: Dictionary containing processed graph data
    :param filename: Output filename for the saved data
    """
    save_object(processed_data, filename)
    print(f"Saved processed graphs to {filename}")

if __name__ == "__main__":
    # Example usage - process the existing dataset
    load_and_process_graphs(
        graphs_filename='nx_generated_graph_nfull_20.pkl',
        terminals_filename='terminals_nx_generated_graph_nfull_20.pkl', 
        max_nodes=502,
        output_filename='nx_DS_generated_graph_nfull_20.pkl',
        save_batch_size=200
    )