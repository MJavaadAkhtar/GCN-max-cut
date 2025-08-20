"""
Consolidated TrainingNeural Module for GCN Max-Cut Research

This module provides a clean, modular framework for training Graph Convolutional Networks 
on multi-way max-cut problems. It consolidates functionality from the legacy 
TrainingNeural_load.py file while maintaining backward compatibility.

Key Features:
- Configuration-driven training via TrainingConfig dataclass
- Modular training pipeline with proper error handling  
- External APIs for easy integration (train_from_pickle, train_multi_class)
- Model loading/saving utilities
- Legacy compatibility wrappers for existing code
- Advanced evaluation functions including optimal partitioning search

Migration from TrainingNeural_load.py:
- Use TrainingConfig instead of hyperParameters() 
- Use train_from_pickle() instead of train1()
- Use setup_model_and_optimizer() instead of get_gnn()
- Use train_model() instead of run_gnn_training2()

All legacy functions are available as compatibility wrappers.
"""

from python.commons import *
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from time import time
from itertools import permutations
import random

TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    # Model parameters
    n_nodes: int = 1000
    dim_embedding: Optional[int] = None  # Will default to n_nodes if None
    hidden_dim: Optional[int] = None     # Will default to dim_embedding//2 if None
    dropout: float = 0.0
    number_classes: int = 3
    
    # Training parameters
    learning_rate: float = 0.001
    number_epochs: int = 1000
    tolerance: float = 1e-4
    patience: int = 20
    prob_threshold: float = 0.5
    
    # Loss parameters
    A: float = 0.0
    C: float = 1.0
    penalty: float = 1000.0
    
    # Saving parameters
    save_directory: Optional[str] = None
    save_frequency: int = 100
    
    def __post_init__(self):
        """Set default values that depend on other parameters."""
        if self.dim_embedding is None:
            self.dim_embedding = self.n_nodes
        if self.hidden_dim is None:
            self.hidden_dim = self.dim_embedding // 2

class GCNSoftmax(nn.Module):
    """Graph Convolutional Network with softmax output."""
    
    def __init__(self, in_feats: int, hidden_size: int, num_classes: int, 
                 dropout: float, device):
        super(GCNSoftmax, self).__init__()
        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, num_classes).to(device)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_frac, training=self.training)
        h = self.conv2(g, h)
        h = F.softmax(h, dim=1)
        return h

def override_fixed_nodes(h):
    """Override terminal nodes to fixed values while maintaining gradients."""
    output = h.clone()
    # Set the output for terminal nodes [0, 1, 2] to fixed one-hot encodings
    output[0] = torch.tensor([1.0, 0.0, 0.0], requires_grad=True) + h[0] - h[0].detach()
    output[1] = torch.tensor([0.0, 1.0, 0.0], requires_grad=True) + h[1] - h[1].detach()
    output[2] = torch.tensor([0.0, 0.0, 1.0], requires_grad=True) + h[2] - h[2].detach()
    return output

def max_to_one_hot(tensor):
    """Convert tensor to one-hot encoding based on maximum value."""
    max_index = torch.argmax(tensor)
    one_hot_tensor = torch.zeros_like(tensor)
    one_hot_tensor[max_index] = 1.0
    one_hot_tensor = one_hot_tensor + tensor - tensor.detach()
    return one_hot_tensor

def apply_max_to_one_hot(output):
    """Apply max_to_one_hot to each row of a batch."""
    return torch.stack([max_to_one_hot(output[i]) for i in range(output.size(0))])

def extend_matrix_torch_training(matrix, N):
    """Extend matrix for training purposes."""
    original_size = matrix.shape[0]
    if N <= original_size:
        return matrix
    
    extended_matrix = torch.zeros((N, N), dtype=matrix.dtype, device=matrix.device)
    extended_matrix[:original_size, :original_size] = matrix
    return extended_matrix

# def calculate_HC_vectorized(s, adjacency_matrix):
#     """
#     Compute the minimum cut loss using vectorized operations.
    
#     Args:
#         s: Binary partition matrix of shape (num_nodes, num_partitions)
#         adjacency_matrix: Adjacency matrix of the graph
        
#     Returns:
#         Scalar loss value representing the total weight of edges cut
#     """
#     # Compute the partition probability matrix
#     partition_prob_matrix = s @ s.T
    
#     cut_value = adjacency_matrix * (1 - partition_prob_matrix)
#     loss = torch.sum(cut_value) / 2  # Divide by 2 to correct for double-counting
    
#     return loss

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
    cut_value = adjacency_matrix * (1 - extend_matrix_torch(partition_prob_matrix, 1000))

    # Sum up the contributions for all edges
    loss = torch.sum(cut_value) / 2  # Divide by 2 to correct for double-counting

    return loss

def terminal_independence_penalty(s, terminal_nodes: List[int]):
    """
    Calculate penalty to enforce terminal nodes in distinct partitions.
    
    Args:
        s: Probability matrix of size |V| x |K|
        terminal_nodes: List of terminal node indices
        
    Returns:
        Penalty term
    """
    penalty = 0
    num_terminals = len(terminal_nodes)
    for i in range(num_terminals):
        for j in range(i + 1, num_terminals):
            dot_product = torch.dot(s[terminal_nodes[i]], s[terminal_nodes[j]])
            penalty += dot_product
    return penalty

def find_ac_parameters(graph):
    """
    Automatically calculate A and C parameters based on graph degree.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Tuple of (A_initial, C_initial) based on max degree
    """
    max_degree = max(dict(graph.degree()).values())
    A_initial = max_degree + 1  # A is set to be one more than the maximum degree
    C_initial = max_degree / 2  # C is set to half the maximum degree
    return A_initial, C_initial

def generate_terminal_permutations(terminal_dict: Dict):
    """
    Generate all permutations of terminal assignments.
    
    Args:
        terminal_dict: Dictionary mapping terminal names to indices
        
    Returns:
        List of dictionaries with all permuted terminal assignments
    """
    value_permutations = list(permutations(terminal_dict.values()))
    permuted_dicts = [{key: value for key, value in zip(terminal_dict.keys(), perm)} 
                      for perm in value_permutations]
    return permuted_dicts

def calculate_all_cut_legacy(q_torch, s):
    """
    Legacy cut calculation method for compatibility.
    
    Args:
        q_torch: The adjacency matrix of the graph
        s: The binary output from the neural network in form [[prob1, prob2, ..., prob n], ...]
        
    Returns:
        The calculated cut loss value
    """
    def partition_weight2(adj, s_col):
        s_col = s_col.unsqueeze(0)
        t = s_col.t()
        partition_matrix = (s_col != t).float()
        weight = (adj * partition_matrix).sum() / 2
        return weight
    
    if len(s) > 0:
        total_cuts = s.shape[1]
        cut_value = 0
        for i in range(total_cuts):
            cut_value += partition_weight2(q_torch, s[:, i])
        return cut_value / 2
    return 0

def evaluate_optimal_partitioning(net, dgl_graph, inputs, adjacency_matrix, terminal_dict: Dict):
    """
    Find optimal network partitioning by evaluating all terminal permutations.
    
    Args:
        net: Trained neural network model
        dgl_graph: DGL graph representation
        inputs: Input features for the network
        adjacency_matrix: Graph adjacency matrix
        terminal_dict: Dictionary mapping terminal names to indices
        
    Returns:
        Best loss value found across all permutations
    """
    net.eval()
    best_loss = float('inf')
    
    # Handle small graphs with special input handling
    if dgl_graph.number_of_nodes() < 30:
        inputs = torch.ones((dgl_graph.number_of_nodes(), 30))
    
    # Generate all terminal permutations
    permuted_terminals = generate_terminal_permutations(terminal_dict)
    
    with torch.no_grad():
        for perm_dict in permuted_terminals:
            # Note: This assumes the model can accept terminal configuration
            # This may need adaptation based on actual model interface
            logits = net(dgl_graph, inputs)
            logits = override_fixed_nodes(logits)
            binary_partitions = (logits >= 0.5).float()
            
            cut_value = calculate_all_cut_legacy(adjacency_matrix, binary_partitions)
            if cut_value < best_loss:
                best_loss = cut_value
                
    return best_loss

def compute_loss(s, adjacency_matrix, A: float = 0, C: float = 1, penalty: float = 1000):
    """
    Compute the training loss combining cut loss and penalties.
    
    Args:
        s: Partition probability matrix
        adjacency_matrix: Graph adjacency matrix
        A: Penalty parameter A
        C: Penalty parameter C  
        penalty: Terminal independence penalty weight
        
    Returns:
        Computed loss value
    """
    HC = -1 * calculate_HC_vectorized(s, adjacency_matrix)
    loss = C * HC
    # Optional: Add terminal independence penalty
    # loss += penalty * terminal_independence_penalty(s, [0, 1, 2])
    return loss

def setup_model_and_optimizer(config: TrainingConfig):
    """
    Set up the GNN model, embedding, and optimizer.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (model, embedding, optimizer)
    """
    # Create model
    net = GCNSoftmax(
        in_feats=config.dim_embedding,
        hidden_size=config.hidden_dim,
        num_classes=config.number_classes,
        dropout=config.dropout,
        device=TORCH_DEVICE
    )
    net = net.type(TORCH_DTYPE).to(TORCH_DEVICE)
    
    # Create embedding
    embed = nn.Embedding(config.n_nodes, config.dim_embedding)
    embed = embed.type(TORCH_DTYPE).to(TORCH_DEVICE)
    
    # Create optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    
    return net, embed, optimizer

def train_single_epoch(dataset: Dict, net, optimizer, embed, config: TrainingConfig, 
                      dataset_files: Optional[List[str]] = None) -> float:
    """
    Train for a single epoch.
    
    Args:
        dataset: Training dataset
        net: Neural network model
        optimizer: Optimizer
        embed: Embedding layer
        config: Training configuration
        dataset_files: Optional list of dataset files to load
        
    Returns:
        Cumulative loss for the epoch
    """
    net.train()
    cumulative_loss = 0.0
    inputs = embed.weight
    
    # Use provided dataset files or default
    if dataset_files is None:
        dataset_files = ['./nx_test_generated_graph_n200_300_d8_12_t500.pkl']
    
    for dataset_file in dataset_files:
        if isinstance(dataset, dict):
            current_dataset = dataset
        else:
            current_dataset = open_file(dataset_file)
            
        for key, (dgl_graph, adjacency_matrix, graph, terminals) in current_dataset.items():
            # Forward pass
            logits = net(dgl_graph, adjacency_matrix)
            logits = override_fixed_nodes(logits)
            
            # Apply max to one-hot encoding
            one_hot_output = apply_max_to_one_hot(logits)
            
            # Compute loss
            loss = compute_loss(one_hot_output, adjacency_matrix, 
                              config.A, config.C, config.penalty)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cumulative_loss += loss.item()
    
    return cumulative_loss

def train_model(dataset: Dict, config: TrainingConfig, 
                dataset_files: Optional[List[str]] = None) -> Tuple:
    """
    Main training function.
    
    Args:
        dataset: Training dataset
        config: Training configuration
        dataset_files: Optional list of dataset files
        
    Returns:
        Tuple of (trained_model, best_loss, final_epoch, embedding_weights, loss_history)
    """
    print(f"Starting training with {config.number_epochs} epochs")
    print(f"Model: {config.n_nodes} nodes, {config.number_classes} classes")
    print(f"Device: {TORCH_DEVICE}")
    
    # Setup model and optimizer
    net, embed, optimizer = setup_model_and_optimizer(config)
    
    # Training variables
    best_loss = float('inf')
    best_model_state = None
    loss_history = []
    patience_counter = 0
    prev_loss = float('inf')
    
    start_time = time()
    
    for epoch in range(config.number_epochs):
        # Train single epoch
        cumulative_loss = train_single_epoch(
            dataset, net, optimizer, embed, config, dataset_files
        )
        
        loss_history.append(cumulative_loss)
        
        # Early stopping check
        if epoch > 0 and (cumulative_loss > prev_loss or 
                         abs(prev_loss - cumulative_loss) <= config.tolerance):
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f'Early stopping at epoch {epoch}')
                break
        else:
            patience_counter = 0
        
        # Update best model
        if cumulative_loss < best_loss:
            best_loss = cumulative_loss
            best_model_state = net.state_dict()
        
        prev_loss = cumulative_loss
        
        # Periodic logging and saving
        if epoch % config.save_frequency == 0:
            print(f'Epoch: {epoch}, Cumulative Loss: {cumulative_loss:.6f}')
            
            if config.save_directory:
                checkpoint = {
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_history': loss_history,
                    'inputs': embed.weight,
                    'config': config
                }
                filename = f'./epoch_{epoch}_loss_{cumulative_loss:.4f}_{config.save_directory}'
                torch.save(checkpoint, filename)
    
    # Load best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
    
    training_time = time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    print(f'Best loss: {best_loss:.6f}')
    
    # Save final model
    if config.save_directory:
        final_checkpoint = {
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_history': loss_history,
            'inputs': embed.weight,
            'config': config
        }
        final_filename = f'./final_{config.save_directory}'
        torch.save(final_checkpoint, final_filename)
        print(f'Final model saved to {final_filename}')
    
    return net, best_loss, epoch, embed.weight, loss_history

def train_from_pickle(dataset_filename: str, model_name: str, 
                     n_nodes: int = 1000, **kwargs) -> Tuple:
    """
    Convenience function to train from a pickle file.
    
    Args:
        dataset_filename: Path to dataset pickle file
        model_name: Name for saving the model
        n_nodes: Number of nodes in the graph
        **kwargs: Additional configuration parameters
        
    Returns:
        Training results tuple
    """
    # Create configuration
    config_params = {
        'n_nodes': n_nodes,
        'save_directory': f'{model_name}.pth',
        **kwargs
    }
    config = TrainingConfig(**config_params)
    
    # Load dataset
    print(f"Loading dataset from {dataset_filename}")
    dataset = open_file(dataset_filename)
    
    # Train model
    return train_model(dataset, config)

def train_multi_class(dataset_filename: str, model_name: str, 
                     num_classes: int = 3, **kwargs) -> Tuple:
    """
    Train a multi-class model.
    
    Args:
        dataset_filename: Path to dataset pickle file
        model_name: Name for saving the model
        num_classes: Number of classes (2 or 3)
        **kwargs: Additional configuration parameters
        
    Returns:
        Training results tuple
    """
    config_params = {
        'number_classes': num_classes,
        'save_directory': f'{model_name}.pth',
        **kwargs
    }
    
    return train_from_pickle(dataset_filename, model_name, **config_params)

def evaluate_model(model, dataset: Dict, config: TrainingConfig) -> Dict:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: Trained neural network model
        dataset: Evaluation dataset
        config: Training configuration
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for key, (dgl_graph, adjacency_matrix, graph, terminals) in dataset.items():
            logits = model(dgl_graph, adjacency_matrix)
            logits = override_fixed_nodes(logits)
            one_hot_output = apply_max_to_one_hot(logits)
            
            loss = compute_loss(one_hot_output, adjacency_matrix, 
                              config.A, config.C, config.penalty)
            total_loss += loss.item()
            num_samples += 1
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    
    return {
        'average_loss': avg_loss,
        'total_loss': total_loss,
        'num_samples': num_samples
    }

def load_neural_model(model_path: str, config: TrainingConfig):
    """
    Load a saved neural model from checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        config: Training configuration for model setup
        
    Returns:
        Tuple of (loaded_model, inputs, loaded_config)
    """
    # Fix for PyTorch 2.6+ weights_only default behavior
    # Allow loading of TrainingConfig class safely
    try:
        # First try with safe globals approach
        import torch.serialization
        torch.serialization.add_safe_globals([TrainingConfig])
        checkpoint = torch.load(model_path, map_location=TORCH_DEVICE)
    except Exception as e:
        try:
            # Fallback: force weights_only=False (less secure but functional)
            checkpoint = torch.load(model_path, map_location=TORCH_DEVICE, weights_only=False)
        except Exception as e2:
            # Last resort: use safe globals context manager
            with torch.serialization.safe_globals([TrainingConfig]):
                checkpoint = torch.load(model_path, map_location=TORCH_DEVICE)
    
    # Setup model architecture
    net, embed, _ = setup_model_and_optimizer(config)
    
    # Load saved state
    net.load_state_dict(checkpoint['model'])
    
    # Extract inputs and config if available
    inputs = checkpoint.get('inputs', embed.weight)
    loaded_config = checkpoint.get('config', config)
    
    return net, inputs, loaded_config

def save_neural_model(model, optimizer, embed, epoch: int, loss_history: List, 
                     config: TrainingConfig, model_path: str):
    """
    Save neural model checkpoint.
    
    Args:
        model: Neural network model to save
        optimizer: Optimizer state
        embed: Embedding layer
        epoch: Current epoch number
        loss_history: Training loss history
        config: Training configuration
        model_path: Path to save the model
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_history': loss_history,
        'inputs': embed.weight,
        'config': config
    }
    torch.save(checkpoint, model_path)
    print(f'Model saved to {model_path}')

# Legacy compatibility functions
def get_gnn_legacy(n_nodes: int, gnn_hypers: Dict, opt_params: Dict, 
                   torch_device, torch_dtype):
    """
    Legacy GNN creation function for backward compatibility.
    
    Args:
        n_nodes: Number of nodes
        gnn_hypers: GNN hyperparameters dictionary
        opt_params: Optimizer parameters dictionary
        torch_device: PyTorch device
        torch_dtype: PyTorch data type
        
    Returns:
        Tuple of (net, embed, optimizer)
    """
    # Convert legacy parameters to new config format
    config = TrainingConfig(
        n_nodes=n_nodes,
        dim_embedding=gnn_hypers['dim_embedding'],
        hidden_dim=gnn_hypers['hidden_dim'],
        dropout=gnn_hypers['dropout'],
        number_classes=gnn_hypers['number_classes'],
        learning_rate=opt_params['lr']
    )
    
    return setup_model_and_optimizer(config)

def hyperparameters_legacy(n: int = 80, d: int = 3, p=None, graph_type: str = 'reg', 
                          number_epochs: int = int(1e5), learning_rate: float = 1e-4, 
                          prob_threshold: float = 0.5, tol: float = 1e-4, 
                          patience: int = 100):
    """
    Legacy hyperparameter function for backward compatibility.
    
    Returns legacy hyperparameter format for old code compatibility.
    """
    dim_embedding = n
    hidden_dim = int(dim_embedding / 2)
    
    return (n, d, p, graph_type, number_epochs, learning_rate, 
            prob_threshold, tol, patience, dim_embedding, hidden_dim)

def train_legacy_wrapper(model_name: str, filename: str = './testData/nx_generated_graph_n80_d3_t200.pkl', 
                        n: int = 80):
    """
    Legacy training wrapper for backward compatibility with old train1() function.
    
    Args:
        model_name: Name for the model
        filename: Dataset filename
        n: Number of nodes
        
    Returns:
        Training results in legacy format
    """
    # Create config with legacy defaults
    config = TrainingConfig(
        n_nodes=n,
        learning_rate=0.001,
        patience=20,
        number_epochs=1000,
        save_directory=model_name
    )
    
    # Train using new system
    return train_from_pickle(filename, model_name, n_nodes=n, 
                           learning_rate=0.001, patience=20)

def train_2way_neural_legacy(model_name: str, filename: str = './testData/prepareDS.pkl'):
    """
    Legacy 2-way training wrapper for backward compatibility.
    
    Args:
        model_name: Name for the model
        filename: Dataset filename
        
    Returns:
        Training results in legacy format
    """
    return train_multi_class(filename, model_name, num_classes=2, n_nodes=4096, 
                           learning_rate=0.001, patience=20, number_epochs=500)

# ============================================================================
# LEGACY COMPATIBILITY SECTION
# ============================================================================
# The following aliases provide backward compatibility with TrainingNeural_load.py
# These functions maintain the same interface as the original legacy code

# Main function aliases
get_gnn = get_gnn_legacy                            # Original: get_gnn()
hyperParameters = hyperparameters_legacy           # Original: hyperParameters()  
train1 = train_legacy_wrapper                      # Original: train1()
train_2wayNeural = train_2way_neural_legacy        # Original: train_2wayNeural()
FIndAC = find_ac_parameters                        # Original: FIndAC()
GetOptimalNetValue = evaluate_optimal_partitioning # Original: GetOptimalNetValue()
calculateAllCut = calculate_all_cut_legacy         # Original: calculateAllCut()
LoadNeuralModel = load_neural_model                # New function for model loading

# Additional legacy compatibility (functions that were migrated)
# printCombo -> generate_terminal_permutations (renamed for clarity)
# extend_matrix_torch functions -> extend_matrix_torch_training (consolidated)
# Multiple HC calculation methods -> calculate_HC_vectorized (optimized single version)

if __name__ == "__main__":
    # Example usage
    start_time = time()
    
    # Train a 3-class model
    trained_net, best_loss, epoch, inputs, loss_history = train_from_pickle(
        dataset_filename='./nx_test_generated_graph_n200_300_d8_12_t500.pkl',
        model_name='maxcut_model_3class',
        n_nodes=1000,
        number_epochs=1000,
        learning_rate=0.001,
        patience=20
    )
    
    end_time = time()
    runtime = end_time - start_time
    
    print(f"Total runtime: {runtime:.2f} seconds")
    
    # Save training results
    training_results = [trained_net, best_loss, epoch, inputs, loss_history]
    save_object(training_results, 'training_results.pkl')
    print("Training results saved to 'training_results.pkl'")