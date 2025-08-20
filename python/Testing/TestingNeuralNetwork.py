"""
Neural Network Testing Module

This module provides core functions for testing trained GCN models on 3-way max-cut problems.
It includes post-processing optimization, performance evaluation, and comprehensive analysis.

Based on functionality from NeuralTestCode.py with clean, modular implementation.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from typing import List, Dict, Tuple, Any, Optional
import random

def assign_partitions(node_probs: np.ndarray) -> List[int]:
    """
    Assign partitions based on node probabilities using random sampling.
    
    This implements the probabilistic post-processing approach where nodes are assigned
    to partitions based on cumulative probability distributions.
    
    Args:
        node_probs: Array of shape (num_nodes, num_classes) with probability distributions
        
    Returns:
        List of partition assignments for each node
    """
    partition_assignment = [0, 1, 2]  # Fixed assignments for terminal nodes
    
    for probs in node_probs[3:]:  # Skip first 3 terminal nodes
        rand_val = np.random.rand()
        cumulative_prob = 0
        
        for i, prob in enumerate(probs):
            cumulative_prob += prob
            if rand_val < cumulative_prob:
                partition_assignment.append(i)
                break
        else:
            # Fallback: assign to last partition if probabilities don't sum to 1
            partition_assignment.append(len(probs) - 1)
                
    return partition_assignment

def calculate_cut_value(partition_assignment: List[int], graph) -> int:
    """
    Calculate the cut value for a given partition assignment.
    
    Args:
        partition_assignment: List of partition assignments for each node
        graph: NetworkX graph
        
    Returns:
        Cut value (sum of edge weights crossing partitions)
    """
    cut_value = 0
    for u, v, data in graph.edges(data=True):
        if u < len(partition_assignment) and v < len(partition_assignment):
            if partition_assignment[u] != partition_assignment[v]:
                cut_value += data.get('weight', 1)
    return cut_value

def post_processing_optimization(node_probabilities: torch.Tensor, graph, iterations: int = 200) -> Tuple[List[int], int]:
    """
    Post-processing optimization to improve GCN results using random sampling.
    
    This function performs multiple random samplings based on the probability distributions
    and returns the partition assignment that achieves the best cut value.
    
    Args:
        node_probabilities: Tensor of node probability distributions
        graph: NetworkX graph
        iterations: Number of random sampling iterations
        
    Returns:
        Tuple of (best_partition_assignment, best_cut_value)
    """
    best_partition_assignment = None
    best_score = -float('inf')
    
    # Convert to numpy for easier manipulation
    if isinstance(node_probabilities, torch.Tensor):
        node_probs = node_probabilities.detach().cpu().numpy()
    else:
        node_probs = node_probabilities
    
    for _ in range(iterations):
        current_assignment = assign_partitions(node_probs)
        current_score = calculate_cut_value(current_assignment, graph)
        
        if current_score > best_score:
            best_score = current_score
            best_partition_assignment = current_assignment
            
    return best_partition_assignment, best_score

def simple_partition_assignment(node_probabilities: torch.Tensor) -> List[int]:
    """
    Simple partition assignment using argmax (no post-processing).
    
    Args:
        node_probabilities: Tensor of node probability distributions
        
    Returns:
        List of partition assignments
    """
    # Get the most likely partition for each node
    _, predicted_partitions = torch.max(node_probabilities, dim=1)
    
    # Convert to list and ensure terminal constraints
    partition_assignment = predicted_partitions.cpu().numpy().tolist()
    
    # Fix terminal node assignments
    if len(partition_assignment) >= 3:
        partition_assignment[0] = 0
        partition_assignment[1] = 1 
        partition_assignment[2] = 2
        
    return partition_assignment

def test_single_graph(model, dgl_graph, adjacency_matrix, nx_graph, terminals: List[int], 
                     post_processing_iterations: int = 200) -> Dict[str, Any]:
    """
    Test a single graph with both simple and post-processed neural network approaches.
    
    Args:
        model: Trained neural network model
        dgl_graph: DGL graph for neural network input
        adjacency_matrix: Adjacency matrix for neural network input
        nx_graph: NetworkX graph for cut value calculation
        terminals: Terminal node assignments
        post_processing_iterations: Number of iterations for post-processing
        
    Returns:
        Dictionary containing test results
    """
    try:
        # Get neural network predictions
        with torch.no_grad():
            node_probabilities = model(dgl_graph, adjacency_matrix)
        
        # Method 1: Simple assignment (argmax, no post-processing)
        simple_start_time = time()
        simple_assignment = simple_partition_assignment(node_probabilities)
        simple_cut_value = calculate_cut_value(simple_assignment, nx_graph)
        simple_time = time() - simple_start_time
        
        # Method 2: Post-processing optimization
        post_start_time = time()
        post_assignment, post_cut_value = post_processing_optimization(
            node_probabilities, 
            nx_graph, 
            post_processing_iterations
        )
        post_time = time() - post_start_time
        
        # Calculate improvement
        improvement = post_cut_value - simple_cut_value
        improvement_percent = (improvement / simple_cut_value * 100) if simple_cut_value > 0 else 0
        
        return {
            'success': True,
            'nodes': len(nx_graph.nodes()),
            'edges': len(nx_graph.edges()),
            'simple_cut': simple_cut_value,
            'simple_time': simple_time,
            'simple_assignment': simple_assignment,
            'post_cut': post_cut_value,
            'post_time': post_time,
            'post_assignment': post_assignment,
            'improvement': improvement,
            'improvement_percent': improvement_percent,
            'terminals': terminals,
            'node_probabilities': node_probabilities.detach().cpu().numpy()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'nodes': len(nx_graph.nodes()) if nx_graph else 0,
            'edges': len(nx_graph.edges()) if nx_graph else 0
        }

def test_multiple_graphs(model, processed_graphs: Dict, graph_sizes: List[int], 
                        post_processing_iterations: int = 200, verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Test multiple graphs and organize results by size.
    
    Args:
        model: Trained neural network model
        processed_graphs: Dictionary of processed graphs
        graph_sizes: List of graph sizes to test
        post_processing_iterations: Number of iterations for post-processing
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (individual_results, results_by_size)
    """
    if verbose:
        print("Testing neural network performance...")
        print("=" * 60)

    # Results storage
    test_results = []
    results_by_size = {}

    # Initialize results by size
    for size in graph_sizes:
        results_by_size[size] = {
            'simple': {'cut_values': [], 'times': []},
            'post_processed': {'cut_values': [], 'times': []}
        }

    total_graphs = len(processed_graphs)
    processed_count = 0

    # Debug: Check what the keys look like
    if verbose:
        print(f"Sample keys from processed_graphs: {list(processed_graphs.keys())[:3]}")

    for key, (dgl_graph, adjacency_matrix, nx_graph, terminals) in processed_graphs.items():
        processed_count += 1
        
        # Handle different key types (string or int)
        if isinstance(key, str):
            graph_name = key
            # Extract graph size from name
            try:
                graph_size = int(graph_name.split('_')[1][1:])  # Extract from "test_n{size}_..."
            except (IndexError, ValueError):
                # Fallback: try to get size from NetworkX graph
                graph_size = len(nx_graph.nodes())
        else:
            # If key is an integer, create a synthetic name and get size from graph
            graph_name = f"graph_{key}"
            graph_size = len(nx_graph.nodes())
            
            # Try to match to closest configured size
            closest_size = min(graph_sizes, key=lambda x: abs(x - graph_size))
            if abs(closest_size - graph_size) <= 5:  # Allow some tolerance
                graph_size = closest_size
        
        if verbose:
            print(f"\\nProcessing graph {processed_count}/{total_graphs}: {graph_name}")
            print(f"  Nodes: {len(nx_graph.nodes())}, Edges: {len(nx_graph.edges())}, Size category: {graph_size}")
        
        # Skip if graph size not in our test configuration
        if graph_size not in graph_sizes:
            if verbose:
                print(f"  Skipping: graph size {graph_size} not in test configuration")
            continue
        
        # Test the graph
        result = test_single_graph(
            model, dgl_graph, adjacency_matrix, nx_graph, terminals, 
            post_processing_iterations
        )
        
        if result['success']:
            # Add metadata
            result.update({
                'graph_name': graph_name,
                'graph_size': graph_size
            })
            test_results.append(result)
            
            # Add to size-specific results
            results_by_size[graph_size]['simple']['cut_values'].append(result['simple_cut'])
            results_by_size[graph_size]['simple']['times'].append(result['simple_time'])
            results_by_size[graph_size]['post_processed']['cut_values'].append(result['post_cut'])
            results_by_size[graph_size]['post_processed']['times'].append(result['post_time'])
            
            if verbose:
                print(f"  Simple GCN:      Cut = {result['simple_cut']}, Time = {result['simple_time']:.4f}s")
                print(f"  Post-processed:  Cut = {result['post_cut']}, Time = {result['post_time']:.4f}s")
                print(f"  Improvement:     {result['improvement']:+d} ({result['improvement_percent']:+.1f}%)")
        else:
            if verbose:
                print(f"  ✗ Error processing graph: {result['error']}")
        
        # Progress indicator
        if verbose and processed_count % 10 == 0:
            progress = (processed_count / total_graphs) * 100
            print(f"\\n--- Progress: {processed_count}/{total_graphs} ({progress:.1f}%) ---")

    if verbose:
        print(f"\\n{'='*60}")
        print("Neural network testing completed!")
        print(f"Successfully processed: {len(test_results)}/{total_graphs} graphs")

    return test_results, results_by_size

def analyze_results(test_results: List[Dict], results_by_size: Dict, graph_sizes: List[int]) -> Dict[str, Any]:
    """
    Analyze test results and generate comprehensive statistics.
    
    Args:
        test_results: List of individual test results
        results_by_size: Results organized by graph size
        graph_sizes: List of graph sizes tested
        
    Returns:
        Dictionary containing analysis results
    """
    if len(test_results) == 0:
        return {'error': 'No test results available'}
    
    # Overall statistics
    total_tests = len(test_results)
    simple_cuts = [r['simple_cut'] for r in test_results]
    post_cuts = [r['post_cut'] for r in test_results]
    improvements = [r['improvement'] for r in test_results]
    improvement_percents = [r['improvement_percent'] for r in test_results]
    
    simple_times = [r['simple_time'] for r in test_results]
    post_times = [r['post_time'] for r in test_results]
    
    # Count improvements
    better_count = sum(1 for imp in improvements if imp > 0)
    same_count = sum(1 for imp in improvements if imp == 0)
    worse_count = sum(1 for imp in improvements if imp < 0)
    
    # Calculate metrics
    avg_simple_cut = np.mean(simple_cuts)
    avg_post_cut = np.mean(post_cuts)
    avg_improvement = np.mean(improvements)
    avg_improvement_pct = np.mean(improvement_percents)
    std_improvement = np.std(improvements)
    
    avg_simple_time = np.mean(simple_times)
    avg_post_time = np.mean(post_times)
    avg_overhead = avg_post_time / avg_simple_time if avg_simple_time > 0 else 0
    
    # Size-specific analysis
    size_analysis = {}
    for size in sorted(graph_sizes):
        if size in results_by_size and results_by_size[size]['simple']['cut_values']:
            simple_vals = results_by_size[size]['simple']['cut_values']
            post_vals = results_by_size[size]['post_processed']['cut_values']
            simple_times_size = results_by_size[size]['simple']['times']
            post_times_size = results_by_size[size]['post_processed']['times']
            
            simple_avg = np.mean(simple_vals)
            post_avg = np.mean(post_vals)
            improvement_avg = post_avg - simple_avg
            improvement_pct = (improvement_avg / simple_avg * 100) if simple_avg > 0 else 0
            
            simple_time_avg = np.mean(simple_times_size)
            post_time_avg = np.mean(post_times_size)
            time_ratio = post_time_avg / simple_time_avg if simple_time_avg > 0 else 0
            
            size_analysis[size] = {
                'count': len(simple_vals),
                'simple_avg': simple_avg,
                'post_avg': post_avg,
                'improvement_avg': improvement_avg,
                'improvement_pct': improvement_pct,
                'simple_time_avg': simple_time_avg,
                'post_time_avg': post_time_avg,
                'time_ratio': time_ratio
            }
    
    return {
        'total_tests': total_tests,
        'avg_simple_cut': avg_simple_cut,
        'avg_post_cut': avg_post_cut,
        'avg_improvement': avg_improvement,
        'avg_improvement_pct': avg_improvement_pct,
        'std_improvement': std_improvement,
        'better_count': better_count,
        'same_count': same_count,
        'worse_count': worse_count,
        'avg_simple_time': avg_simple_time,
        'avg_post_time': avg_post_time,
        'avg_overhead': avg_overhead,
        'size_analysis': size_analysis,
        'improvement_rate': better_count / total_tests if total_tests > 0 else 0
    }

def print_analysis_report(analysis: Dict[str, Any], graph_sizes: List[int]):
    """
    Print a comprehensive analysis report.
    
    Args:
        analysis: Analysis results from analyze_results()
        graph_sizes: List of graph sizes tested
    """
    if 'error' in analysis:
        print(f"Analysis Error: {analysis['error']}")
        return
    
    print("Performance Analysis")
    print("=" * 60)
    
    total_tests = analysis['total_tests']
    
    print(f"Overall Results ({total_tests} graphs):")
    print(f"")
    print(f"Cut Value Performance:")
    print(f"  Simple GCN Average:     {analysis['avg_simple_cut']:.2f}")
    print(f"  Post-processed Average: {analysis['avg_post_cut']:.2f}")
    print(f"  Average Improvement:    {analysis['avg_improvement']:+.2f} ({analysis['avg_improvement_pct']:+.1f}%)")
    print(f"  Std Dev Improvement:    {analysis['std_improvement']:.2f}")
    print(f"")
    print(f"Improvement Distribution:")
    print(f"  Post-processing better: {analysis['better_count']}/{total_tests} ({analysis['improvement_rate']*100:.1f}%)")
    print(f"  Same performance:       {analysis['same_count']}/{total_tests} ({analysis['same_count']/total_tests*100:.1f}%)")
    print(f"  Post-processing worse:  {analysis['worse_count']}/{total_tests} ({analysis['worse_count']/total_tests*100:.1f}%)")
    print(f"")
    print(f"Runtime Performance:")
    print(f"  Simple GCN Average:     {analysis['avg_simple_time']:.4f}s")
    print(f"  Post-processed Average: {analysis['avg_post_time']:.4f}s")
    print(f"  Runtime Overhead:       {analysis['avg_overhead']:.1f}x")
    
    print(f"\\n{'='*60}")
    print("Results by Graph Size:")
    print(f"{'Size':<6} {'Count':<6} {'Simple':<8} {'Post':<8} {'Improvement':<12} {'Runtime':<10}")
    print(f"{'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
    
    for size in sorted(graph_sizes):
        if size in analysis['size_analysis']:
            sa = analysis['size_analysis'][size]
            print(f"{size:<6} {sa['count']:<6} {sa['simple_avg']:<8.1f} {sa['post_avg']:<8.1f} " +
                  f"{sa['improvement_pct']:<+7.1f}%     {sa['time_ratio']:<6.1f}x")

def create_visualizations(test_results: List[Dict], results_by_size: Dict, graph_sizes: List[int], 
                         save_path: Optional[str] = None) -> None:
    """
    Create comprehensive visualizations of the test results.
    
    Args:
        test_results: List of individual test results
        results_by_size: Results organized by graph size
        graph_sizes: List of graph sizes tested
        save_path: Optional path to save the plot
    """
    if len(test_results) == 0:
        print("No results to visualize!")
        return
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cut value comparison by graph size
    sizes = []
    simple_means = []
    post_means = []
    simple_stds = []
    post_stds = []
    
    for size in sorted(graph_sizes):
        if size in results_by_size and results_by_size[size]['simple']['cut_values']:
            sizes.append(size)
            simple_vals = results_by_size[size]['simple']['cut_values']
            post_vals = results_by_size[size]['post_processed']['cut_values']
            
            simple_means.append(np.mean(simple_vals))
            post_means.append(np.mean(post_vals))
            simple_stds.append(np.std(simple_vals))
            post_stds.append(np.std(post_vals))
    
    x = np.arange(len(sizes))
    width = 0.35
    
    ax1.bar(x - width/2, simple_means, width, label='Simple GCN', 
            yerr=simple_stds, capsize=5, alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, post_means, width, label='Post-processed GCN', 
            yerr=post_stds, capsize=5, alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Graph Size (nodes)')
    ax1.set_ylabel('Average Cut Value')
    ax1.set_title('Cut Value Comparison by Graph Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement distribution
    improvements = [r['improvement'] for r in test_results]
    ax2.hist(improvements, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='No improvement')
    ax2.axvline(x=np.mean(improvements), color='green', linestyle='-', alpha=0.8, 
                label=f'Mean: {np.mean(improvements):.1f}')
    ax2.set_xlabel('Improvement (Post-processed - Simple)')
    ax2.set_ylabel('Number of Graphs')
    ax2.set_title('Distribution of Post-processing Improvements')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Runtime comparison
    simple_time_means = []
    post_time_means = []
    
    for size in sizes:
        simple_times = results_by_size[size]['simple']['times']
        post_times = results_by_size[size]['post_processed']['times']
        simple_time_means.append(np.mean(simple_times))
        post_time_means.append(np.mean(post_times))
    
    ax3.bar(x - width/2, simple_time_means, width, label='Simple GCN', alpha=0.8, color='skyblue')
    ax3.bar(x + width/2, post_time_means, width, label='Post-processed GCN', alpha=0.8, color='lightgreen')
    
    ax3.set_xlabel('Graph Size (nodes)')
    ax3.set_ylabel('Average Runtime (seconds)')
    ax3.set_title('Runtime Comparison by Graph Size')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement percentage by graph size
    improvement_pcts = []
    for size in sizes:
        simple_vals = results_by_size[size]['simple']['cut_values']
        post_vals = results_by_size[size]['post_processed']['cut_values']
        improvements_size = [(post - simple) / simple * 100 if simple > 0 else 0 
                           for simple, post in zip(simple_vals, post_vals)]
        improvement_pcts.append(np.mean(improvements_size))
    
    ax4.bar(sizes, improvement_pcts, alpha=0.8, color='coral')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Graph Size (nodes)')
    ax4.set_ylabel('Average Improvement (%)')
    ax4.set_title('Post-processing Improvement by Graph Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

def save_results(test_results: List[Dict], results_by_size: Dict, analysis: Dict, 
                testing_config: Dict, model_config, output_file: str) -> None:
    """
    Save detailed results to a pickle file.
    
    Args:
        test_results: List of individual test results
        results_by_size: Results organized by graph size
        analysis: Analysis results
        testing_config: Testing configuration used
        model_config: Model configuration used
        output_file: Path to save the results
    """
    from commons import save_object
    
    detailed_results = {
        'individual_results': test_results,
        'results_by_size': results_by_size,
        'analysis': analysis,
        'testing_config': testing_config,
        'model_config': model_config,
        'timestamp': time()
    }
    
    save_object(detailed_results, output_file)
    print(f"Detailed results saved to: {output_file}")

def generate_summary_report(analysis: Dict, testing_config: Dict) -> str:
    """
    Generate a comprehensive summary report.
    
    Args:
        analysis: Analysis results
        testing_config: Testing configuration
        
    Returns:
        Formatted summary report string
    """
    if 'error' in analysis:
        return f"Summary Report Error: {analysis['error']}"
    
    total_tests = analysis['total_tests']
    better_count = analysis['better_count']
    avg_improvement = analysis['avg_improvement']
    avg_improvement_pct = analysis['avg_improvement_pct']
    avg_overhead = analysis['avg_overhead']
    
    report = f"""Neural Network Testing Summary
{'='*60}

Testing completed on {total_tests} graphs across {len(testing_config['graph_sizes'])} different sizes.

Key Findings:

1. Post-processing Effectiveness:
   - Improved results in {better_count}/{total_tests} cases ({better_count/total_tests*100:.1f}%)
   - Average improvement: {avg_improvement:+.2f} cut value ({avg_improvement_pct:+.1f}%)
   - Standard deviation: {analysis['std_improvement']:.2f}

2. Computational Cost:
   - Post-processing adds {avg_overhead:.1f}x runtime overhead
   - Simple GCN: {analysis['avg_simple_time']:.4f}s average
   - Post-processed: {analysis['avg_post_time']:.4f}s average

3. Scalability:"""

    for size in sorted(testing_config['graph_sizes']):
        if size in analysis['size_analysis']:
            sa = analysis['size_analysis'][size]
            report += f"\\n   - {size} nodes ({sa['count']} graphs): {sa['improvement_pct']:+.1f}% improvement"

    report += f"""

Recommendations:"""

    if avg_improvement > 0:
        report += f"\\n✓ Post-processing shows positive results on average"
        if better_count/total_tests >= 0.7:
            report += f"\\n✓ Post-processing improves results in most cases - recommended"
        else:
            report += f"\\n⚠ Post-processing is inconsistent - use with caution"
    else:
        report += f"\\n✗ Post-processing shows negative results on average - not recommended"

    if avg_overhead <= 2.0:
        report += f"\\n✓ Computational overhead is reasonable ({avg_overhead:.1f}x)"
    else:
        report += f"\\n⚠ High computational overhead ({avg_overhead:.1f}x) - consider cost vs benefit"

    report += f"""

Configuration Used:
  Model: {testing_config['model_path']}
  Post-processing iterations: {testing_config['post_processing_iterations']}
  Graph sizes tested: {testing_config['graph_sizes']}
  Graphs per size: {testing_config['graphs_per_size']}

{'='*60}
Testing completed successfully!"""

    return report