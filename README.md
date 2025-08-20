# GCN Max-Cut Research Repository

This repository implements Graph Convolutional Networks (GCNs) for solving 3-way max-cut problems on graph datasets. It provides a complete pipeline from graph generation to neural network training, testing, and visualization.

## Overview

The repository contains:
- **Graph Generation**: Create regular graphs with specified parameters
- **Data Processing**: Convert NetworkX graphs to DGL/PyTorch format for neural networks
- **GCN Training**: Train neural networks on 3-way max-cut problems with terminal constraints
- **Algorithm Comparison**: Compare GCN performance against randomized algorithms and CPLEX
- **Visualization**: Generate publication-quality charts and analysis reports

## Quick Start

### Prerequisites

```bash
pip install torch dgl networkx numpy matplotlib seaborn jupyter
```

### 1. Navigate to Python Directory

```bash
cd python/
```

### 2. Train a Model (Complete Pipeline)

Use the complete training pipeline notebook:

```bash
jupyter notebook complete_training_pipeline.ipynb
```

This will:
- Generate 20 graphs with 500 nodes each
- Process them with GraphExtender
- Train a GCN model
- Save the trained model as `final_maxcut_3way_n500_d6_8_t300_model.pth`

### 3. Test the Model

```bash
jupyter notebook neural_network_testing.ipynb
```

This will:
- Load the trained model
- Generate test graphs
- Compare simple GCN vs post-processed GCN performance
- Create performance analysis reports

### 4. Create Visualizations

```bash
jupyter notebook neural_network_visualization.ipynb
```

This generates publication-quality charts comparing different algorithms.

## Repository Structure

```
GCN-max-cut/
├── python/                          # Main code directory
│   ├── DataGenerator/               # Graph creation and processing
│   │   ├── GraphCreator.py         # Generate regular graphs
│   │   └── graphExtender.py        # Convert to neural network format
│   ├── Training/                    # Neural network training
│   │   ├── TrainingNeural.py       # Core training functions
│   │   └── TrainingNeural_load.py  # Model loading utilities
│   ├── Testing/                     # Model evaluation
│   │   └── TestingNeuralNetwork.py # Testing framework
│   ├── RandomAlgorithm/            # Baseline algorithms
│   │   └── RandomizedMaxCut.py     # Randomized max-cut algorithm
│   ├── CPLEX/                      # CPLEX optimization
│   │   ├── CplexCode.py            # CPLEX solver interface
│   │   └── CplexCode.ipynb         # CPLEX examples
│   ├── commons.py                   # Utility functions
│   ├── utils.py                     # Additional utilities
│   ├── complete_training_pipeline.ipynb  # Complete training workflow
│   ├── neural_network_testing.ipynb      # Model testing
│   └── neural_network_visualization.ipynb # Visualization
└── testData/                        # Pre-generated test graphs
    ├── testDataTxt/                 # Text format graphs
    └── testData_8/                  # Additional test set
```

## Step-by-Step Usage Guide

### Step 1: Graph Generation

Generate training graphs using GraphCreator:

```python
from DataGenerator.GraphCreator import generate_graph, generate_unique_terminals

# Generate a single graph
graph = generate_graph(
    n=500,              # Number of nodes
    d=7,                # Average degree
    graph_type='reg',   # Regular graph
    random_seed=42,     # For reproducibility
    edge_weight=1,      # Edge weights
    edge_capacity=1     # Edge capacities
)

# Generate terminal nodes for 3-way max-cut
terminals = generate_unique_terminals(500, 3)
```

### Step 2: Data Processing

Convert graphs for neural network training:

```python
from DataGenerator.graphExtender import process_graphs_from_folder

# Process graphs for training
processed_dataset = process_graphs_from_folder(
    all_graphs=your_graphs_dict,
    all_terminals=your_terminals_dict,
    max_nodes=1000,     # Extended matrix size
    save_batch_size=100,
    output_filename_prefix="training_data"
)
```

### Step 3: Model Training

Train the GCN model:

```python
from Training.TrainingNeural import train_from_pickle, TrainingConfig

# Configure training
config = TrainingConfig(
    n_nodes=1000,
    dim_embedding=1000,
    hidden_dim=500,
    number_classes=3,    # 3-way max-cut
    learning_rate=0.001,
    number_epochs=1000,
    tolerance=1e-4,
    patience=20
)

# Train model
model, best_loss, final_epoch, embeddings, loss_history = train_from_pickle(
    dataset_filename="path/to/processed_data.pkl",
    model_name="maxcut_3way_model",
    **config.__dict__
)
```

### Step 4: Model Loading

Load a trained model:

```python
from Training.TrainingNeural import load_neural_model

# Load model
model, inputs, config = load_neural_model(
    "final_maxcut_3way_n500_d6_8_t300_model.pth",
    config
)
```

### Step 5: Model Testing

Test the model performance:

```python
from Testing.TestingNeuralNetwork import test_multiple_graphs, analyze_results

# Test on multiple graphs
test_results, results_by_size = test_multiple_graphs(
    model=model,
    processed_graphs=test_graphs,
    graph_sizes=[50, 100, 200, 300, 500],
    post_processing_iterations=200
)

# Analyze results
analysis = analyze_results(test_results, results_by_size, [50, 100, 200, 300, 500])
```

### Step 6: Algorithm Comparison

Compare against baseline algorithms:

```python
from RandomAlgorithm.RandomizedMaxCut import randomized_k_way_maxcut

# Run randomized algorithm
cut_value, partition = randomized_k_way_maxcut(
    graph,
    k=3,
    max_iterations=1000,
    fixed_terminals={0: 0, 1: 1, 2: 2}
)
```

### Step 7: Visualization

Create publication-quality visualizations:

```python
from Testing.TestingNeuralNetwork import create_visualizations

# Generate charts
create_visualizations(
    test_results=test_results,
    results_by_size=results_by_size,
    graph_sizes=[50, 100, 200, 300, 500],
    save_path="analysis_plots.png"
)
```

## Advanced Features

### Post-Processing Optimization

The framework includes probabilistic post-processing:

```python
from Testing.TestingNeuralNetwork import post_processing_optimization

# Improve GCN results with random sampling
best_partition, best_cut = post_processing_optimization(
    node_probabilities=model_output,
    graph=networkx_graph,
    iterations=200
)
```

### CPLEX Integration

Compare against optimal solutions:

```python
from CPLEX.CplexCode import solve_max_cut_cplex

# Solve with CPLEX (for small graphs)
optimal_cut, optimal_partition = solve_max_cut_cplex(graph, terminals)
```

### Batch Processing

Process large datasets efficiently:

```python
# The system automatically handles batching for large datasets
# Adjust batch_size in process_graphs_from_folder for memory management
processed_data = process_graphs_from_folder(
    all_graphs=large_graph_dataset,
    all_terminals=large_terminal_dataset,
    max_nodes=1000,
    save_batch_size=50  # Smaller batches for large datasets
)
```

## Configuration Files

### Training Configuration

Key parameters in `TrainingConfig`:

- `n_nodes`: Maximum number of nodes (matrix size)
- `dim_embedding`: Embedding dimension (typically equals n_nodes)
- `hidden_dim`: Hidden layer size (typically n_nodes/2)
- `number_classes`: Number of partitions (3 for 3-way max-cut)
- `learning_rate`: Optimizer learning rate
- `number_epochs`: Maximum training epochs
- `patience`: Early stopping patience
- `penalty`: Terminal constraint penalty

### Graph Generation

Key parameters for graph generation:

- `n`: Number of nodes
- `d`: Average degree
- `graph_type`: 'reg' for regular graphs
- `random_seed`: For reproducibility
- `edge_weight`: Edge weight (typically 1)

## Output Files

The pipeline generates several output files:

### Training Data
- `*_graphs.pkl`: Raw NetworkX graphs
- `*_terminals.pkl`: Terminal node assignments
- `*_training_ready.pkl`: Processed DGL/PyTorch format
- `*_complete_results.pkl`: Full training results

### Models
- `final_*.pth`: Trained neural network model
- `epoch_*_loss_*.pth`: Checkpoint models during training

### Results
- `*_test_results.pkl`: Testing performance results
- `*_analysis.png`: Visualization charts
- `*_summary_report.txt`: Performance summary
- `*_SUMMARY.md`: Complete pipeline documentation

## Troubleshooting

### Common Issues

1. **PyTorch Model Loading Error**
   ```
   TypeError: Weights only load failed
   ```
   **Solution**: The code handles PyTorch 2.6+ compatibility automatically with `weights_only=False`.

2. **Log Scale Plotting Error**
   ```
   ValueError: Data has no positive values, and therefore cannot be log-scaled
   ```
   **Solution**: The code automatically takes absolute values for negative loss values.

3. **Memory Issues with Large Graphs**
   - Reduce `save_batch_size` in `process_graphs_from_folder`
   - Use smaller `max_nodes` value
   - Process datasets in smaller chunks

4. **CUDA/GPU Issues**
   - The code automatically detects and uses available hardware
   - CPU training is fully supported for smaller datasets

### Performance Tips

1. **For Large Datasets**:
   - Use `save_batch_size=50` or smaller
   - Consider using GPU if available
   - Monitor memory usage during processing

2. **For Better Model Performance**:
   - Increase `post_processing_iterations` for better results
   - Tune `penalty` parameter for terminal constraints
   - Experiment with different `hidden_dim` sizes

3. **For Faster Training**:
   - Use GPU if available
   - Reduce `max_nodes` if graphs are smaller
   - Use early stopping with appropriate `patience`

## File Organization

Keep your workspace organized:

```
your_project/
├── training_data/          # Generated training datasets
├── testing_data/           # Generated test datasets  
├── models/                 # Saved neural network models
├── results/                # Testing and analysis results
└── visualizations/         # Generated plots and charts
```

## Research Notes

### Algorithm Performance

Based on testing:
- **GCN vs Randomized**: Variable performance, GCN typically 5-7x faster
- **Post-processing**: Improves results in 100% of cases with 200 iterations
- **Scalability**: Performance decreases with larger graphs
- **Runtime**: Post-processing adds ~300-400x overhead but better quality

### Best Practices

1. **Graph Sizes**: Test on 50-500 node graphs for optimal performance
2. **Terminal Constraints**: Always verify terminal assignments are satisfied
3. **Model Validation**: Use separate test sets for unbiased evaluation
4. **Hyperparameters**: Start with provided defaults, then tune for your use case

### Future Improvements

Areas for enhancement:
- Implement attention mechanisms in GCN architecture
- Add support for weighted graphs
- Optimize post-processing algorithms
- Extend to k-way max-cut for k > 3

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gcn-maxcut-2024,
  title={Graph Convolutional Networks for 3-Way Max-Cut Problems},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```

## License

[Add your license information here]

## Contact

[Add your contact information here]