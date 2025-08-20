
# GCN Max-Cut Training Pipeline - Complete Summary

## Experiment Overview
- **Date**: 2025-08-19 20:46:23
- **Dataset**: maxcut_3way_n500_d6_8_t300
- **Objective**: 3-way max-cut problem with GCN

## Dataset Generation
- **Graphs generated**: 20
- **Nodes per graph**: 500
- **Target degree range**: 6-8
- **Actual average degree**: 7.10
- **Generation time**: 0.18 seconds

## Data Processing
- **Extended matrix size**: 1000 x 1000
- **Processing time**: 12.86 seconds
- **Terminal normalization**: All graphs normalized to terminals [0, 1, 2]

## Model Training
- **Architecture**: GCN with 1000 -> 500 -> 3
- **Training time**: 171.81 seconds (2.9 minutes)
- **Epochs completed**: 486/1000
- **Best loss**: -31355.000000
- **Convergence**: Early stopping

## File Artifacts
- **Raw graphs**: ./training_data/maxcut_3way_n500_d6_8_t300_graphs.pkl
- **Raw terminals**: ./training_data/maxcut_3way_n500_d6_8_t300_terminals.pkl
- **Training dataset**: ./training_data/maxcut_3way_n500_d6_8_t300_training_ready.pkl
- **Trained model**: ./final_maxcut_3way_n500_d6_8_t300_model.pth
- **Complete results**: ./training_data/maxcut_3way_n500_d6_8_t300_complete_results.pkl

## Usage Instructions

### Load the trained model:
```python
from Training.TrainingNeural import load_neural_model, TrainingConfig
config = TrainingConfig(n_nodes=1000, number_classes=3)
model, inputs, config = load_neural_model('./final_maxcut_3way_n500_d6_8_t300_model.pth', config)
```

### Load the training dataset:
```python
from commons import open_file
dataset = open_file('./training_data/maxcut_3way_n500_d6_8_t300_training_ready.pkl')
```

### Load complete results:
```python
results = open_file('./training_data/maxcut_3way_n500_d6_8_t300_complete_results.pkl')
```

## Total Pipeline Time
- **Graph generation**: 0.2s
- **Data processing**: 12.9s  
- **Model training**: 171.8s
- **Total time**: 184.9s (3.1 minutes)
