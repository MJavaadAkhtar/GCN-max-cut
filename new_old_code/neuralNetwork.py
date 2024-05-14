import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class GCNSoftmax(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout, device):
        super(GCNSoftmax, self).__init__()
        self.dropout_frac = dropout
        self.conv1 = GraphConv(in_feats, hidden_size).to(device)
        self.conv2 = GraphConv(hidden_size, num_classes).to(device)

    def forward(self, g, inputs, terminals=None):
        # Basic forward pass
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_frac, training=self.training)
        h = self.conv2(g, h)
        h = F.softmax(h, dim=1)  # Apply softmax over the classes dimension

        # Handling terminals if provided
        if terminals is not None:
            # Clone h to avoid in-place operations
            h_clone = h.clone()
            # Ensure terminals is a dictionary with node index as keys and target class as values
            for node_index, class_index in terminals.items():
                # Create a one-hot encoded vector for the terminal node's class
                terminal_vector = torch.zeros(h.shape[1], device=h.device)
                terminal_vector[class_index] = 1
                # Use indexing that does not modify the original tensor in-place
                h_clone[node_index] = terminal_vector
            h = h_clone

        return h