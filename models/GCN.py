    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:01:33 2022

@author: kaouther
"""

import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

# We first define the message and reduce function
gcn_msg = fn.copy_u('h', 'm')# Copy the node feature 'h' to message 'm'
gcn_reduce = fn.sum(msg='m', out='h')# Sum the incoming messages into node feature 'h'



class GCNLayer(nn.Module):
    """A single layer of the Graph Convolutional Network (GCN)."""
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.ln = nn.LayerNorm(in_feats)
        
    def forward(self, g, feature):
        """Forward pass for GCN layer.
        
        Args:
            g: The graph input.
            feature: The input features for the nodes.

        Returns:
            Transformed node features after applying GCN operations.
        """
        with g.local_scope():
            g.ndata["h"] = feature # Store input features in graph's node data
            g.update_all(gcn_msg, gcn_reduce)  # Perform message passing and reduction
            h = g.ndata["h"]  # Retrieve updated node features
            h = self.ln(h)  # Apply Layer Normalization
            h = self.linear(h) # Apply linear transformation
           
            return h
        
class GCNModule(nn.Module):

    """A multi-layer Graph Convolutional Network module."""
    def __init__(self,in_feats,hidden_dim,out_feats):
        super(GCNModule, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_dim) # First GCN layer
        self.layer2 = GCNLayer(hidden_dim, hidden_dim) # Second GCN layer
        self.layer3 = GCNLayer(hidden_dim, out_feats) # Output GCN layer

    def forward(self, g, features):
        """Forward pass for the GCN module.
        
        Args:
            g: The graph input.
            features: The input features for the nodes.

        Returns:
            The final output features after passing through all layers.
        """
        x = F.relu(self.layer1(g, features)) # Apply first layer and ReLU activation
        x = F.relu(self.layer2(g, x)) # Apply second layer and ReLU activation
        x = self.layer3(g, x) # Apply third layer without activation (for final output)
        return x







