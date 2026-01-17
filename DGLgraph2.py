#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:29:35 2022

@author: kaouther
"""

"""
Similar to DGLgraph, but the features used here exclude scalability metrics.
"""


import dgl 
import torch

class Graph():
    
    def __init__(self,network): 
        
        self.graph=self.generatGraph(network,network.getNetworkx())
    
    def getFeatures(self):
        features=self.graph.ndata['features'] 
        return features
    
    def getGraph(self):
        return self.graph
    
    def generatGraph(self,network,networkx):
        graph=dgl.from_networkx(networkx)
        features=network.getFeatures2()
        # If features is already a torch tensor, move graph to same device and assign
        if isinstance(features, torch.Tensor):
            device = features.device
            graph = graph.to(device)
            graph.ndata['features'] = features
        else:
            graph.ndata['features'] = torch.tensor(features, dtype=torch.float32)
        return graph
    
    def updateFeatures(self,features):
        if isinstance(features, torch.Tensor):
            device = features.device
            self.graph = self.graph.to(device)
            self.graph.ndata['features'] = features
        else:
            self.graph.ndata['features'] = torch.tensor(features, dtype=torch.float32)
        
class SnGraph(Graph):
    def __init__(self,network,vnf_cpu):
        self.vnf_cpu =vnf_cpu
        super().__init__(network)

    def generatGraph(self,network,networkx):
        graph=dgl.from_networkx(networkx)
        features=network.getFeatures2(self.vnf_cpu)
        # If features is already a torch tensor, move graph to same device and assign
        if isinstance(features, torch.Tensor):
            device = features.device
            graph = graph.to(device)
            graph.ndata['features'] = features
        else:
            graph.ndata['features'] = torch.tensor(features, dtype=torch.float32)
        return graph