#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:29:35 2022

@author: kaouther
"""
import dgl 
import torch

class Graph():
    
    def __init__(self,network): 
        """ A dgl graph that represents the network's topology and associated features """
        self.graph=self.generatGraph(network,network.getNetworkx())
    
    def getFeatures(self):
        """ Retrieves node Features from the DGL graph"""
        features=self.graph.ndata['features'] 
        return features
    
    def getGraph(self):
        """ Return the DGL Graph"""
        return self.graph
    
    def generatGraph(self,network,networkx):
        """
        Generates a DGL graph from a NetworkX graph and assigns node features.
        
        Args:
            network: The network object containing network information, here it is the VNR 
            networkx: A NetworkX graph representing the network topology.
        
        Returns:
            graph: A DGL graph with node features added from the network object.
        """
        graph=dgl.from_networkx(networkx)
        
        
        features=network.getFeatures()
        # If features is already a torch tensor, move graph to same device and assign
        if isinstance(features, torch.Tensor):
            device = features.device
            graph = graph.to(device)
            graph.ndata['features'] = features
        else:
            graph.ndata['features'] = torch.tensor(features, dtype=torch.float32)
        return graph
    
    def updateFeatures(self,features):
        """
        Updates the node features of the DGL graph.
        """
        if isinstance(features, torch.Tensor):
            device = features.device
            self.graph = self.graph.to(device)
            self.graph.ndata['features'] = features
        else:
            self.graph.ndata['features'] = torch.tensor(features, dtype=torch.float32)
        
class SnGraph(Graph):
    def __init__(self,network,vnf_cpu):
        self.vnf_cpu =vnf_cpu
        """ 
        The CPU requirement for the VNF placement, used to filter out substrate nodes (Snodes) that lack sufficient CPU resources.
        """
        super().__init__(network)

    def generatGraph(self,network,networkx):
        """
        Generates a DGL graph from a NetworkX graph and assigns node features.
        
        Args:
            network: The network object containing network information, here it is the SN 
            networkx: A NetworkX graph representing the network topology.
        
        Returns:
            graph: A DGL graph with node features added from the network object,
            Each node is assigned a flag based on its lastCPU value to indicate its suitability for VNF placement according to vnf_CPU.
        """
        graph=dgl.from_networkx(networkx)
        features=network.getFeatures(self.vnf_cpu)
        # If features is already a torch tensor, move graph to same device and assign
        if isinstance(features, torch.Tensor):
            device = features.device
            graph = graph.to(device)
            graph.ndata['features'] = features
        else:
            graph.ndata['features'] = torch.tensor(features, dtype=torch.float32)
        return graph