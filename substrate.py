#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:42:55 2022

@author: kaouther
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from node import Snode
from edege import Sedege

# Cache device globally to avoid repeated torch.device() calls
_DEVICE_CACHE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SN :
    
    def __init__(self, num_nodes, cpu_range, bw_range,lt_range,topology):

        self.num_nodes = num_nodes
        """  Total number of nodes in the substrate network"""
        self.snode = []
        """ List of Snodes, containing objects of type Snode, representing the actual nodes of the SN """
        self.sedege = []
        """ List of Edges, containing objects of type Sedege, representing the actual Edges of the SN """
        self.g = topology 
        """ The SN topology"""
        self.edges = list(self.g.edges())
        """ List of edges in the SN, containing source and destination nodes, used for managing topology """
        self.numedges = len(self.edges)
        """ Total number of Edges in the SN"""

        # Snodes Creation 
        #-------------------------------------------------------------#
        for i in range(self.num_nodes):
            cpu = np.random.randint(cpu_range[0],cpu_range[1])
            self.snode.append(Snode(i,cpu))
        #-------------------------------------------------------------#

        # Sedges Creation
        #-------------------------------------------------------------#
        for i in range(self.numedges):
            bw = np.random.randint(bw_range[0],bw_range[1])
            lt = np.random.randint(lt_range[0], lt_range[1])
            a_t_b = [self.edges[i][0], self.edges[i][1]]
            self.sedege.append(Sedege(i, bw,lt, a_t_b))
            self.snode[a_t_b[0]].links.append(i)
            self.snode[a_t_b[1]].links.append(i)
        #-------------------------------------------------------------#

        # Calculating the total bandwidth of connected edges for each Snode.
        #----------------------------------------------------------------#
        for n in self.snode:
            for e in self.sedege:
                if n.index in e.nodeindex:
                    n.bw += e.bandwidth
            n.lastbw = n.bw
        #----------------------------------------------------------------# 

        # Assigning neighbors to each Snode and Snode degree
        #----------------------------------------------------------------#
        for el in self.snode:
            el.neighbors = [n for n in self.g.neighbors(el.index)]
        for el in self.snode:
            el.degree = len(el.neighbors)
        #----------------------------------------------------------------#



    def updateCpu(self,nodemapping,req_cpu):
        """ 
        Updates the CPU of substrate nodes after a vertical scaling up operation.
        
        For each node in the `nodemapping` list, the corresponding substrate node's CPU 
        is updated with the required CPU specified in the `req_cpu` list.
        
        Args:
            nodemapping (list): A list of indices representing the substrate nodes to be updated.
            req_cpu (list): A list of CPU requirements to be applied to the corresponding substrate nodes.
        """
        for i in range(len(nodemapping)):
            self.snode[nodemapping[i]].updateCpu(req_cpu[i])


    def remCpu(self,nodemapping):
        """ 
        Calculates the remaining CPU for each substrate node in the nodemapping list.
        
        This method iterates over the `nodemapping` list and appends the remaining CPU 
        (stored in `lastcpu`) of each corresponding substrate node to a list.
        
        Args:
            nodemapping (list): A list of indices representing the substrate nodes.

        Returns:
            list: A list containing the remaining CPU for each node in the nodemapping list.
        """
        remCpu=[]
        for i in nodemapping:
            remCpu.append(self.snode[i].lastcpu)  
        return remCpu

    def getCpu(self):
        """ 
        Calculates the remaining CPU for each substrate node in the SN.
        
        This method iterates over the snode list and appends the remaining CPU 
        (stored in `lastcpu`) of each corresponding substrate node to a list.
        
        Returns:
            list: A list containing the remaining CPU for each node in the SN.
        """
        cpu = []
        for node in self.snode:
            cpu.append(node.lastcpu)
        return cpu
    
    def copy_for_placement(self):
        """
        Fast copy of SN for VNR placement. Only copies state, not topology.
        This is much faster than deepcopy for repeated placements.
        
        Returns:
            SN: A shallow copy with deep-copied mutable state only
        """
        from edege import Sedege
        
        sn_copy = object.__new__(SN)
        sn_copy.num_nodes = self.num_nodes
        sn_copy.numedges = self.numedges
        sn_copy.g = self.g  # Share topology graph
        sn_copy.edges = self.edges  # Share edge list
        
        # Only deep copy the nodes (which have mutable state)
        sn_copy.snode = [Snode(node.index, node.cpu) for node in self.snode]
        for i, node_copy in enumerate(sn_copy.snode):
            node_copy.lastcpu = self.snode[i].lastcpu
            node_copy.vnodeindexs = self.snode[i].vnodeindexs.copy()
            node_copy.links = self.snode[i].links
            node_copy.neighbors = self.snode[i].neighbors
            node_copy.bw = self.snode[i].bw
            node_copy.lastbw = self.snode[i].lastbw
            node_copy.p_load = self.snode[i].p_load
            node_copy.degree = self.snode[i].degree
        
        # Copy edges with mutable state (vedegeindexs can be modified during placement)
        sn_copy.sedege = []
        for edge in self.sedege:
            edge_copy = Sedege(edge.index, edge.bandwidth, edge.latency, edge.nodeindex)
            edge_copy.lastbandwidth = edge.lastbandwidth
            edge_copy.vedegeindexs = edge.vedegeindexs.copy()
            edge_copy.open = edge.open
            edge_copy.mappable_flag = edge.mappable_flag
            sn_copy.sedege.append(edge_copy)
        
        return sn_copy
        
    def removenodemapping(self, vnr):
        """ 
        Removes the node mapping of a VNR from the substrate network.

        This method updates the substrate nodes by removing the VNFs that were 
        previously mapped to them. It performs the following steps:
        
        1. Iterates over all substrate nodes (`self.snode`).
        2. For each substrate node, it checks if any VNF from the given VNR is mapped to that node.
        3. If a match is found, the substrate node's remaining CPU (`lastcpu`) is incremented by 
        the CPU of the corresponding VNF.
        4. The matching VNF is removed from the `vnodeindexs` list of the substrate node.
        5. Finally, it clears the host mapping (`sn_host`) for each VNF in the VNR.
        
        Args:
            vnr: The VNR whose node mappings should be removed.
        """
        sn = len(self.snode)
        vn = len(vnr.vnode)
        for i in range(sn):
            xtemp = []
            for x in self.snode[i].vnodeindexs:
                if vnr.id == x[0]:
                    self.snode[i].lastcpu = self.snode[i].lastcpu + vnr.vnode[x[1]].cpu
                    xtemp.append(x)
            for x in xtemp:
                self.snode[i].vnodeindexs.remove(x)
        for j in range(vn):
            vnr.vnode[j].sn_host = None
            
    def removeedegemapping(self, vnr):
        """ 
        Removes the edge mapping of a VNR from the substrate network.
        
        This method updates the substrate edges by removing the virtual edges (vedge) that were
        previously mapped to them. It performs the following steps:
        
        1. Iterates over all substrate edges (`self.sedege`).
        2. For each substrate edge, checks if any virtual edge from the given VNR is mapped to it.
        3. If a match is found, the substrate edge's remaining bandwidth (`lastbandwidth`) is 
        incremented by the bandwidth of the corresponding virtual edge.
        4. The matching virtual edge is removed from the `vedegeindexs` list of the substrate edge.
        5. If a substrate edge no longer has any mapped virtual edges, it is marked as closed (`open = False`).
        6. Resets the bandwidth usage (`lastbw`) of each substrate node by summing the bandwidth 
        of its connected edges.
        7. Clears the shortest path connections (`spc`) for each virtual edge in the VNR.
        
        Args:
            vnr: The VNR whose edge mappings should be removed.
        """
        en = len(self.sedege)
        vn = len(vnr.vedege)
        for e in range(en):
            tempx = []
            for x in self.sedege[e].vedegeindexs:
                if vnr.id == x[0]:
                    self.sedege[e].lastbandwidth = self.sedege[e].lastbandwidth + vnr.vedege[x[1]].bandwidth
                    tempx.append(x)
            for x in tempx:
                self.sedege[e].vedegeindexs.remove(x)
            if not self.sedege[e].vedegeindexs:
                self.sedege[e].open = False
        for n in self.snode:
            n.lastbw = 0
            for e in self.sedege:
                if n.index in e.nodeindex:
                    n.lastbw += e.lastbandwidth
        for ve in range(vn):
            vnr.vedege[ve].spc = []
    
    
    def Sn2_networkxG(self, bandlimit=0):
        """Converts the substrate network to a NetworkX graph representation."""
        g = nx.Graph()
        for snod in self.snode:
            g.add_node(snod.index, index=snod.index, cpu=snod.cpu, lastcpu=snod.lastcpu)
        en = len(self.sedege)
        for i in range(en):
            if self.sedege[i].lastbandwidth > bandlimit:
                g.add_edge(self.sedege[i].nodeindex[0], self.sedege[i].nodeindex[1], index=self.sedege[i].index,lastbandwidth=self.sedege[i].lastbandwidth, bandwidth=self.sedege[i].bandwidth,capacity=1)
        return g
    
    def drawSN(self,edege_label=False,classflag=False):
        """Draws the substrate network using NetworkX with optional edge labels and color-coding."""
        plt.figure()
        g = self.Sn2_networkxG()
        pos = nx.fruchterman_reingold_layout(g)
        if classflag:
            color = {}
            colomap = []
            for i in range(len(self.snode)):
                if self.snode[i].classflag not in color.keys():
                    color[self.snode[i].classflag] = self.randRGB()
                colomap.append(color[self.snode[i].classflag])
            nx.draw(g, node_color=colomap, font_size=8, node_size=300, pos=pos, with_labels=True,
                    nodelist=g.nodes())
            if  edege_label:nx.draw_networkx_edge_labels(g, pos, edge_labels={edege: g[edege[0]][edege[1]]["lastbandwidth"] for edege in g.edges()})
        else:
            nx.draw(g, node_color=[[0.5, 0.8, 0.8]], font_size=8, node_size=300, pos=pos, with_labels=True,nodelist=g.nodes())
            if  edege_label:nx.draw_networkx_edge_labels(g, pos, edge_labels={edege: g[edege[0]][edege[1]]["lastbandwidth"] for edege in g.edges()})
        plt.show()
        
    def randRGB(self):
        return (np.random.randint(0, 255) / 255.0,
                np.random.randint(0, 255) / 255.0,
                np.random.randint(0, 255) / 255.0)

    def getNetworkx(self):
        """ 
        Returns the NetworkX graph representation of the SN.
        This method provides access to the internal graph structure.
        """
        return self.g

    def msg(self):
        """calls the msg method for each Snode and Sedge to print their information."""
        n = len(self.snode)
        for i in range(n):
            print('--------------------')
            self.snode[i].msg()
        n = len(self.sedege)
        for i in range(n):
            print('--------------------')
            self.sedege[i].msg()
            
            
    
    '''
    def getFeatures(self,vnf_cpu):
        """
        Extract features for the substrate network
        
        This function generates various features from the substrate nodes (snode) such as CPU, bandwidth, 
        degree, and p_load, scaled and formatted for input into a machine learning model. It also 
        computes a feasible flag for each node, indicating whether it can host the given VNF based 
        on available CPU.
        
        Args:
            vnf_cpu (int): CPU requirement of the VNF being considered for mapping.
            
        Returns:
            features (np.ndarray): A matrix of substrate network features, where each row corresponds 
                                to a node and its associated features. 
                                The features are concatenated, transposed, and returned as a NumPy array for further processing.
        """
        
        cpu = [el.lastcpu for el in self.snode]
        cmax = np.max(cpu)
        scaled_cpu = cpu  / cmax
        cpu = torch.from_numpy(np.squeeze(scaled_cpu))
        cpu = torch.unsqueeze(cpu, dim=0).numpy()
        
        bw = [el.lastbw  for el in self.snode]
        bmax = np.max(bw)
        scaled_bw = bw / bmax
        bw = torch.from_numpy(np.squeeze(scaled_bw))
        bw = torch.unsqueeze(bw, dim=0).numpy()
        
        bw_av = [el.bw / el.degree for el in self.snode]
        maxb = np.max(bw_av)
        scaled_bw_av = bw_av/maxb
        bw_av = torch.from_numpy(np.squeeze(scaled_bw_av))
        bw_av = torch.unsqueeze(bw_av, dim=0).numpy()

        bw_max = [el.max_bw(self.sedege) for el in self.snode]
        scaled_max = bw_max/np.max(bw_max)
        bw_max = torch.from_numpy(np.squeeze(scaled_max))
        bw_max = torch.unsqueeze(bw_max, dim=0).numpy()

        bw_min = [el.min_bw(self.sedege) for el in self.snode]
        scaled_min = bw_min/np.max(bw_min)
        bw_min = torch.from_numpy(np.squeeze(scaled_min))
        bw_min = torch.unsqueeze(bw_min, dim=0).numpy()

        degree = [el.degree for el in self.snode]
        scaled_degree = degree/np.max(degree)
        degree = torch.from_numpy(np.squeeze(scaled_degree))
        degree = torch.unsqueeze(degree, dim=0).numpy()

        feasible_flag = [1 if el.lastcpu > vnf_cpu else 0 for el in self.snode]
        feasible_flag = torch.from_numpy(np.squeeze(feasible_flag))
        feasible_flag = torch.unsqueeze(feasible_flag, dim=0).numpy()

        p_load = torch.from_numpy(np.squeeze([el.p_load for el in self.snode]))
        p_load = torch.unsqueeze(p_load, dim=0).numpy()

        features = np.transpose(np.concatenate((feasible_flag ,cpu,bw, bw_av, bw_max, bw_min, degree,p_load)))
        return features
    '''

    def getFeatures(self, vnf_cpu):
        snode = self.snode
        # Vecteurs de base
        cpu = np.array([n.lastcpu for n in snode], dtype=np.float32)
        bw = np.array([n.lastbw for n in snode], dtype=np.float32)
        bw_av = np.array([n.bw / n.degree for n in snode], dtype=np.float32)
        bw_max = np.array([n.max_bw(self.sedege) for n in snode], dtype=np.float32)
        bw_min = np.array([n.min_bw(self.sedege) for n in snode], dtype=np.float32)
        degree = np.array([n.degree for n in snode], dtype=np.float32)
        p_load = np.array([n.p_load for n in snode], dtype=np.float32)

        # Petites gardes pour éviter div/0
        def safe_scale(x):
            m = np.max(x)
            return x / m if m > 0 else x

        scaled_cpu = safe_scale(cpu)
        scaled_bw = safe_scale(bw)
        scaled_bw_av = safe_scale(bw_av)
        scaled_bw_max = safe_scale(bw_max)
        scaled_bw_min = safe_scale(bw_min)
        scaled_degree = safe_scale(degree)

        feasible_flag = (cpu > vnf_cpu).astype(np.float32)

        features = np.column_stack(
            [
                feasible_flag,
                scaled_cpu,
                scaled_bw,
                scaled_bw_av,
                scaled_bw_max,
                scaled_bw_min,
                scaled_degree,
                p_load,  # p_load n’est pas normalisé dans la version originale ; on le laisse brut pour conserver la sémantique
            ]
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.from_numpy(features).to(_DEVICE_CACHE)
        
    def getFeatures2(self,vnf_cpu):
        """
        Extract features for the substrate network without  (p_load) information.
        
        Similar to `getFeatures`, this function extracts and scales various substrate node features,
        but excludes p_load (scalability metrics). It returns the feature matrix excluding p_load.
        
        Args:
            vnf_cpu (int): CPU requirement of the VNF being considered for mapping.
            
        Returns:
            features (np.ndarray): A matrix of substrate network features, where each row corresponds 
                                to a node and its associated features.
        """
        snode = self.snode

        cpu = np.array([n.lastcpu for n in snode], dtype=np.float32)
        bw = np.array([n.lastbw for n in snode], dtype=np.float32)
        bw_av = np.array([n.bw / n.degree for n in snode], dtype=np.float32)
        bw_max = np.array([n.max_bw(self.sedege) for n in snode], dtype=np.float32)
        bw_min = np.array([n.min_bw(self.sedege) for n in snode], dtype=np.float32)
        degree = np.array([n.degree for n in snode], dtype=np.float32)

        def safe_scale(x):
            m = np.max(x)
            return x / m if m > 0 else x

        scaled_cpu = safe_scale(cpu)
        scaled_bw = safe_scale(bw)
        scaled_bw_av = safe_scale(bw_av)
        scaled_bw_max = safe_scale(bw_max)
        scaled_bw_min = safe_scale(bw_min)
        scaled_degree = safe_scale(degree)

        feasible_flag = (cpu > vnf_cpu).astype(np.float32)

        features = np.column_stack(
            [
                feasible_flag,
                scaled_cpu,
                scaled_bw,
                scaled_bw_av,
                scaled_bw_max,
                scaled_bw_min,
                scaled_degree,
            ]
        )
        

        return torch.from_numpy(features).to(_DEVICE_CACHE)
    

    def get_used_ressources(self):
        """
        Calculate the used resources in the substrate network.
        
        This function iterates through all substrate nodes and edges to determine 
        the amount of resources used, counting how many nodes and links are in use.
        
        Returns:
            dict: A dictionary containing the used CPU, number of used nodes, 
                used bandwidth, and number of used links.
        """
        used_cpu=0
        cpu =0
        used_bw = 0
        bw = 0
        used_nodes = 0
        used_links = 0
        for node in self.snode:
            used_cpu +=node.cpu- node.lastcpu
            #cpu+= node.cpu
            if node.cpu > node.lastcpu :
                used_nodes+=1
        for edge in self.sedege:
            used_bw+=edge.bandwidth-edge.lastbandwidth
            #bw += edge.bandwidth
            if edge.bandwidth > edge.lastbandwidth:
                used_links+=1
        
        return {"used_cpu":used_cpu,"used_nodes":used_nodes,"used_bw":used_bw,"used_links":used_links}
            
    def sn_state(self):
        return {"snodes_state" : [el.__str__() for el in self.snode] ,
                "sedges_state" : [el.__str__() for el in self.sedege]}
        
 
        


    
