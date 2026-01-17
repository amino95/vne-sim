#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:43:05 2022
@author: kaouther
"""
from node import Vnf
from edege import Vedege
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import torch

# Cache device globally to avoid repeated torch.device() calls
_DEVICE_CACHE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_connected_er_graph(num_nodes, probability):
    """Generate a VNR topology using the Erdős-Rényi model"""
    while True:
        er_graph = nx.erdos_renyi_graph(num_nodes, probability, directed=False)
        if nx.is_connected(er_graph):
            return er_graph
class VNR:
    ID_ = 0
    """ A class variable used to assign a unique identifier to each VNR instance.  """
    def __init__(self,vnf_range, cpu_range, bw_range,lt_range,flavor_size,duration,mtbs):
        VNR.ID_+=1
        self.mtbs=mtbs
        """ Mean time between scale demands that arrive during the VNF lifespan """
        self.id = VNR.ID_
        """ VNR ID used to facilitate VNR management """
        self.num_vnfs = np.random.randint(vnf_range[0], vnf_range[1])
        """ The total number of VNFs in the VNR """
        self.p = ( np.log(self.num_vnfs)) / self.num_vnfs
        """ See Erdős-Rényi model to understand its parameters  """
        self.graph =  generate_connected_er_graph(self.num_vnfs, self.p)
        """ Generate a VNR topology using the Erdős-Rényi model """
        self.num_vedges = len(self.graph.edges())
        """ The total number of Edges in the VNR """
        self.edges = list(self.graph.edges())
        """ List of edges in the VNR, containing source and destination nodes, used for managing topology """
        self.duration  = duration
        """ The lifespan of the VNR """
        self.vnode = []
        """ List of VNFs, containing objects of type VNF, representing the actual VNFs of the VNR """
        self.vedege = []
        """ List of Virtual Edges, containing objects of type Vedege, representing the actual Edges of the VNR """
        self.nodemapping = []
        """ VNFs placement in the Service Network (SN) """
        self.edgemapping = []
        """ Edges placement in the Service Network (SN) """
        p_scalingUp=int(1/2*duration/mtbs)+1
        """ Potential number of scaling up """
        # VNFs Creation
        #----------------------------------------------------------------#
        for i in range(self.num_vnfs):
            cpu=np.random.randint(cpu_range[0],cpu_range[1]//2)
            vno = Vnf(i,cpu,cpu_range[1],self.id,flavor_size,p_scalingUp)
            self.vnode.append(vno)
        #----------------------------------------------------------------#
        # Edges Creation
        #----------------------------------------------------------------#
        for i in range(self.num_vedges):
            a_t_b = list(self.graph.edges())[i]
            bw = np.random.randint(bw_range[0],bw_range[1])
            lt = np.random.randint(lt_range[0],lt_range[1])
            ved = Vedege(i,bw,lt,a_t_b)
            self.vedege.append(ved)
            self.vnode[a_t_b[0]].links.append(i)
            self.vnode[a_t_b[1]].links.append(i)
        #----------------------------------------------------------------#
        # Calculating the total bandwidth of connected edges for each VNF.
        #----------------------------------------------------------------#
        for n in self.vnode:
            for e in self.vedege:
                if n.index in e.nodeindex:
                    n.bw += e.bandwidth
        #----------------------------------------------------------------#
        # Assigning neighbors to each VNF and VNF degree
        #----------------------------------------------------------------#
        for el in self.vnode:
            el.neighbors = [n for n in self.graph.neighbors(el.index)]
        for el in self.vnode:
            el.degree = len(el.neighbors)
        #----------------------------------------------------------------#
    def getG(self):
        """ Return a topology of the VNR in NetworkX form """
        bandwidth = dict()
        for i in range (len(self.graph.edges())):
            ed = self.vedege[i]
            bandwidth[(ed.nodeindex[0],ed.nodeindex[1])] = ed.bandwidth
        nx.set_edge_attributes(self.graph, bandwidth,'bandwidth')
        return self.graph
    def drawVNR(self):
        """
        Draws the VNR topology using NetworkX, visualizing nodes and their connections.
        The graph layout is generated using the Fruchterman-Reingold algorithm.
        Node colors, sizes, and edge bandwidth labels are also specified for clarity.
        """
        plt.figure()
        g = self.getG()
        pos = nx.fruchterman_reingold_layout(g)
        nx.draw(g, node_color=[[0.5, 0.8, 0.5]], font_size=8, node_size=300, with_labels=True, nodelist=g.nodes())
        nx.draw_networkx_edge_labels(g, pos,edge_labels={edege: g[edege[0]][edege[1]]["bandwidth"] for edege in g.edges()})
        plt.show()
    def msg(self):
        """
        Displays the VNR ID and duration,
        and calls the msg method for each VNode and VEdge to print their information.
        """
        print("vnr_id",self.id,"duration",self.duration)
        for i in range(len(self.vnode)):
            self.vnode[i].msg()
        for i in range(len(self.vedege)):
            self.vedege[i].msg()
    def generate_scale_up(self):
        """
        Generates a scale-up configuration for VNFs.
        Selects a random subset of VNFs, increments their CPU index if possible (if the CPU demanded does not exceed the flavor maximum),
        and updates their CPU requirements accordingly.
        Returns a list of VNFs that demand a scale-up.
        In case our simulation contains multiple solvers for comparison,
        this method will generate a scaling up configuration for the first VNF.
        """
        rng = default_rng()
        random_chain= np.sort(rng.choice(self.num_vnfs, size=np.random.randint(1,self.num_vnfs+1), replace=False))
        scaling_chaine=[]
        for i in range(len(random_chain)):
            j=random_chain[i]
            if self.vnode[j].cpu_index< len(self.vnode[j].flavor)-1:
                scaling_chaine.append(j)
                self.vnode[j].cpu_index+=1
                self.vnode[j].req_cpu=self.vnode[j].flavor[self.vnode[j].cpu_index]-self.vnode[j].cpu
        return scaling_chaine
    def copy_scale_up(self,scaling_chaine):
        """
        In case our simulation contains multiple solvers for comparison,
        this function will be called to copy the scaling chain configuration
        to the corresponding VNFs.
        """
        for i in range(len(scaling_chaine)):
            j=scaling_chaine[i]
            if self.vnode[j].cpu_index< len(self.vnode[j].flavor)-1:
                self.vnode[j].cpu_index+=1
                self.vnode[j].req_cpu=self.vnode[j].flavor[self.vnode[j].cpu_index]-self.vnode[j].cpu
    def generate_scale_down(self):
        """
        Generates a scale-down configuration for VNFs.
        Selects a random subset of VNFs, decrements their CPU index if possible (if it is greater than 0),
        and updates their CPU requirements accordingly.
        Returns a list of VNFs that have been scaled down.
        In case our simulation contains multiple solvers for comparison,
        this method will generate a scaling down configuration for the first VNF.
        """
        rng = default_rng()
        random_chain= np.sort(rng.choice(self.num_vnfs, size=np.random.randint(1,self.num_vnfs+1), replace=False))
        scaling_chaine=[]
        for i in range(len(random_chain)):
            j=random_chain[i]
            if self.vnode[j].cpu_index> 0:
                scaling_chaine.append(j)
                self.vnode[j].cpu_index-=1
                self.vnode[j].req_cpu=self.vnode[j].cpu-self.vnode[j].flavor[self.vnode[j].cpu_index]
        return scaling_chaine
    def copy_scale_down(self,scaling_chaine):
        """
        In case our simulation contains multiple solvers for comparison,
        this method can be used to adjust the scaling configuration for each VNF accordingly.
        """
        for i in range(len(scaling_chaine)):
            j=scaling_chaine[i]
            if self.vnode[j].cpu_index> 0:
                self.vnode[j].cpu_index-=1
                self.vnode[j].req_cpu=self.vnode[j].cpu-self.vnode[j].flavor[self.vnode[j].cpu_index]
    def p_loadRset(self,sb):
        """Remove p_load of the VNFS of this VNR from Substrate network, To be utilized when ending a VNR"""
        for j in range(self.num_vnfs):
            i=self.nodemapping[j]
            if (i>-1):
                sb.snode[i].p_load=(sb.snode[i].p_load*sb.snode[i].cpu-self.vnode[j].p_maxCpu)/sb.snode[i].cpu
                if sb.snode[i].p_load<0:
                    sb.snode[i].p_load=0.0
    def EndsVnr(self,sb,VNRSS):
        """
        Updates substrate resources after freeing the allocated resources for a VNR.
        This method will be called when the VNR's lifespan ends or when a scalability failure occurs.
        It performs the following actions:
        1. Gets the index of the VNR in the VNRSS
        2. Removes the p_load of the VNFs of this VNR from the substrate network.
        3. Removes the VNF mapping and edge mapping from the substrate.
        4. Removes the VNR from the VNRSS by updating the corresponding lists and decrementing the number of requests.
        5. Deletes the current instance of the VNR.
        """
        VNRindex=VNRSS.reqs_ids.index(self.id)
        self.p_loadRset(sb)
        sb.removenodemapping(VNRSS.reqs[VNRindex])
        sb.removeedegemapping(VNRSS.reqs[VNRindex])
        VNRSS.reqs_ids.remove(VNRSS.reqs_ids[VNRindex])
        VNRSS.reqs.remove(VNRSS.reqs[VNRindex])
        VNRSS.vedges.remove(VNRSS.vedges[VNRindex])
        VNRSS.vnfs.remove(VNRSS.vnfs[VNRindex])
        VNRSS.num_reqs-=1
        del self
        return
    # Remove the VNR from VNRSS, to be used when a VNR placement failure occurs.
    def DropVnr(self,VNRSS):
        """
        Removes the VNR from the VNRSS in case of a placement failure.
        This method performs the following actions:
        1. Retrieves the index of the VNR in the VNRSS based on its ID.
        2. Removes the VNR from various lists in the VNRSS:
        - Removes the VNR ID from the list of request IDs.
        - Removes the VNR from the list of requests.
        - Removes associated edges from the edge list.
        - Removes the VNFs from the VNF list.
        3. Decrements the count of total requests in the VNRSS.
        4. Deletes the current instance of the VNR.
        """
        VNRindex=VNRSS.reqs_ids.index(self.id)
        # Remove VNR from VNRSS
        VNRSS.reqs_ids.remove(VNRSS.reqs_ids[VNRindex])
        VNRSS.reqs.remove(VNRSS.reqs[VNRindex])
        VNRSS.vedges.remove(VNRSS.vedges[VNRindex])
        VNRSS.vnfs.remove(VNRSS.vnfs[VNRindex])
        VNRSS.num_reqs-=1
        del self
        return
    def getNetworkx(self):
        """
        Returns the NetworkX graph representation of the VNR.
        This method provides access to the internal graph structure.
        """
        return self.graph
    # Extracting features to feed into the feature extraction architecture
    def getFeatures(self):
        """
        Extracts and scales various features from the VNR's VNFs for use in a feature extraction architecture.
        This method computes the following features for each VNF:
        1. CPU utilization, scaled by the maximum CPU across all VNFs.
        2. Bandwidth, scaled by the maximum bandwidth across all VNFs.
        3. Average bandwidth per degree, scaled by the maximum average bandwidth.
        4. Maximum bandwidth, scaled by the maximum bandwidth among all VNFs.
        5. Minimum bandwidth, scaled by the minimum bandwidth across all VNFs.
        6. Degree of each VNF, scaled by the maximum degree across all VNFs.
        7. Potentiel Maximum CPU capacity of each VNF, scaled by the maximum capacity across all VNFs.
        The features are concatenated, transposed, and returned as a NumPy array for further processing.
        """
        vnf = self.vnode

        cpu = np.array([el.cpu for el in vnf], dtype=np.float32)
        bw = np.array([el.bw for el in vnf], dtype=np.float32)
        bw_av = np.array([el.bw / el.degree for el in vnf], dtype=np.float32)
        bw_max = np.array([el.max_bw(self.vedege) for el in vnf], dtype=np.float32)
        bw_min = np.array([el.min_bw(self.vedege) for el in vnf], dtype=np.float32)
        degree = np.array([el.degree for el in vnf], dtype=np.float32)
        p_max_cpu = np.array([el.p_maxCpu for el in vnf], dtype=np.float32)

        def safe_scale(x):
            m = np.max(x)
            return x / m if m > 0 else x

        scaled_cpu = safe_scale(cpu)
        scaled_bw = safe_scale(bw)
        scaled_bw_av = safe_scale(bw_av)
        scaled_bw_max = safe_scale(bw_max)
        scaled_bw_min = safe_scale(bw_min)
        scaled_degree = safe_scale(degree)
        scaled_p_max_cpu = safe_scale(p_max_cpu)

        features = np.column_stack(
            [
                scaled_cpu,
                scaled_bw,
                scaled_bw_av,
                scaled_bw_max,
                scaled_bw_min,
                scaled_degree,
                scaled_p_max_cpu,
            ]
        )

        return torch.from_numpy(features).to(_DEVICE_CACHE)
    def getFeatures2(self):
        """
        Extracts and scales various features from the VNR's VNFs for use in a feature extraction architecture,
        while disregarding scalability metrics.
        This method computes the following features for each VNF:
        1. CPU utilization, scaled by the maximum CPU across all VNFs.
        2. Bandwidth, scaled by the maximum bandwidth across all VNFs.
        3. Average bandwidth per degree, scaled by the maximum average bandwidth.
        4. Maximum bandwidth, scaled by the maximum bandwidth among all VNFs.
        5. Minimum bandwidth, scaled by the minimum bandwidth across all VNFs.
        6. Degree of each VNF, scaled by the maximum degree across all VNFs.
        The features are concatenated, transposed, and returned as a NumPy array for further processing.
        """
        vnf = self.vnode

        cpu = np.array([el.cpu for el in vnf], dtype=np.float32)
        bw = np.array([el.bw for el in vnf], dtype=np.float32)
        bw_av = np.array([el.bw / el.degree for el in vnf], dtype=np.float32)
        bw_max = np.array([el.max_bw(self.vedege) for el in vnf], dtype=np.float32)
        bw_min = np.array([el.min_bw(self.vedege) for el in vnf], dtype=np.float32)
        degree = np.array([el.degree for el in vnf], dtype=np.float32)

        def safe_scale(x):
            m = np.max(x)
            return x / m if m > 0 else x

        scaled_cpu = safe_scale(cpu)
        scaled_bw = safe_scale(bw)
        scaled_bw_av = safe_scale(bw_av)
        scaled_bw_max = safe_scale(bw_max)
        scaled_bw_min = safe_scale(bw_min)
        scaled_degree = safe_scale(degree)

        features = np.column_stack(
            [
                scaled_cpu,
                scaled_bw,
                scaled_bw_av,
                scaled_bw_max,
                scaled_bw_min,
                scaled_degree,
            ]
        )

        return torch.from_numpy(features).to(_DEVICE_CACHE)
    def __del__(self):
        return
    def vnr_state(self):
        return {"vnf_state" : [el.__str__() for el in self.vnode],
                "vedges_state" : [el.__str__() for el in self.vedege]}
