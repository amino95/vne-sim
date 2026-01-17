#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:07:12 2022

@author: kaouther
"""
import networkx as nx
import numpy as np
from copy import deepcopy as dc
import random
import torch

from models.Agent import Agent, Transition
from models.DQNAgent import DQNAgent, DQNTransition
from models.observation import Observation

from DGLgraph import Graph as DGLgraph
from DGLgraph import SnGraph as SnDGLgraph


from DGLgraph2 import Graph as DGLgraph2
from DGLgraph2 import SnGraph as SnDGLgraph2

import abc 
import math



class Solver():
    
    def __init__(self, sigma,rejection_penalty):
        self.rejection_penalty= rejection_penalty
        """ A variable representing the penalty assigned to the solver if it fails the VNR placement. """
        self.itteration = False
        """ A boolean indicating if the solver has multiple attempts to place the VNR: 
        False if the solver has a single chance, True if it can try multiple times. """
        self.max_itteration = None
        """ The maximum number of attempts given to the solver to find a feasible placement. 
        It is None if the iteration variable is False, or a positive integer if True"""
        self.sigma=sigma
        """ A parameter used to calculate the reward given to the solver after a successful VNR placement. """

        
    def rev2cost(self,vnr):
        """
        Calculate the revenue-to-cost (R2C) ratio for a VNR placement.
        
        The R2C ratio is determined by dividing the total bandwidth of the VNR edges by 
        the sum of the total bandwidth and the duplicate bandwidth (when an edge is mapped 
        to multiple paths).

        Args:
            vnr: The virtual network request (VNR) containing edge information.

        Returns:
            float: The R2C ratio (revenue to cost).
        """
        vsum_edeges = 0
        for i in range(len(vnr.vedege)):
            vsum_edeges = vsum_edeges+ vnr.vedege[i].bandwidth
        dup=0
        for i in range(len(vnr.vedege)):
            if len(vnr.vedege[i].spc)>1:
                dup=dup+vnr.vedege[i].bandwidth*(len(vnr.vedege[i].spc)-1)
        return vsum_edeges/(vsum_edeges+dup)
    
    
    def shortpath(self,G,fromnode,tonode,weight=None):
        """
        Finds a random shortest path between two nodes in a graph using the NetworkX library.

        This function computes all shortest paths from `fromnode` to `tonode` in the given graph `G` and 
        randomly selects one of these paths. The function then returns the indices of the edges along 
        the chosen path 
        Args:
            G: The graph in which the shortest path is computed (NetworkX graph of the SN).
            fromnode: The source node in the graph.
            tonode: The target node in the graph.
            weight: Optional; a string representing the edge weight attribute for the shortest path calculation.

        Returns:
            tuple:
                - sedegeindex: A list of edge indices along the chosen path.
                - cost: An integer representing the cost of the path (default is 0).
        
        Raises:
            nx.NetworkXNoPath: If no path exists between the nodes, returns empty lists for both edge indices and cost.
        """

        try:
            sedegeindex=[]
            all_shortest_paths =[p for p in nx.all_shortest_paths(G,fromnode,tonode,weight=weight)]
            choosen_path= np.random.choice(range(len(all_shortest_paths)))
            path = all_shortest_paths[choosen_path]
            for i in range(len(path)-1):
                sedegeindex.append(G[path[i]][path[i+1]]["index"])
            cost=0
        except nx.NetworkXNoPath:
            return [],[]
        return sedegeindex,cost
    
    def Sn2_networkxG(self,snode,sedege,bandlimit=0):
        """
        Converts the substrate network (represented by nodes and edges) into a NetworkX graph.

        This function creates a NetworkX graph `g` from a list of substrate nodes (`snode`) and edges (`sedege`).
        Only edges with bandwidth greater than the specified `bandlimit` are added to the graph. The graph's nodes
        are added using the index of each substrate node, and the edges are added with their bandwidth, latency, and index 
        as attributes.

        Args:
            snode: A list of substrate nodes of type Snode
            sedege: A list of substrate edges of type Sedege
            bandlimit: A threshold value to filter edges. Only edges with `lastbandwidth` greater than `bandlimit` 
                    are added to the graph (default is 0).

        Returns:
            g: A NetworkX graph object representing the substrate network.
        """
        g=nx.Graph()
        for s in snode:
            g.add_node(s.index)
        en=len(sedege)
        for i in range(en):
            if  sedege[i].lastbandwidth>bandlimit:
                g.add_edge(sedege[i].nodeindex[0], sedege[i].nodeindex[1],bandwidth=sedege[i].lastbandwidth,latency=sedege[i].latency,index=sedege[i].index)
        return g
    
    @abc.abstractmethod 
    def mapping(self,sb,vnr):
        return 
    
    @abc.abstractmethod 
    def nodemapping(self,sb,vnr):
        return 
    
    @abc.abstractmethod
    def scaling_down(self,vnr,sn,scaling_chaine):
        return

    @abc.abstractmethod
    def scaling_up(self,vnr,sn,scaling_chaine):
        return
    
    def edegemapping(self,asnode,asedege,vnr,v2sindex):
        """
        Maps virtual edges (links between virtual nodes in the VNR) to substrate edges (links between substrate nodes) 
        in the substrate network. This method Try to map all the edges in the VNR. 
        
        This function attempts to find paths in the substrate network for each virtual edge in the VNR. If a path is found,
        it updates the bandwidth resources along that path and records the mapping from virtual edges to substrate edges.

        Args:
            asnode: A list of substrate nodes.
            asedege: A list of substrate edges.
            vnr: The virtual network request (VNR)
            v2sindex: A mapping of virtual node indices to substrate node indices (i.e., where each virtual node is mapped).

        Returns:
            success (bool): True if all virtual edges are successfully mapped to substrate edges, False if any mapping fails.
            ve2seindex (list): A list of lists, where each inner list contains the substrate edge indices that correspond 
                            to the path mapped to each virtual edge.
        """

        success=True
        ve2seindex=[]
        ven=len(vnr.vedege)
        for i in range(ven):
            fromnode=vnr.vedege[i].nodeindex[0]
            tonode = vnr.vedege[i].nodeindex[1]
            fromnode=v2sindex[fromnode]
            tonode=v2sindex[tonode]
            bandlimit=vnr.vedege[i].bandwidth
            g=self.Sn2_networkxG(asnode,asedege,bandlimit)
           
            pathindex,cost=self.shortpath(g,fromnode,tonode,weight=None)
            if not pathindex:
                return False,[]
            for j in pathindex:
                asedege[j].lastbandwidth=asedege[j].lastbandwidth-vnr.vedege[i].bandwidth
                asedege[j].vedegeindexs.append([vnr.id,i])
                nodeindex=asedege[j].nodeindex
                asnode[nodeindex[0]].lastbw-=vnr.vedege[i].bandwidth
                asnode[nodeindex[1]].lastbw-=vnr.vedege[i].bandwidth
  
            ve2seindex.append(pathindex)
        return success,ve2seindex
    
    def getReward(self,vnr,sn):
        """
        Calculates the reward for placing a virtual network request (VNR) in the substrate network. 
        The reward is computed as: Reward = sigma * R2C + (1 - sigma) * e^(-p_load)

        Args:
            vnr: The virtual network request 
            sn: The substrate network

        Returns:
            tuple: A tuple containing:
                - r2c (float): The revenue-to-cost ratio for the placement of the VNR.
                - p_load (float): The average p_load of the substrate nodes involved in the VNR placement.
                - reward (float): The calculated reward based on the revenue-to-cost ratio and the average p_load.
        """    
        r2c=self.rev2cost(vnr)
        '''
        p_load=0
        for i in range(vnr.num_vnfs):
            p_load=p_load+sn.snode[vnr.nodemapping[i]].p_load
        p_load=p_load/vnr.num_vnfs
        
        return r2c,p_load,self.sigma*r2c+(1-self.sigma)/math.exp(p_load)
        '''
    
        # Collect p_load of all mapped nodes
        p_loads = []
        for i in range(vnr.num_vnfs):
            p_loads.append(sn.snode[vnr.nodemapping[i]].p_load)
        
        p_load_mean = np.mean(p_loads)
        p_load_std = np.std(p_loads)  # Mesure de variance/équilibre
        
        # Récompenser une distribution équilibrée (faible variance)
        # Plus l'écart-type est petit, plus balance_factor est proche de 1
        balance_factor = 1.0 / (1.0 + p_load_std)  # Varie entre 0 et 1
        
        reward = self.sigma * r2c + (1 - self.sigma) * math.exp(-p_load_mean) * balance_factor
        
        return r2c,p_load_mean,reward
    
    
class GNNDQN(Solver):
    
    def __init__(self, sigma,gamma,rejection_penalty,  learning_rate, epsilon, memory_size, batch_size, num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,max_itteration,eps_min , eps_dec ):
        super().__init__(sigma,rejection_penalty)
        
        self.saved_reward= None
        """ 
        A variable used to store the reward from the last placement. 
        This reward will be used to save a transition in the agent's memory. 
        A transition consists of a sequence of steps, where each step corresponds to a VNF placement. 
        It includes the observation (the system state), the action taken, the reward received, 
        and a boolean indicating if all VNFs have been placed.
        """
        self.saved_observation= None
        """ 
        A variable used to store the observation from the last placement. 
        This observation will be used to save a transition in the agent's memory. 
        A transition consists of a sequence of steps, where each step corresponds to a VNF placement. 
        It includes the observation (the system state), the action taken, the reward received, 
        and a boolean indicating if all VNFs have been placed.
        """
        self.saved_action= None
        """ 
        A variable used to store the action taken from the last placement. 
        This action will be used to save a transition in the agent's memory. 
        A transition consists of a sequence of steps, where each step corresponds to a VNF placement. 
        It includes the observation (the system state), the action taken, the reward received, 
        and a boolean indicating if all VNFs have been placed.
        """
        self.saved_done= None
        """ 
        A boolean used to store the variable done from the last placement. 
        This boolean will be used to save a transition in the agent's memory. 
        A transition consists of a sequence of steps, where each step corresponds to a VNF placement. 
        It includes the observation (the system state), the action taken, the reward received, 
        and a boolean indicating if all VNFs have been placed.
        """
        self.agent=DQNAgent(gamma, learning_rate, epsilon, memory_size, batch_size, num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,eps_min, eps_dec )
        """ The DRL Agent, here it is a DQN Agent that uses DQN to learn"""
        self.saved_transition = None
        """ A transition consists of a sequence of steps, where each step corresponds to a VNF placement. 
        It includes the observation (the system state), the action taken, the reward received, 
        and a boolean indicating if all VNFs have been placed. We use this variable to save a transition of the last placement.  """
        
        self.itteration = True
        """ A boolean indicating whether this solver has more than one chance to attempt finding a placement. """
        
        self.max_itteration = max_itteration
        """  The maximum number of iterations allowed to find a feasible placement. """


    def mapping(self, sn, vnr):
            """
            This function performs the mapping of a VNR's virtual nodes (VNFs) and virtual edges onto the substrate network (SN).

            It uses a reinforcement learning agent to iteratively place each VNF, checking if both node and edge mappings are feasible.
            If the mapping is successful, it calculates the reward based on R2C and  p_load .

            Args:
            - sn: The substrate network (SN) object.
            - vnr: The virtual network request (VNR) object.

            Returns: A dictionary containing the following keys:

            1. success (bool): Indicates if the mapping was successful.
                - True: All VNFs and edges were mapped successfully.
                - False: Mapping failed for at least one VNF or edge.
            2. nodemapping (list): The mapping of VNR's virtual nodes (VNFs) to substrate nodes.
                - Non-empty list if mapping is successful; empty if not.
            3. edgemapping (list): The mapping of VNR's virtual edges (VLinks) to substrate edges.
                - Non-empty list if mapping is successful; empty if not.
            4. nb_vnfs (int): The number of successfully mapped VNFs.
                - Number of VNFs in VNR if successful, 0 if not.
            5. nb_vls (int): The number of successfully mapped virtual links (edges).
                - Number of VLinks in VNR if successful, 0 if not.
            6. R2C (float): The calculated revenue-to-cost ratio for the mapping.
                - 0 if mapping is unsuccessful.
            7. p_load (float): The average p_load of the mapped nodes.
                - 0 if mapping is unsuccessful.
            8. reward (float): The reward value calculated based on the mapping success.
                - The rejection penalty if mapping is unsuccessful.
            9. sn (object): The current state of the substrate network.
            10. cause (str): The reason for mapping failure, "node" or "edge".
                - None if mapping is successful
            11. nb_iter (int): The number of iterations taken to find a mapping.
                - 0 for both successful and unsuccessful mappings, an it will be updated by the global solver that calculates the number of itteration 
            """
            r2c=0
            p_load = 0
            success=False
            cause = None
            num_vnfs=vnr.num_vnfs
            ve2seindex=[-1]*vnr.num_vedges 
            vnr.nodemapping=[-1]*num_vnfs
            sn_c=sn.copy_for_placement()  # Fast copy instead of deepcopy
            sn_graph=SnDGLgraph(sn_c,vnr.vnode[0].cpu)
            vnr_graph=DGLgraph(vnr)
            obs=Observation(sn_graph, vnr_graph, 0, [])
            transition= DQNTransition()
            if vnr.id>1  :
                self.saved_transition.store_step(self.saved_observation,self.saved_action,self.saved_reward, True,obs)
                self.agent.store_transition(self.saved_transition)
                self.agent.learn()
        
            for idx in range(num_vnfs):
                nsuccess,action,v2sindex_c=self.nodemapping(obs,sn_c,vnr,idx)

                if nsuccess:
                    if idx>0:
                       
                        nodes_mapped=[i for i in range(idx+1)]
                        vsuccess=self.edegemapping(sn_c, vnr,idx,nodes_mapped,ve2seindex) 
                        if vsuccess:
                            if idx == num_vnfs-1:

                                success= True
                                r2c,p_load,self.saved_reward=self.getReward(vnr, sn_c)
                                self.saved_observation=dc(obs)
                                self.saved_action=action
                                self.saved_done=True
                                self.saved_transition = transition

                                vnr.edgemapping=ve2seindex
               
                            else:
                                reward=0
                                sn_graph=SnDGLgraph(sn_c,vnr.vnode[idx+1].cpu)
                                vnr_graph=DGLgraph(vnr)
                                new_obs=Observation(sn_graph, vnr_graph, idx+1,vnr.nodemapping[:idx+1])
                                transition.store_step(obs,action,reward, False,new_obs)
                                obs=new_obs
                        else:
                           
                            cause = 'edge'
                            self.saved_reward=self.rejection_penalty 
                            self.saved_observation= dc(obs)
                            self.saved_action= action
                            self.saved_done= False 
                            self.saved_transition = transition
                            success=False
                            break
                            
                            
                    else:
                        reward=0
                        sn_graph=SnDGLgraph(sn_c,vnr.vnode[idx+1].cpu)
                        vnr_graph=DGLgraph(vnr)
                        new_obs=Observation(sn_graph, vnr_graph, idx+1,vnr.nodemapping[:idx+1])
                        transition.store_step(obs,action,reward, False,new_obs)
                        obs=new_obs

                else:
                    self.saved_reward=self.rejection_penalty 
                    self.saved_observation=dc(obs)
                    self.saved_action=action
                    self.saved_done=False
                    self.saved_transition = transition
                    success=False
                    cause = 'node'
                    break
                    
            if success:

                results={'success':success,'nodemapping':vnr.nodemapping,'edgemapping':vnr.edgemapping,'nb_vnfs':vnr.num_vnfs,"nb_vls":vnr.num_vedges,'R2C':r2c,'p_load':p_load,'reward':self.saved_reward,'sn':sn_c,'cause':None,'nb_iter':0}
                return results
            else : 
                results={'success':success,'nodemapping':[],'edgemapping':[],'nb_vnfs':0,"nb_vls":0,'R2C':0,'p_load':0,'reward':self.rejection_penalty ,'sn':sn,'cause':cause,'nb_iter':0}
                return results
            
           
    def nodemapping(self,observation,sn,vnr,idx):
        """ 
        Maps a virtual node (VNF) to a substrate node in the network.

        Parameters:
            observation: The current state observation of the system, see observation class for more details 
            sn: The substrate network.
            vnr: The virtual network request.
            idx: The index of the VNF in the virtual network.

        Returns:
            nsuccess (bool): Indicates if the mapping was successful (True) or not (False).
            action (int): The index of the substrate node selected for mapping the VNF. 
                        Returns -1 if mapping fails.
            value: The value associated with the chosen action (from the agent).
            log_prob: The log probability of the action taken (from the agent).
            vnr.nodemapping (list): Updated mapping of VNR nodes to substrate nodes.
        """
        nsuccess=True
        action=self.agent.choose_action(observation,vnr.vnode[idx].cpu,sn.getCpu())
        if action>=0 and sn.snode[action].lastcpu> vnr.vnode[idx].cpu:
            sn.snode[action].lastcpu-=vnr.vnode[idx].cpu
            sn.snode[action].vnodeindexs.append([vnr.id,idx])
            sn.snode[action].p_load=(sn.snode[action].p_load*sn.snode[action].cpu+vnr.vnode[idx].p_maxCpu)/sn.snode[action].cpu
            vnr.nodemapping[idx]=action
            vnr.vnode[idx].sn_host=action
        else:
            nsuccess=False
        return nsuccess,action,vnr.nodemapping
         
    def edegemapping(self,sn,vnr,idx,nodes_mapped,ve2seindex):
        """ 
        Maps the edges that link mapped VNFs to the VNF idx

        Parameters:
            sn: The substrate network.
            vnr: The virtual network request.
            idx: The index of the last VNF placed .
            nodes_mapped: A list of nodes that have already been mapped to substrate nodes.
            ve2seindex: A list to store the mapping of virtual edges to substrate edges.

        Returns:
            success (bool): Indicates if the edge mapping was successful (True) or not (False).
        """
        success=True
        neighbors=vnr.vnode[idx].neighbors
        mapped=nodes_mapped
        intersection= np.intersect1d(neighbors,mapped)
        for i in intersection:
            if idx< i :
                s=idx
                d=i
            else: 
                s=i
                d=idx
            edege_list=list(vnr.graph.edges())
            index=edege_list.index((s,d))
            bw=vnr.vedege[index].bandwidth
            fromnode=vnr.nodemapping[s]
            tonode=vnr.nodemapping[d]
            g=self.Sn2_networkxG(sn.snode,sn.sedege,bw)
            pathindex,cost=self.shortpath(g,fromnode,tonode,weight=None)
            if not pathindex:
                return False
            for j in pathindex:
                sn.sedege[j].lastbandwidth-=vnr.vedege[index].bandwidth
                sn.sedege[j].vedegeindexs.append([vnr.id,index])
                nodeindex=sn.sedege[j].nodeindex
                sn.snode[nodeindex[0]].lastbw-=vnr.vedege[index].bandwidth
                sn.snode[nodeindex[1]].lastbw-=vnr.vedege[index].bandwidth
            vnr.vedege[index].spc = pathindex
            ve2seindex[index]=pathindex
        return success


    def scaling_down(self,vnr,sn,scaling_chaine):
        """ 
        Scales down the resource allocation for the virtual nodes in the VNR.

        This function updates the CPU resources of the virtual nodes and the 
        corresponding substrate nodes when scaling down. It reduces the CPU 
        allocation for each virtual node specified in the scaling chain and 
        increases the available CPU resources in the corresponding substrate node.

        Parameters:
            vnr: The virtual network request.
            sn: The substrate network containing the substrate nodes.
            scaling_chaine: A list of indices representing the virtual nodes that need to be scaled down.

        Returns:
            sn: The updated substrate network after scaling down the resources.
        """
        for i in scaling_chaine:
            vnr.vnode[i].cpu-=vnr.vnode[i].req_cpu
            sn.snode[vnr.vnode[i].sn_host].lastcpu+=vnr.vnode[i].req_cpu
        return sn     
               
    def scaling_up(self,vnr,sn,scaling_chaine):
        """ 
        Scales up the resource allocation for the virtual nodes in the VNR.

        This function attempts to increase the CPU resources for each virtual node specified 
        in the scaling chain. If the corresponding substrate node has sufficient CPU 
        resources available, the function updates both the substrate node and the virtual 
        node's CPU allocation. If any virtual node cannot be scaled up due to resource 
        constraints, it returns a failure status.

        Parameters:
            vnr: The virtual network request.
            sn: The substrate network.
            scaling_chaine: A list of indices representing the virtual nodes that need to be scaled up.

        Returns:
            dict: A dictionary containing:
                - "success": A boolean indicating whether the scaling operation was successful.
                - "sn": The updated substrate network.
                - "vnr": The updated virtual network request.
        """
        sn_c=sn.copy_for_placement()  # Fast copy instead of deepcopy
        vnr_c=dc(vnr)
        remapping_nodes=[]
        for i in scaling_chaine:
            if sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu >vnr_c.vnode[i].req_cpu:
                #vertical scalability
                sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu-=vnr_c.vnode[i].req_cpu
                vnr_c.vnode[i].cpu+=vnr_c.vnode[i].req_cpu
            else:
                remapping_nodes.append(i)     
        if len(remapping_nodes)>0:
            return {"success":False, "sn":sn,"vnr":vnr}
        else:
           
            return {"success":True, "sn":sn_c,"vnr":vnr_c}


class GNNDRL(Solver):
    
    def __init__(self, sigma,gamma,rejection_penalty,  learning_rate, epsilon, memory_size, batch_size, num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,max_itteration,eps_min , eps_dec):
        super().__init__(sigma,rejection_penalty)

        self.agent=Agent(gamma, learning_rate, epsilon, memory_size, batch_size, num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,eps_min, eps_dec )
        """ The DRL Agent"""
        self.saved_transition = None
        """ A transition consists of a sequence of steps, where each step corresponds to a VNF placement. 
        It includes the observation (the system state), the action taken, the reward received, 
        and a boolean indicating if all VNFs have been placed. We use this variable to save a transition of the last placement.  """
        self.itteration = True
        """ A boolean indicating whether this solver has more than one chance to attempt finding a placement. """
        self.max_itteration = max_itteration
        """  The maximum number of iterations allowed to find a feasible placement. """

    def mapping(self, sn, vnr):
            """
            This function performs the mapping of a VNR's virtual nodes (VNFs) and virtual edges onto the substrate network (SN).

            It uses a reinforcement learning agent to iteratively place each VNF, checking if both node and edge mappings are feasible.
            If the mapping is successful, it calculates the reward based on R2C and  p_load .

            Args:
            - sn: The substrate network (SN) object.
            - vnr: The virtual network request (VNR) object.

            Returns: A dictionary containing the following keys:

            1. success (bool): Indicates if the mapping was successful.
                - True: All VNFs and edges were mapped successfully.
                - False: Mapping failed for at least one VNF or edge.
            2. nodemapping (list): The mapping of VNR's virtual nodes (VNFs) to substrate nodes.
                - Non-empty list if mapping is successful; empty if not.
            3. edgemapping (list): The mapping of VNR's virtual edges (VLinks) to substrate edges.
                - Non-empty list if mapping is successful; empty if not.
            4. nb_vnfs (int): The number of successfully mapped VNFs.
                - Number of VNFs in VNR if successful, 0 if not.
            5. nb_vls (int): The number of successfully mapped virtual links (edges).
                - Number of VLinks in VNR if successful, 0 if not.
            6. R2C (float): The calculated revenue-to-cost ratio for the mapping.
                - 0 if mapping is unsuccessful.
            7. p_load (float): The average p_load of the mapped nodes.
                - 0 if mapping is unsuccessful.
            8. reward (float): The reward value calculated based on the mapping success.
                - The rejection penalty if mapping is unsuccessful.
            9. sn (object): The current state of the substrate network.
            10. cause (str): The reason for mapping failure, "node" or "edge".
                - None if mapping is successful
            11. nb_iter (int): The number of iterations taken to find a mapping.
                - 0 for both successful and unsuccessful mappings, an it will be updated by the global solver that calculates the number of itteration 
            """
            r2c=0
            p_load = 0
            success=False
            cause = None
            num_vnfs=vnr.num_vnfs
            ve2seindex=[-1]*vnr.num_vedges 
            vnr.nodemapping=[-1]*num_vnfs
            sn_c=sn.copy_for_placement()  # Fast copy instead of deepcopy
            # Cache VNR graph since VNR doesn't change during placement
            if not hasattr(vnr, '_dgl_graph_cache'):
                vnr._dgl_graph_cache = DGLgraph(vnr)
            vnr_graph = vnr._dgl_graph_cache
            sn_graph=SnDGLgraph(sn_c,vnr.vnode[0].cpu)
            obs=Observation(sn_graph, vnr_graph, 0, [])
            transition= Transition()
            self.saved_reward = 0
            if vnr.id>1  :

                _,last_next_value,_= self.agent.choose_action(obs,vnr.vnode[0].cpu,sn_c.getCpu())
                self.saved_transition.next_value = last_next_value.detach() if isinstance(last_next_value, torch.Tensor) else last_next_value  # Keep tensor on GPU
                self.agent.store_transition(self.saved_transition)
                self.agent.learn()
        
            for idx in range(num_vnfs):
                nsuccess,action,value,log_prob,v2sindex_c=self.nodemapping(obs,sn_c,vnr,idx)

                if nsuccess:
                    if idx>0:
                       
                        nodes_mapped=[i for i in range(idx+1)]
                        vsuccess=self.edegemapping(sn_c, vnr,idx,nodes_mapped,ve2seindex) 
                        if vsuccess:
                            if idx == num_vnfs-1:

                                success= True
                                r2c,p_load,reward=self.getReward(vnr, sn_c)
                                transition.store_step(obs,action,reward, True)
                                self.saved_transition = transition
                                vnr.edgemapping=ve2seindex
                                self.saved_reward = reward
                            else:
                                reward=0
                                sn_graph=SnDGLgraph(sn_c,vnr.vnode[idx+1].cpu)
                                new_obs=Observation(sn_graph, vnr_graph, idx+1,vnr.nodemapping[:idx+1])
                                transition.store_step(obs,action,reward, False)
                                obs=new_obs
                        else:
                            cause = 'edge'
                            success=False
                            transition.store_step(obs,action,self.rejection_penalty, False)
                            self.saved_transition = transition
                            break
                            
                            
                    else:
                        reward=0
                        sn_graph=SnDGLgraph(sn_c,vnr.vnode[idx+1].cpu)
                        transition.store_step(obs,action,reward, False)
                        obs=Observation(sn_graph, vnr_graph, idx+1,vnr.nodemapping[:idx+1])
                else:
                    success=False
                    cause = 'node'
                    transition.store_step(obs,action,self.rejection_penalty , False)
                    self.saved_transition = transition
                    break
                    
            if success:

                results={'success':success,'nodemapping':vnr.nodemapping,'edgemapping':vnr.edgemapping,'nb_vnfs':vnr.num_vnfs,"nb_vls":vnr.num_vedges,'R2C':r2c,'p_load':p_load,'reward':self.saved_reward,'sn':sn_c,'cause':None,'nb_iter':0}
                return results
            else : 
                results={'success':success,'nodemapping':[],'edgemapping':[],'nb_vnfs':0,"nb_vls":0,'R2C':0,'p_load':0,'reward':self.rejection_penalty ,'sn':sn,'cause':cause,'nb_iter':0}
                return results
            
           
    def nodemapping(self,observation,sn,vnr,idx):
        """ 
        Maps a virtual node (VNF) to a substrate node in the network.

        Parameters:
            observation: The current state observation of the system, see observation class for more details 
            sn: The substrate network.
            vnr: The virtual network request.
            idx: The index of the VNF in the virtual network.

        Returns:
            nsuccess (bool): Indicates if the mapping was successful (True) or not (False).
            action (int): The index of the substrate node selected for mapping the VNF. 
                        Returns -1 if mapping fails.
            value: The value associated with the chosen action (from the agent).
            log_prob: The log probability of the action taken (from the agent).
            vnr.nodemapping (list): Updated mapping of VNR nodes to substrate nodes.
        """
        nsuccess=True
        action,value,log_prob=self.agent.choose_action(observation,vnr.vnode[idx].cpu,sn.getCpu())
        if action>=0 and sn.snode[action].lastcpu> vnr.vnode[idx].cpu:
            sn.snode[action].lastcpu-=vnr.vnode[idx].cpu
            sn.snode[action].vnodeindexs.append([vnr.id,idx])
            sn.snode[action].p_load=(sn.snode[action].p_load*sn.snode[action].cpu+vnr.vnode[idx].p_maxCpu)/sn.snode[action].cpu
            vnr.nodemapping[idx]=action
            vnr.vnode[idx].sn_host=action
        else:
            nsuccess=False
        return nsuccess,action,value,log_prob,vnr.nodemapping
         

    def edegemapping(self,sn,vnr,idx,nodes_mapped,ve2seindex):
        """ 
        Maps the edges that link mapped VNFs to the VNF idx

        Parameters:
            sn: The substrate network.
            vnr: The virtual network request.
            idx: The index of the last VNF placed .
            nodes_mapped: A list of nodes that have already been mapped to substrate nodes.
            ve2seindex: A list to store the mapping of virtual edges to substrate edges.

        Returns:
            success (bool): Indicates if the edge mapping was successful (True) or not (False).
        """
        success=True
        neighbors=vnr.vnode[idx].neighbors
        mapped=nodes_mapped
        intersection= np.intersect1d(neighbors,mapped)
        for i in intersection:
            if idx< i :
                s=idx
                d=i
            else: 
                s=i
                d=idx
            edege_list=list(vnr.graph.edges())
            index=edege_list.index((s,d))
            bw=vnr.vedege[index].bandwidth
            fromnode=vnr.nodemapping[s]
            tonode=vnr.nodemapping[d]
            g=self.Sn2_networkxG(sn.snode,sn.sedege,bw)
            pathindex,cost=self.shortpath(g,fromnode,tonode,weight=None)
            if not pathindex:
                return False
            for j in pathindex:
                sn.sedege[j].lastbandwidth-=vnr.vedege[index].bandwidth
                sn.sedege[j].vedegeindexs.append([vnr.id,index])
                nodeindex=sn.sedege[j].nodeindex
                sn.snode[nodeindex[0]].lastbw-=vnr.vedege[index].bandwidth
                sn.snode[nodeindex[1]].lastbw-=vnr.vedege[index].bandwidth
            vnr.vedege[index].spc = pathindex
            ve2seindex[index]=pathindex
        return success


    def scaling_down(self,vnr,sn,scaling_chaine):
        """ 
        Scales down the resource allocation for the virtual nodes in the VNR.

        This function updates the CPU resources of the virtual nodes and the 
        corresponding substrate nodes when scaling down. It reduces the CPU 
        allocation for each virtual node specified in the scaling chain and 
        increases the available CPU resources in the corresponding substrate node.

        Parameters:
            vnr: The virtual network request.
            sn: The substrate network containing the substrate nodes.
            scaling_chaine: A list of indices representing the virtual nodes that need to be scaled down.

        Returns:
            sn: The updated substrate network after scaling down the resources.
        """
        for i in scaling_chaine:
            vnr.vnode[i].cpu-=vnr.vnode[i].req_cpu
            sn.snode[vnr.vnode[i].sn_host].lastcpu+=vnr.vnode[i].req_cpu
        return sn     
               
    def scaling_up(self,vnr,sn,scaling_chaine):
        """ 
        Scales up the resource allocation for the virtual nodes in the VNR.

        This function attempts to increase the CPU resources for each virtual node specified 
        in the scaling chain. If the corresponding substrate node has sufficient CPU 
        resources available, the function updates both the substrate node and the virtual 
        node's CPU allocation. If any virtual node cannot be scaled up due to resource 
        constraints, it returns a failure status.

        Parameters:
            vnr: The virtual network request.
            sn: The substrate network.
            scaling_chaine: A list of indices representing the virtual nodes that need to be scaled up.

        Returns:
            dict: A dictionary containing:
                - "success": A boolean indicating whether the scaling operation was successful.
                - "sn": The updated substrate network.
                - "vnr": The updated virtual network request.
        """
        sn_c=sn.copy_for_placement()  # Fast copy instead of deepcopy
        vnr_c=dc(vnr)
        remapping_nodes=[]
        for i in scaling_chaine:
            if sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu >vnr_c.vnode[i].req_cpu:
                #vertical scalability
                sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu-=vnr_c.vnode[i].req_cpu
                vnr_c.vnode[i].cpu+=vnr_c.vnode[i].req_cpu
            else:
                remapping_nodes.append(i)     
        if len(remapping_nodes)>0:
            return {"success":False, "sn":sn,"vnr":vnr}
        else:
           
            return {"success":True, "sn":sn_c,"vnr":vnr_c}



class FirstFit(Solver):
    
    def __init__(self,sigma,rejection_penalty):
        super().__init__(sigma,rejection_penalty)
        
    
    def nodemapping(self, sb, vnr):
        """
        Maps the virtual nodes (VNFs) of a VNR to substrate nodes in the substrate network (sb).

        The function tries to place each VNF onto a substrate node, checking if the substrate node
        has enough available CPU resources (lastcpu). The mapping is done in a random order to balance
        the load across the network. If a virtual node cannot be placed on any substrate node, the
        placement fails.

        Returns:
            - success: Boolean indicating if the mapping was successful or not.
            - v2sindex: List mapping each virtual node to a corresponding substrate node.
        """
        success=True 
        vn=vnr.num_vnfs 
        v2sindex=[] 
        sn = random.sample(list(range(len(sb.snode))), len(sb.snode))
        for vi in range(vn):
            for i,si in enumerate(sn):
                if vnr.vnode[vi].cpu<sb.snode[si].lastcpu and si not in  v2sindex:
                    v2sindex.append(si) 
                    break
                else:
                    if (i==len(sn)-1) : # no substrate node available for placement
                        success=False
                        return success,[]
        return success, v2sindex
    
    def mapping(self,sb,vnr):
        """
        Maps the VNR (virtual network request) to the substrate network (sb).
        First, it tries to map the virtual nodes (VNFs), and if successful, 
        it proceeds to map the edges (links). If either fails, the mapping is considered unsuccessful.
        
        Returns a dictionary containing the mapping results.
        """
        success=True 
        vn=vnr.num_vnfs 
        v2sindex=[]
        vese2index=[]
        nodesuccess, v2sindex=self.nodemapping(sb, vnr)
        if nodesuccess:
            asnode = dc(sb.snode)
            asedge = dc(sb.sedege)
            edgesuccess, vese2index=self.edegemapping(asnode, asedge, vnr, v2sindex)
        if nodesuccess and edgesuccess:
                for i in range(vn):
                    sb.snode[v2sindex[i]].lastcpu=sb.snode[v2sindex[i]].lastcpu-vnr.vnode[i].cpu
                    sb.snode[v2sindex[i]].vnodeindexs.append([vnr.id,vnr.vnode[i].index])
                    sb.snode[v2sindex[i]].p_load=(sb.snode[v2sindex[i]].p_load*sb.snode[v2sindex[i]].cpu+vnr.vnode[i].p_maxCpu)/sb.snode[v2sindex[i]].cpu
                    vnr.vnode[i].sn_host = v2sindex[i]
                for i in range(len(vese2index)):
                    pathindex=vese2index[i]
                    for j in pathindex:
                        sb.sedege[j].lastbandwidth=sb.sedege[j].lastbandwidth-vnr.vedege[i].bandwidth
                        sb.sedege[j].vedegeindexs.append([vnr.id,i])
                        nodeindex=sb.sedege[j].nodeindex
                        sb.snode[nodeindex[0]].lastbw-=vnr.vedege[i].bandwidth
                        sb.snode[nodeindex[1]].lastbw-=vnr.vedege[i].bandwidth
                    vnr.vedege[i].spc = pathindex
                vnr.nodemapping=v2sindex
                vnr.edgemapping=vese2index
                r2c=self.rev2cost(vnr)
                r2c,p_load,metric=self.getReward(vnr, sb)
                results={'success':success,'nodemapping':vnr.nodemapping,'edgemapping':vnr.edgemapping,'nb_vnfs':vnr.num_vnfs,"nb_vls":vnr.num_vedges,'R2C':r2c,'p_load':p_load,'reward':metric,'sn':sb,'cause':None,'nb_iter':None}
                return results

        else:
            success = False
            if not nodesuccess :
                cause = "node"
            else : 
                cause = 'edge'
            results={'success':success,'nodemapping':[],'edgemapping':[],'nb_vnfs':0,"nb_vls":0,'R2C':0,'p_load':0,'reward':self.rejection_penalty,'sn':sb,'cause':cause,'nb_iter':None}    
            return results
        

                
        
    def scaling_down(self,vnr,sn,scaling_chaine):
        """ 
        Scales down the resource allocation for the virtual nodes in the VNR.

        This function updates the CPU resources of the virtual nodes and the 
        corresponding substrate nodes when scaling down. It reduces the CPU 
        allocation for each virtual node specified in the scaling chain and 
        increases the available CPU resources in the corresponding substrate node.

        Parameters:
            vnr: The virtual network request.
            sn: The substrate network containing the substrate nodes.
            scaling_chaine: A list of indices representing the virtual nodes that need to be scaled down.

        Returns:
            sn: The updated substrate network after scaling down the resources.
        """
        for i in scaling_chaine:
            vnr.vnode[i].cpu-=vnr.vnode[i].req_cpu
            sn.snode[vnr.vnode[i].sn_host].lastcpu+=vnr.vnode[i].req_cpu
        return sn 
    
    
    def scaling_up(self,vnr,sn,scaling_chaine):
        """ 
        Scales up the resource allocation for the virtual nodes in the VNR.

        This function attempts to increase the CPU resources for each virtual node specified 
        in the scaling chain. If the corresponding substrate node has sufficient CPU 
        resources available, the function updates both the substrate node and the virtual 
        node's CPU allocation. If any virtual node cannot be scaled up due to resource 
        constraints, it returns a failure status.

        Parameters:
            vnr: The virtual network request.
            sn: The substrate network.
            scaling_chaine: A list of indices representing the virtual nodes that need to be scaled up.

        Returns:
            dict: A dictionary containing:
                - "success": A boolean indicating whether the scaling operation was successful.
                - "sn": The updated substrate network.
                - "vnr": The updated virtual network request.
        """
        sn_c=sn.copy_for_placement()  # Fast copy instead of deepcopy
        vnr_c=dc(vnr)
        remapping_nodes=[]
        for i in scaling_chaine:
            if sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu >vnr_c.vnode[i].req_cpu:
                #vertical scalability
                sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu-=vnr_c.vnode[i].req_cpu
                vnr_c.vnode[i].cpu+=vnr_c.vnode[i].req_cpu
            else:
                remapping_nodes.append(i)     
        if len(remapping_nodes)>0:
            return {"success":False, "sn":sn,"vnr":vnr}
        else:
            return {"success":True, "sn":sn_c,"vnr":vnr_c}



class GNNDRL2(Solver):
    """ Identical to GNNDRL, but used in a scenario where scalability measures are disabled.
    Can also use PPO agent if clip_ratio, ppo_epochs, and entropy_coef are provided."""
    def __init__(self, sigma, gamma, rejection_penalty, learning_rate, epsilon, memory_size, 
                 batch_size, num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,
                 max_itteration, eps_min, eps_dec, clip_ratio=None, ppo_epochs=None, entropy_coef=None):
        super().__init__(sigma, rejection_penalty)

        # Use PPO agent if PPO parameters are provided
        if clip_ratio is not None and ppo_epochs is not None and entropy_coef is not None:
            from models.PPOAgent import PPOAgent
            self.agent = PPOAgent(gamma, learning_rate, epsilon, memory_size, batch_size, 
                                 num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,
                                 eps_min, eps_dec, clip_ratio=clip_ratio, ppo_epochs=ppo_epochs,
                                 entropy_coef=entropy_coef)
        else:
            self.agent = Agent(gamma, learning_rate, epsilon, memory_size, batch_size, 
                             num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,
                             eps_min, eps_dec)
        
        self.saved_transition = None
        self.itteration = True
        self.max_itteration = max_itteration
        
    def mapping(self, sn, vnr):
            r2c=0
            p_load = 0
            success=False
            cause = None
            num_vnfs=vnr.num_vnfs
            ve2seindex=[-1]*vnr.num_vedges 
            vnr.nodemapping=[-1]*num_vnfs
            sn_c=sn.copy_for_placement()  # Fast copy instead of deepcopy
            self.saved_reward = 0
            """ Here's the difference between this class and GNNDRL: 
            here we use SNDGLgraoh2, which doesn't take scalability metrics into account."""
            # Cache VNR graph since VNR doesn't change during placement
            if not hasattr(vnr, '_dgl_graph2_cache'):
                vnr._dgl_graph2_cache = DGLgraph2(vnr)
            vnr_graph = vnr._dgl_graph2_cache
            sn_graph=SnDGLgraph2(sn_c,vnr.vnode[0].cpu) 
            obs=Observation(sn_graph, vnr_graph, 0, [])
            transition= Transition()
            if vnr.id>1  :

                _,last_next_value,_= self.agent.choose_action(obs,vnr.vnode[0].cpu,sn_c.getCpu())
                self.saved_transition.next_value = last_next_value.detach() if isinstance(last_next_value, torch.Tensor) else last_next_value  # Keep tensor on GPU
                self.agent.store_transition(self.saved_transition)
                self.agent.learn()
        
            for idx in range(num_vnfs):
                nsuccess,action,value,log_prob,v2sindex_c=self.nodemapping(obs,sn_c,vnr,idx)

                if nsuccess:
                    if idx>0:
                       
                        nodes_mapped=[i for i in range(idx+1)]
                        vsuccess=self.edegemapping(sn_c, vnr,idx,nodes_mapped,ve2seindex) 
                        if vsuccess:
                            if idx == num_vnfs-1:

                                success= True
                                r2c,p_load,reward=self.getReward(vnr, sn_c)
                                transition.store_step(obs,action,reward, True)
                                self.saved_transition = transition
                                vnr.edgemapping=ve2seindex
               
                            else:
                                reward=0
                                sn_graph=SnDGLgraph2(sn_c,vnr.vnode[idx+1].cpu)
                                new_obs=Observation(sn_graph, vnr_graph, idx+1,vnr.nodemapping[:idx+1])
                                transition.store_step(obs,action,reward, False)
                                obs=new_obs
                        else:
                           
                            cause = 'edge'
                            self.saved_reward=self.rejection_penalty 
                            success=False
                            transition.store_step(obs,action,self.rejection_penalty, False)
                            self.saved_transition = transition
                            break
                            
                            
                    else:
                        reward=0
                        sn_graph=SnDGLgraph2(sn_c,vnr.vnode[idx+1].cpu)
                        transition.store_step(obs,action,reward, False)
                        obs=Observation(sn_graph, vnr_graph, idx+1,vnr.nodemapping[:idx+1])
                else:
                    success=False
                    cause = 'node'
                    transition.store_step(obs,action,self.rejection_penalty , False)
                    self.saved_transition = transition
                    break
                    
            if success:

                results={'success':success,'nodemapping':vnr.nodemapping,'edgemapping':vnr.edgemapping,'nb_vnfs':vnr.num_vnfs,"nb_vls":vnr.num_vedges,'R2C':r2c,'p_load':p_load,'reward':self.saved_reward,'sn':sn_c,'cause':None,'nb_iter':0}
                return results
            else : 
                results={'success':success,'nodemapping':[],'edgemapping':[],'nb_vnfs':0,"nb_vls":0,'R2C':0,'p_load':0,'reward':self.rejection_penalty ,'sn':sn,'cause':cause,'nb_iter':0}
                return results
            
           
    def nodemapping(self,observation,sn,vnr,idx):
        nsuccess=True
        action,value,log_prob=self.agent.choose_action(observation,vnr.vnode[idx].cpu,sn.getCpu())
        if action>=0 and sn.snode[action].lastcpu> vnr.vnode[idx].cpu:
            sn.snode[action].lastcpu-=vnr.vnode[idx].cpu
            sn.snode[action].vnodeindexs.append([vnr.id,idx])
            sn.snode[action].p_load=(sn.snode[action].p_load*sn.snode[action].cpu+vnr.vnode[idx].p_maxCpu)/sn.snode[action].cpu
            vnr.nodemapping[idx]=action
            vnr.vnode[idx].sn_host=action
        else:
            nsuccess=False
        return nsuccess,action,value,log_prob,vnr.nodemapping
         
    def edegemapping(self,sn,vnr,idx,nodes_mapped,ve2seindex):
        success=True
        neighbors=vnr.vnode[idx].neighbors
        mapped=nodes_mapped
        intersection= np.intersect1d(neighbors,mapped)
        for i in intersection:
            if idx< i :
                s=idx
                d=i
            else: 
                s=i
                d=idx
            edege_list=list(vnr.graph.edges())
            index=edege_list.index((s,d))
            bw=vnr.vedege[index].bandwidth
            fromnode=vnr.nodemapping[s]
            tonode=vnr.nodemapping[d]
            g=self.Sn2_networkxG(sn.snode,sn.sedege,bw)
            pathindex,cost=self.shortpath(g,fromnode,tonode,weight=None)
            if not pathindex:
                return False
            for j in pathindex:
                sn.sedege[j].lastbandwidth-=vnr.vedege[index].bandwidth
                sn.sedege[j].vedegeindexs.append([vnr.id,index])
                nodeindex=sn.sedege[j].nodeindex
                sn.snode[nodeindex[0]].lastbw-=vnr.vedege[index].bandwidth
                sn.snode[nodeindex[1]].lastbw-=vnr.vedege[index].bandwidth
            vnr.vedege[index].spc = pathindex
            ve2seindex[index]=pathindex
        return success


    def scaling_down(self,vnr,sn,scaling_chaine):
        for i in scaling_chaine:
            vnr.vnode[i].cpu-=vnr.vnode[i].req_cpu
            sn.snode[vnr.vnode[i].sn_host].lastcpu+=vnr.vnode[i].req_cpu
        return sn     
               
    def scaling_up(self,vnr,sn,scaling_chaine):
        sn_c=sn.copy_for_placement()  # Fast copy instead of deepcopy
        vnr_c=dc(vnr)
        remapping_nodes=[]
        for i in scaling_chaine:
            if sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu >vnr_c.vnode[i].req_cpu:
                #vertical scalability
                sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu-=vnr_c.vnode[i].req_cpu
                vnr_c.vnode[i].cpu+=vnr_c.vnode[i].req_cpu
            else:
                remapping_nodes.append(i)     
        if len(remapping_nodes)>0:
            return {"success":False, "sn":sn,"vnr":vnr}
        else:
           
            return {"success":True, "sn":sn_c,"vnr":vnr_c}


class GNNDRLPPO(Solver):
    """
    GNN-based DRL solver using PPO (Proximal Policy Optimization) algorithm.
    
    PPO provides more stable training than vanilla policy gradients by using
    a clipped objective function that prevents excessively large policy updates.
    """
    
    def __init__(self, sigma, gamma, rejection_penalty, learning_rate, epsilon, 
                 memory_size, batch_size, num_inputs_sn, num_inputs_vnr, hidden_size, 
                 GCN_out, num_actions, max_itteration, eps_min, eps_dec,
                 clip_ratio=0.2, ppo_epochs=4, entropy_coef=0.01):
        super().__init__(sigma, rejection_penalty)
        
        from models.PPOAgent import PPOAgent
        
        self.agent = PPOAgent(
            gamma, learning_rate, epsilon, memory_size, batch_size, 
            num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,
            eps_min, eps_dec, clip_ratio=clip_ratio, ppo_epochs=ppo_epochs,
            entropy_coef=entropy_coef
        )
        """ The PPO Agent """
        
        self.saved_transition = None
        """ Stores the transition from the last placement """
        
        self.itteration = True
        """ Allows multiple attempts to find a feasible placement """
        
        self.max_itteration = max_itteration
        """ Maximum number of placement attempts """

    def mapping(self, sn, vnr):
        """
        Map VNR to substrate network using PPO agent.
        
        Args:
            sn: Substrate network
            vnr: Virtual Network Request
            
        Returns:
            dict: Mapping results including success status, rewards, and resource usage
        """
        r2c = 0
        p_load = 0
        success = False
        cause = None
        num_vnfs = vnr.num_vnfs
        ve2seindex = [-1] * vnr.num_vedges 
        vnr.nodemapping = [-1] * num_vnfs
        sn_c = sn.copy_for_placement()
        
        # Cache VNR graph since VNR doesn't change during placement
        if not hasattr(vnr, '_dgl_graph2_cache'):
            vnr._dgl_graph2_cache = DGLgraph2(vnr)
        vnr_graph = vnr._dgl_graph2_cache
        
        sn_graph = SnDGLgraph2(sn_c, vnr.vnode[0].cpu)
        obs = Observation(sn_graph, vnr_graph, 0, [])
        transition = Transition()
        self.saved_reward = 0
        
        if vnr.id > 1:
            _, last_next_value, _ = self.agent.choose_action(obs, vnr.vnode[0].cpu, sn_c.getCpu())
            self.saved_transition.next_value = last_next_value.detach() if isinstance(last_next_value, torch.Tensor) else last_next_value
            self.agent.store_transition(self.saved_transition)
            self.agent.learn()
        
        for idx in range(num_vnfs):
            nsuccess, action, value, log_prob, v2sindex_c = self.nodemapping(obs, sn_c, vnr, idx)

            if nsuccess:
                if idx > 0:
                    nodes_mapped = [i for i in range(idx + 1)]
                    vsuccess = self.edegemapping(sn_c, vnr, idx, nodes_mapped, ve2seindex) 
                    
                    if vsuccess:
                        if idx == num_vnfs - 1:
                            success = True
                            r2c, p_load, reward = self.getReward(vnr, sn_c)
                            transition.store_step(obs, action, reward, True)
                            self.saved_transition = transition
                            vnr.edgemapping = ve2seindex
                            self.saved_reward = reward
                        else:
                            reward = 0
                            sn_graph = SnDGLgraph2(sn_c, vnr.vnode[idx + 1].cpu)
                            new_obs = Observation(sn_graph, vnr_graph, idx + 1, vnr.nodemapping[:idx + 1])
                            transition.store_step(obs, action, reward, False)
                            obs = new_obs
                    else:
                        cause = 'edge'
                        success = False
                        transition.store_step(obs, action, self.rejection_penalty, False)
                        self.saved_transition = transition
                        break
                else:
                    reward = 0
                    sn_graph = SnDGLgraph2(sn_c, vnr.vnode[idx + 1].cpu)
                    transition.store_step(obs, action, reward, False)
                    obs = Observation(sn_graph, vnr_graph, idx + 1, vnr.nodemapping[:idx + 1])
            else:
                success = False
                cause = 'node'
                transition.store_step(obs, action, self.rejection_penalty, False)
                self.saved_transition = transition
                break
                
        if success:
            results = {
                'success': success, 'nodemapping': vnr.nodemapping, 
                'edgemapping': vnr.edgemapping, 'nb_vnfs': vnr.num_vnfs, 
                "nb_vls": vnr.num_vedges, 'R2C': r2c, 'p_load': p_load, 
                'reward': self.saved_reward, 'sn': sn_c, 'cause': None, 'nb_iter': 0
            }
            return results
        else:
            results = {
                'success': success, 'nodemapping': [], 'edgemapping': [], 
                'nb_vnfs': 0, "nb_vls": 0, 'R2C': 0, 'p_load': 0, 
                'reward': self.rejection_penalty, 'sn': sn, 'cause': cause, 'nb_iter': 0
            }
            return results
           
    def nodemapping(self, observation, sn, vnr, idx):
        """Map a VNF to a substrate node"""
        nsuccess = True
        action, value, log_prob = self.agent.choose_action(observation, vnr.vnode[idx].cpu, sn.getCpu())
        
        if action >= 0 and sn.snode[action].lastcpu > vnr.vnode[idx].cpu:
            sn.snode[action].lastcpu -= vnr.vnode[idx].cpu
            sn.snode[action].vnodeindexs.append([vnr.id, idx])
            sn.snode[action].p_load = (sn.snode[action].p_load * sn.snode[action].cpu + vnr.vnode[idx].p_maxCpu) / sn.snode[action].cpu
            vnr.nodemapping[idx] = action
            vnr.vnode[idx].sn_host = action
        else:
            nsuccess = False
            
        return nsuccess, action, value, log_prob, vnr.nodemapping
    
    def scaling_up(self, vnr, sn, scaling_chaine):
        """
        Scale up VNFs in the scaling chain.
        
        Args:
            vnr: Virtual Network Request
            sn: Substrate network
            scaling_chaine: List of VNF indices to scale up
            
        Returns:
            dict: Results with success status, updated sn and vnr
        """
        sn_c = sn.copy_for_placement()
        vnr_c = dc(vnr)
        remapping_nodes = []
        
        for i in scaling_chaine:
            if sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu > vnr_c.vnode[i].req_cpu:
                # Vertical scalability
                sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu -= vnr_c.vnode[i].req_cpu
                vnr_c.vnode[i].cpu += vnr_c.vnode[i].req_cpu
            else:
                remapping_nodes.append(i)
                
        if len(remapping_nodes) > 0:
            return {"success": False, "sn": sn, "vnr": vnr}
        else:
            return {"success": True, "sn": sn_c, "vnr": vnr_c}
    
    def scaling_down(self, vnr, sn, scaling_chaine):
        """
        Scale down VNFs in the scaling chain.
        
        Args:
            vnr: Virtual Network Request
            sn: Substrate network
            scaling_chaine: List of VNF indices to scale down
            
        Returns:
            sn: Updated substrate network
        """
        for i in scaling_chaine:
            vnr.vnode[i].cpu -= vnr.vnode[i].req_cpu
            sn.snode[vnr.vnode[i].sn_host].lastcpu += vnr.vnode[i].req_cpu
        return sn
         
    def edegemapping(self, sn, vnr, idx, nodes_mapped, ve2seindex):
        """Map virtual edges to substrate paths"""
        success = True
        neighbors = vnr.vnode[idx].neighbors
        
        for i in neighbors:
            if i in nodes_mapped:
                vl = [j for j, vedge in enumerate(vnr.vedege) 
                      if (vedge.nodeindex[0] == i and vedge.nodeindex[1] == idx) 
                      or (vedge.nodeindex[1] == i and vedge.nodeindex[0] == idx)][0]
                
                g = self.Sn2_networkxG(sn.snode, sn.sedege, vnr.vedege[vl].bandwidth)
                sedges, cost = self.shortpath(
                    g, vnr.nodemapping[i], vnr.nodemapping[idx], 
                    weight=None
                )
                
                if not sedges:
                    success = False
                    break
                    
                for sedge in sedges:
                    if sn.sedege[sedge].lastbandwidth < vnr.vedege[vl].bandwidth:
                        success = False
                        break
                        
                if not success:
                    break
                    
                for sedge in sedges:
                    sn.sedege[sedge].lastbandwidth -= vnr.vedege[vl].bandwidth
                    sn.sedege[sedge].vedegeindexs.append([vnr.id, vl])
                    
                vnr.vedege[vl].spc = sedges
                ve2seindex[vl] = sedges
                
        return success


    

class GlobalSolver():
    """
    This class is used to manage all the solvers created in the system, facilitating placement and scaling operations.
    """

    def __init__(self,solvers):
        """
        Initialize the GlobalSolver with a list of solvers.
        
        :param solvers: List of individual solvers that will attempt to map VNRs.
        """
        self.solvers=solvers

    def mapping(self,states):
        """
        Attempt to map VNRs to the substrate network using the available solvers.
        
        :param states: A list of dictionaries containing the 'sn' (substrate network) and 'vnr' (VNR) for each solver.
        :return: A list of results from each solver, containing success status and other metrics.
        """
        results= []
        for i in range(len(self.solvers)):
            results.append(self.solvers[i].mapping(states[i]['sn'],states[i]['vnr']))
            itteration = 1
            while not results[i]["success"] and self.solvers[i].itteration and itteration < self.solvers[i].max_itteration:
                results[i] = self.solvers[i].mapping(states[i]['sn'],states[i]['vnr'])
                itteration+=1
            results[i]['nb_iter']= itteration
        return results

    def scaling_down(self,i,vnr,sn,scaling_chaine):
        """
        Perform a scaling down operation using the ith solver.
        
        :param i: Index of the solver to use.
        :param vnr: The VNR to scale down.
        :param sn: The substrate network.
        :param scaling_chaine: The chain of scaling actions to perform.
        :return: Result of the scaling down operation.
        """
        return self.solvers[i].scaling_down(vnr,sn,scaling_chaine)

    def scaling_up(self,i,vnr,sn,scaling_chaine):
        """
        Perform a scaling up operation using the ith solver.
        
        :param i: Index of the solver to use.
        :param vnr: The VNR to scale up.
        :param sn: The substrate network.
        :param scaling_chaine: The chain of scaling actions to perform.
        :return: Result of the scaling up operation.
        """
        return self.solvers[i].scaling_up(vnr,sn,scaling_chaine)
        

    
 