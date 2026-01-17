#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:41:52 2022

@author: kaouther
""" 

import abc
import numpy as np


class Node:
    
    def __init__(self,index,cpu):
        self.index=index
        """ Node index/ID """
        self.cpu=cpu
        """ Node's CPU capacity """
        self.bw = 0
        """ The bandwidth for the node (initially set to 0) will be updated as the sum of the bandwidth of all links connected to this node """
        self.neighbors = []
        """ List of neighboring nodes """
        self.links =[]
        """ List of links (edges) connected to the node """
        self.degree = 0
        """ Degree of the node (number of connections) """
        
    @abc.abstractmethod   
    def __str__(self):
        ''' This method returns a string of node characteristics to be printed in msg'''
        return 
    
    def msg(self):
        """
        Print the string representation of the node's characteristics by calling __str__().
        """
        print(self.__str__())
    


class Vnf(Node):
    
    def __init__(self, index,cpu,cpu_max,req,flavor_size,p_scalingUp):
        super().__init__(index, cpu)
        self.flavor = []
        """ A predefined range of CPU capacity options that the VNF can request based on its scaling needs, Each flavor is a multiple of the base CPU (i*self.cpu), as long as it doesn't exceed the maximum CPU capacity (cpu_max)"""
        self.cpu_index = 0
        """ The current index in the flavor range representing the CPU capacity demanded by the VNF """
        i=1
        
        # Flavor Creation 
        #---------------------------------------------------------#
        while (i <= flavor_size and i*self.cpu <= cpu_max):
            self.flavor.append(i*self.cpu)
            i+=1
        #---------------------------------------------------------#

        self.p_maxCpu = 0
        """ 
        This variable indicates the maximum CPU capacity that the VNF can potentially reach during its lifecycle. 
        It is computed by multiplying the scaling factor (p_scalingUp, which represents the probability of scaling up) by the base CPU capacity. 
        If this potential maximum exceeds the highest available flavor capacity, it is set to that maximum flavor value. 
        Otherwise, it retains the value calculated from the scaling factor.
        """
        if p_scalingUp*self.cpu>self.flavor[len(self.flavor)-1]:
            self.p_maxCpu=self.flavor[len(self.flavor)-1]
        else:
            self.p_maxCpu=p_scalingUp*self.cpu

        self.req = req
        """ ID of the VNR that includes this VNF """
        self.sn_host = None
        """ The ID of the substrate node that hosts thhis VNF. """
        self.req_cpu = 0
        """ The CPU requirement for the VNF when a scaling request (either up or down) occurs. """
        self.placed_flag = -1
        """ A flag indicating the placement status of the VNF.
            - If the value is -1, the VNF is not placed yet.
            - If the value is 1, the VNF is placed.
        """
        self.current = 0
        """
        A flag indicating whether the VNF is the currently selected one for placement.
            - If the value is 0, the VNF is not the current one.
            - If the value is 1, the VNF is the current one for placement.
        """

    def max_bw(self,edges):
        """
        Returns the maximum bandwidth from the list of links associated with this VNF node.
        This is calculated by checking the bandwidth of each edge in the 'edges' list that 
        connects to this node.
        """
        bw =[edges[i].bandwidth for i in self.links]
        return np.max(bw)
    
    def min_bw(self,edges):
        """
        Returns the minimum bandwidth from the list of links associated with this VNF node.
        This is calculated by checking the bandwidth of each edge in the 'edges' list that 
        connects to this node's links.
        """
        bw =[edges[i].bandwidth for i in self.links]
        return np.min(bw)
    



        
    def __str__(self):
       """ Returns a dictionary containing key attributes of the VNF """
       return {'vnf':str(self.index),'cpu':str(self.cpu),'cpu index':str(self.cpu_index),'flavor':self.flavor,'max_cpu':str(self.p_maxCpu),'bw':str(self.bw),'neighbors':str(self.neighbors),'placement':str(self.sn_host)}
    
    def dm_scaling(self,scale_up):
        """
        Adjusts the CPU demand of the VNF based on scaling needs (up or down) if the flavor options allow.
        
        - If 'scale_up' is True and the current CPU index is not the maximum, increase the CPU request to the next flavor level.
        - If 'scale_up' is False and the current CPU index is not the minimum, decrease the CPU request to the previous flavor level.
        
        Returns the difference between the new CPU demand and the current CPU, and the updated CPU request.
        """
        self.req_cpu=self.cpu
        if scale_up==True and self.cpu_index<len(self.flavor)-1:
            self.req_cpu=self.flavor[self.cpu_index+1]
            self.cpu_index+=1
        elif scale_up==False and self.cpu_index>0:
            self.req_cpu= self.flavor[self.cpu_index-1]
            self.cpu_index-=1
        return self.req_cpu-self.cpu, self.req_cpu
    

        
class Snode(Node):
    
    def __init__(self,index,cpu):

        super().__init__(index, cpu)
        self.lastcpu=cpu
        """ Remaining CPU capacity available in the substrate node after resource allocation. """
        self.lastbw =0
        """ The total remaining bandwidth of all links connected to this node after resource allocation. """
        self.vnodeindexs = []
        """ A list of VNFs hosted on this node, where each entry contains the VNR ID and the corresponding VNF ID in the format [vnr.id,vnf.index]"""
        self.p_load=0
        """ The node's potential load, representing the total potential load from all VNFs hosted on this node. It is calculated as the sum of the maximum potential CPU of each hosted VNF, divided by the node's maximum CPU capacity. """
        

    def __str__(self):
        """ Returns a dictionary containing key attributes of the Snode """
        return {'snode' :str(self.index), 'cpu': str(self.cpu), 'lastcpu': str(self.lastcpu),'vnodeindexs':self.vnodeindexs,'neighbors': self.neighbors,'bw':str(self.bw),'lastbw':str(self.lastbw),'p_load':str(self.p_load)}

    def max_bw(self,edges):
        """Returns the maximum remaining bandwidth from all the links connected to this node."""
        bw =[edges[i].lastbandwidth for i in self.links]
        return np.max(bw)
        
    def min_bw(self,edges):
        """Returns the minimum remaining bandwidth from all the links connected to this node."""
        bw =[edges[i].lastbandwidth for i in self.links]
        return np.min(bw)

    
    