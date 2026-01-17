#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:42:22 2022

@author: kaouther
"""
import abc

class Edege:
    
    def __init__(self,index,bw,lt,a_t_b):
        self.index=index  
        """ Edge index/ID """          
        self.bandwidth = bw
        """ The bandwidth capacity of the edge. """ 
        self.latency = lt
        """ The latency of the edge. Not used in this version """
        self.nodeindex=[int(a_t_b[0]),int(a_t_b[1])]
        """ The indices of the two nodes connected by the edge, extracted from the a_t_b parameter. """

    @abc.abstractmethod   
    def __str__(self):
        ''' This method returns a string of node characteristics to be printed in msg'''
        return 
    
    def msg(self):
        print(self.__str__())


class Vedege(Edege):
    
    def __init__(self,index,bw,lt,a_t_b):
        super().__init__(index, bw,lt, a_t_b)
        self.spc = [] 
        """ 
        A list representing the path of physical edges (in the substrate network) that are used to map the virtual edge (in the VNR). 
        Each element in the list corresponds to the index of a physical edge that is part of the mapping.
        """  
        
    def __str__(self):
        return {'vedege':self.index,'bandwidth':self.bandwidth,'latency':self.latency,'nodeindex':self.nodeindex,'spc':self.spc}
    

class Sedege(Edege):
    
    def __init__(self,index,bw,lt,a_t_b):
        super().__init__(index, bw, lt,a_t_b)
        self.lastbandwidth = bw
        """ Tracks the available bandwidth after resource allocation to virtual edges. """
        self.vedegeindexs = []
        """ A list of virtual edge indices mapped to this substrate edge. Each element contains the VNR ID and the virtual edge ID. """
        self.open=False
        """If a substrate edge no longer has any mapped virtual edges, it is marked as closed (`open = False`)"""
        self.mappable_flag = False    
        
    def __str__(self):
        return {'sedege': self.index, 'bandwidth': self.bandwidth,'lastbandwidth':self.lastbandwidth, 'latency':self.latency, 'nodeindex': self.nodeindex,'vedegeindexs':self.vedegeindexs}