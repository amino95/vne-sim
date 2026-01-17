#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:17:42 2022

@author: kaouther
"""

class Observation():
    """ Represents an observation of the system's state, which is used to extract the current state and facilitate action selection in the reinforcement learning process"""
    def __init__(self,sn,vnr,idx,node_mapping):
        self.sn=sn
        """ The substrate network at the time of observation."""
        self.vnr=vnr
        """ The virtual network request being processed."""
        self.idx=idx
        """The current VNF index within the VNR placement process."""
        self.node_mapping=node_mapping
        """ The mapping of VNR nodes to substrate network nodes."""