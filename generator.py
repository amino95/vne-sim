#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:02:36 2022

@author: kaouther
"""
import numpy as np
from vnr import VNR
from copy import deepcopy as dc

class VNRS :
    """ 
    A class that encapsulates the VNRs currently present in the system, along with their information. 
    This class facilitates the management and manipulation of VNRs throughout the simulation process. 
    """
    def __init__(self,num_reqs):
        self.num_reqs = num_reqs
        """ The total count of VNRs present in the system."""
        self.reqs_ids=[]
        """ A list containing the IDs of the VNRs currently present in the system."""
        self.reqs = []
        """ A list containing the VNRs currently present in the system."""
        self.vnfs = []
        """ A list containing the VNFs of the VNRs currently present in the system."""
        self.vedges = []
        """ A list containing the Vedges of the VNRs currently present in the system."""

    def msg(self):
        """Displays details of all VNRs currently present in the system."""
        n = self.num_reqs
        for i in range(n):
            print('----------------------')
            self.reqs[i].msg()
    

class Generator():
    """ 
    A class responsible for generating Virtual Network Requests (VNRs).
    """
    def __init__(self,vnr_classes,mlt,mtbs,mtba,vnfs_range,vcpu_range,vbw_range,vlt_range,flavor_tab,p_flavors,nb_solvers):

        self.vnr_classes=vnr_classes
        """
        A list of VNR classes, where each entry represents the probability or proportion for generating each class of VNRs uniformly. 
        For example, [0.2, 0.3, 0.5] indicates that 20% of the generated VNRs will belong to class 0, 30% to class 1, and 50% to class 2. 
        These classes vary in terms of their lifetime and the intervals between scaling events.
        """
        self.mlt=mlt
        """
        A list containing the mean lifetimes for each of the VNR classes. 
        Each entry corresponds to the average duration that VNRs of that class are expected to remain in the system before termination.
        """
        self.mtbs=mtbs
        """
        A list that holds the mean time between scaling arrivals for each VNR class. 
        Each entry represents the average interval before a scaling demand occurs for VNRs of that class.
        """
        self.mtba=mtba
        """ 
        The average time interval between the arrivals of VNRs. 
        """
        self.vnfs_range=vnfs_range
        """ 
        The range specifying the number of VNFs that can be included in a VNR chain. 
        """
        self.vcpu_range=vcpu_range
        """ 
        The range of CPU capacities available for each VNF within the VNR. 
        """
        self.vbw_range=vbw_range
        """ 
        The range of bandwidth capacities available for each Vedge within the VNR. 
        """
        self.vlt_range=vlt_range
        """ 
        Latency associated with Vedges (currently unused in this version). 
        """
        self.flavor_tab=flavor_tab
        """  The flavor of a VNF is a predefined range of CPU capacity options that the VNF can request based on its scaling needs, Each flavor is a multiple of the base CPU (i*self.cpu), 
        as long as it doesn't exceed the maximum CPU capacity (cpu_max), Flavor_tab is a list of flavor lenght, each VNR has a favor list of a lenght from flavor_tab"""
        """ 
        The flavor of a VNF represents a predefined set of CPU capacity options that the VNF can request based on its scaling needs. 
        Each flavor is defined as a multiple of the base CPU (i * self.cpu), provided it does not exceed the maximum CPU capacity (vcpu_range[1]). 
        The flavor_tab is a list that determines the lengths of flavor options for each VNR, with each VNR having its own flavor list based on this tab.
        """
        self.p_flavors=p_flavors
        """ 
        The probability or proportion for uniformly selecting a flavor length from the flavor_tab.
        """
        self.nb_solvers=nb_solvers
        """ 
        Number of solvers that will be used to place VNRs in the subnetworks (SN).
        """

    def generate_flavor(self):
           """ Generates a flavor length for the VNF based on predefined probabilities. """
           u=np.random.uniform(0,1)
           p_sum=0
           for i in range (len(self.flavor_tab)):
               p_sum+=self.p_flavors[i]
               if u<p_sum:
                   return self.flavor_tab[i]
        
    def vnr_ClassGenrator(self):
        """ Generates a VNR class index based on predefined probabilities. """
        nb_classes=len(self.vnr_classes)
        u=np.random.uniform(0,1)
        p_sum=0
        for i in range(nb_classes):
            p_sum+=self.vnr_classes[i]
            if u<p_sum:
                return i
       
    def VnrGenerator_poisson(self,env,manoSimulator):
        """ 
        Generates VNRs based on a Poisson arrival process. 
        This method continuously creates new VNRs and send it to the mano to manage their lifecycle in the simulation environment .

        Steps involved:
        1. Initializes a list `VNRSS` to hold VNR instances for each solver.
        2. During the Simulation time: 
            - Waits for the next arrival time, drawn from an exponential distribution defined by `mtba` (mean time between arrivals).
            - Determines the class of the new VNR using the `vnr_ClassGenrator` method, which assigns a class based on predefined probabilities.
            - Generates a duration for the VNR, again using an exponential distribution based on the mean lifetime for the selected class (`mlt`).
            - Calls `generate_flavor()` to determine the size of the flavor for the VNR.
            - Creates a new `VNR` instance with the specified parameters.
            
        3. For each solver:
            - Clones the newly created `VNR` instance to create a distinct request for each solver.
            - Appends the new VNR to the corresponding `VNRSS` list.
            - Updates the VNR-related lists (`reqs_ids`, `vnfs`, and `vedges`) in `VNRSS` with the newly added VNR information.
            - Increments the request count (`num_reqs`) for the current VNRSS.
            
        4. After generating the VNR, initiates the lifecycle process of the VNR using `manoSimulator.vnr_life_cycle`.

        Note: This method employs a Poisson distribution to model the random arrival of VNRs, facilitating the simulation of network demand over time.
        """

        VNRSS=[]
        
        for i in range(self.nb_solvers):
            VNRSS.append(VNRS(0))
        while True:
            vnrs =[]
            next_arrival = np.random.exponential(self.mtba)
            yield env.timeout(next_arrival)
            vnr_class = self.vnr_ClassGenrator()
            duration  = np.random.exponential(self.mlt[vnr_class])
            flavor_size=self.generate_flavor()
            request = VNR(self.vnfs_range,self.vcpu_range,self.vbw_range,self.vlt_range,flavor_size,duration,self.mtbs[vnr_class])
            for i in range(self.nb_solvers):
                vnr=dc(request)
                vnrs.append(vnr)
                VNRSS[i].reqs.append(vnr)
                VNRSS[i].reqs_ids.append(vnr.id)
                VNRSS[i].vnfs.append((VNRSS[i].reqs[VNRSS[i].num_reqs].id,VNRSS[i].reqs[VNRSS[i].num_reqs].vnode))
                VNRSS[i].vedges.append((VNRSS[i].reqs[VNRSS[i].num_reqs].id,VNRSS[i].reqs[VNRSS[i].num_reqs].vedege))
                VNRSS[i].num_reqs+=1
            del request        
            env.process(manoSimulator.vnr_life_cycle(VNRSS,vnrs))
        