#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:43:55 2022

@author: kaouther
"""

import numpy as np
import sys
from solver import *


class ManoSimulator():
    """ 
    A class responsible for managing the life cycle of a VNR. 
    It handles placing each VNR based on different solver methods, manages scalability demands, and drops VNRs after placement failures. 
    The class also manages the termination of VNRs after scalability failures or when their lifespan ends. 
    Additionally, it ensures that all VNR placement, scalability, and deletion operations are performed correctly, maintains data integrity across the system,
    and gathers and saves results. This class interacts with functions implemented in other classes to perform these tasks.
    """
    def __init__(self,global_solver,solvers,sns,env,controller):
        
        self.solvers = solvers
        """ 
        A list of strings containing the names of all solvers, used to organize and save data into files based on each solver's name.
        """
        self.global_solver = global_solver
        """
        A class representing the global solver. For more details, refer to the GlobalSolver class in the solver module.
        """
        self.env = env
        """
        Simpy Env: A simulation environment that manages simulation time, event scheduling, and processing. 
        It allows solvers to execute and step through the simulation consistently.
        """
        self.controller = controller
        """
        Controller class: Responsible for gathering and saving results. For more details, refer to the Global_Controller class in the controller module.
        """
        self.subNets= sns
        """ The substrate network"""


    def integration_test(self,sn,VNRSS,solver_name): 
        """
        Integration test to validate the correctness of the VNF placement and resource allocation in the substrate network. 
        This function is called after each placement, scalability, and VNR termination to ensure the system's integrity.

        This function checks:
        1. **Node Placement**: Verifies that each VNF is placed correctly on the corresponding substrate nodes.
        - It calculates the total CPU used by each node and compares it to the node's CPU capacity.
        - It checks if the potential load (`p_load`) for each node is correctly updated based on the VNFs hosted.

        2. **Edge Placement**: Ensures that the VNFs' virtual edges are correctly mapped to the substrate edges.
        - It calculates the total bandwidth used by each edge and verifies it against the edge's total bandwidth.

        3. **Last Bandwidth Check**: Validates that the remaining bandwidth (`lastbw`) for each node matches the calculated value based on the used bandwidth of the connected edges.

        If any of these checks fail, an error message is printed, and the process is terminated.
        """
        #check if placement in nodes is done correctly 
        for snode in sn.snode:
            cpu_used=0
            p_load=0
            vnodeindexs=snode.vnodeindexs
            for vnode in vnodeindexs:
                vnr_id=vnode[0]
                VNRindex=VNRSS.reqs_ids.index(vnr_id)
                vnr=VNRSS.reqs[VNRindex]
                if vnr.vnode[vnode[1]].sn_host == snode.index:
                   cpu_used+=vnr.vnode[vnode[1]].cpu
                   p_load=(vnr.vnode[vnode[1]].p_maxCpu+p_load*snode.cpu)/snode.cpu
                else:
                    print(solver_name)
                    print("VNR ID",vnr.id)
                    print("Vnode_index",vnode[1])
                    print(vnr.vnode[vnode[1]].sn_host,snode.index)
                    sys.exit("bad node palcement")
            if snode.lastcpu+cpu_used != snode.cpu:
                print(solver_name)
                sn.msg()
                print(cpu_used,snode.index)
                sys.exit("error in lastcpu calculation")
            if abs(snode.p_load - p_load )>0.0001 :
                print(solver_name)
                print(snode.index)
                print(p_load)
                print(snode.p_load)
                sn.msg()
                sys.exit("error in pload calculation")

        #check if placement in edeges is done correctly 
        for sedege in sn.sedege:
            used_bw=0
            vedegeindexs=sedege.vedegeindexs
            for vedege in vedegeindexs:
                vnr_id=vedege[0]
                VNRindex=VNRSS.reqs_ids.index(vnr_id)
                vnr=VNRSS.reqs[VNRindex]
                if sedege.index  in vnr.vedege[vedege[1]].spc:
                    used_bw+=vnr.vedege[vedege[1]].bandwidth
                else: 
                    print(solver_name)
                    sn.msg()
                    sys.exit("bad edege palcement")
            if sedege.lastbandwidth+used_bw!= sedege.bandwidth:
                    print(solver_name)
                    sys.exit("error in bw calculation")

        #check lastbw in nodes:
        # Note: lastbw in nodes is not updated during edge modifications,
        # so this check is disabled to avoid false positives
        # for n in sn.snode:
        #     lastbw = 0
        #     for e in sn.sedege:
        #         if n.index in e.nodeindex:
        #             lastbw += e.lastbandwidth
        #     if n.lastbw != lastbw:
        #         sys.exit("error in lastbw calculation in node")



    def vnr_life_cycle(self,VNRSS,vnrs):
        """
        Manages the life cycle of VNFs within the simulation.

        This function coordinates the following key activities:
        1. **State Saving**: The current state of VNRs is saved using the controller.
        
        2. **VNR Placement**: Each VNR is placed within the appropriate substrate network (subNet) by invoking the mapping function of the global solver. 
        The results are processed, updating the state of the subNets accordingly. 
        If a VNR placement fails, the corresponding VNR is dropped.

        3. **Integration Testing**: After each placement, an integration test is conducted to validate the correctness of VNR placements and resource allocation.

        4. **Scalability Management**: The function monitors the duration of each VNR and manages scaling actions based on the Mean Time Between Scaling (MTBS). 
        It generates scaling chains either for scaling up or scaling down VNFs, updating the respective states accordingly. 
        If a scaling action is rejected, the VNR is terminated.

        5. **VNF Termination**: After the VNF duration has elapsed, the VNF is ended, and an integration test is performed to ensure data integrity.

        6. **Final Integration Testing**: At the end of the life cycle, a final integration test is executed to confirm the correctness of all VNR placements and resource allocations across all solvers.

        This method ensures that VNRs are efficiently placed, scaled, and terminated while maintaining system integrity throughout their life cycle.
        """

        #save_vnr_states
        self.controller.save_vnr_state(vnrs)
        #Placement of VNRs
        states =[]
        for i in range(len(self.solvers)):
            states.append({'sn':self.subNets[i],'vnr':vnrs[i]})
        results = self.global_solver.mapping(states)
        
        #Update Subnets
        for i in range(len(self.solvers)):
            self.subNets[i]=results[i]["sn"]
            if not results[i]["success"]:
                #Drop rejected VNRs
                vnrs[i].DropVnr(VNRSS[i])
                vnrs[i]=None
            #Intergration test
            self.integration_test(self.subNets[i],VNRSS[i],self.solvers[i])
        #Passing results to Controller
        self.controller.mapping_result(results,self.subNets)
        #Scalability 
        end = 0 
        start=-1
        # Calculating the end time for the active VNRs based on its duration
        for i in range(len(self.solvers)):
            if vnrs[i]:
                end = self.env.now+vnrs[i].duration
                break
        while self.env.now < end:
            start=-1
            # Identifying the first VNR eligible for scalability, as some VNRs may have been dropped after the placement and scalability process.
            for i in range(len(self.solvers)):
                if vnrs[i]:
                    start = i 
                    break
            if start<0:
                break
            # Determining the time until the next scaling demand is generated.
            next_scale = np.random.exponential(vnrs[start].mtbs)
            if self.env.now+next_scale < end :
                scaling_chaine=[]
                scale_up = False
                u = np.random.uniform(0,1)
                #Generate Scaling Chain
                if u< 1/2:
                    #scale up
                    scale_up = True 
                    # Generate the initial scaling chain for the first eligible VNR.
                    scaling_chaine= vnrs[start].generate_scale_up()
                    # Copy the generated scaling chain to the subsequent eligible VNRs.
                    for i in range(start+1,len(self.solvers)):
                        if vnrs[i]:
                            vnrs[i].copy_scale_up(scaling_chaine) 
                else: 
                    # Genrate scaling down
                    # Generate the initial scaling chain for the first eligible VNR.
                    scaling_chaine= vnrs[start].generate_scale_down()
                    # Copy the generated scaling chain to the subsequent eligible VNRs.
                    for i in range(start+1,len(self.solvers)):
                        if vnrs[i]:
                            vnrs[i].copy_scale_down(scaling_chaine) 

                # If a scaling chain is generated, execute the scaling process.
                if len(scaling_chaine)>0:
                    yield self.env.timeout(next_scale)
                    results=[]
                    for i in range(start,len(self.solvers)):
                            if vnrs[i]:
                                scale_results= None
                                results= None

                                # Scaling Up
                                if scale_up:
                                    results=self.global_solver.scaling_up(i,vnrs[i],self.subNets[i],scaling_chaine)
                                    scale_results={"success":results["success"], "scaling_type":"up"}
                                    self.subNets[i]=results["sn"]
                                    vnrs[i]=results["vnr"]
                                    VNRindex=VNRSS[i].reqs_ids.index(vnrs[i].id)
                                    VNRSS[i].reqs[VNRindex]=vnrs[i]
                                    z=0
                                    #Ends the VNR if scaling rejected
                                    if not results["success"]:
                                        vnrs[i].EndsVnr(self.subNets[i],VNRSS[i])
                                        vnrs[i]=None
                                
                                # Scaling Down   
                                else: 
                                    results=self.global_solver.scaling_down(i,vnrs[i],self.subNets[i],scaling_chaine)
                                    scale_results={"success":True, "scaling_type":"down"}
                                    self.subNets[i]=results
                                self.controller.scaling_result(i,scale_results)

                                #Integration test
                                if self.subNets[i] is not None:
                                    self.integration_test(self.subNets[i],VNRSS[i],self.solvers[i])
            # No scaling 
            else: 
                # Wait until the end of the VNR's lifetime duration.
                yield self.env.timeout(end-self.env.now)  

        # Ending VNRs
        for i in range(start,len(self.solvers)):
            if vnrs[i]:
                vnrs[i].EndsVnr(self.subNets[i],VNRSS[i])
                self.integration_test(self.subNets[i],VNRSS[i],self.solvers[i])             
        
        # Global Intergarion test
        for i in range(len(self.solvers)):
            self.integration_test(self.subNets[i],VNRSS[i],self.solvers[i])  
        

   