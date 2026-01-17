import numpy as np
import os
import json


class Result_saver():
    """ A class used to save results for a specific solver."""
    def __init__(self,solver,max_vnf,result_location,episode_per_file):
        self.result_location=result_location+"/"+solver
        """ The path to the folder where the results for this solver will be saved. """
        self.solver=solver
        """ A string representing the solver's name. """
        self.max_vnf = max_vnf
        """ The maximum number of VNFs allowed in a VNR. """

        self.R2C= []
        """ A list containing the R2C ratio for each placement during a single episode. """
        self.reward =[]
        """ A list containing the Reward for each placement during a single episode. """


        self.nb_requests = 0
        """ The total number of VNR requests that arrived during a single episode. """
        self.nb_accepted_requests =0
        """ The total number of acceptec VNR requests during a single episode. """
        self.nb_iteration = []
        """ A list containing the number of iterations taken to place each arrived VNR. """
        self.nb_rejected_requests =0
        """ The total number of rejected VNR requests during a single episode. """
        self.rejection_cause = {'node': 0, 'edge': 0}
        """ A dictionary that contains the total number of rejections categorized by cause, either due to node or edge resource limitations during a single episode. """
        self.nb_vnfs =[]
        """ A list containing the number of VNFs in each VNR that was successfully placed during a single episode. """
        self.nb_vls =[]
        """ A list containing the number of Virtual Edges in each VNR that was successfully placed during a single episode. """
        self.used_ressources = {'used_nodes': [0], 'used_links': [0],'used_cpu':[0],'used_bw':[0]}
        """
        A dictionary tracking the usage of resources during a single episode:
        - 'used_nodes': List tracking the number of nodes used.
        - 'used_links': List tracking the number of links used.
        - 'used_cpu': List tracking the total CPU used.
        - 'used_bw': List tracking the total bandwidth used.
        """
        self.nb_scaling_requests = {'up': 0, 'down': 0}
        """
        A dictionary tracking the number of scaling requests:
        - 'up': Total number of scale-up requests.
        - 'down': Total number of scale-down requests.
        """
        self.nb_accepted_scaling =0
        """
        The total number of accepted scaling up requests.
        """
        self.nb_rejected_scaling =0
        """
        The total number of rejected scaling up requests.
        """
        self.results_file = self.result_location+"/results.csv"
        """ The path to the Results of each episode"""
        self.placement_file = self.result_location+"/placement.csv"
        """ The path to the Results of each placement"""

        # Used to save sn state and VNF state 
        #---------------------------------------------------------------#
        self.episode_per_file = episode_per_file
        self.sn_state_folder = self.result_location+"/sn_state"
        self.vnr_state_folder = self.result_location+"/vnr_state"
        
        self.episode_counter = 0
        self.file_counter = 0
        #---------------------------------------------------------------#

    def create_directories(self):
        """
        Create folders to save results, ensuring no duplication of folder names to avoid errors or accidental overwriting of results.
        """
        os.makedirs(self.result_location,exist_ok=True)
        os.makedirs(self.sn_state_folder,exist_ok=True)
        os.makedirs(self.vnr_state_folder,exist_ok=True)

    def init_files(self):
        """  Initialize results file with headers"""
        f = open(self.results_file, 'w+')
        f.write("episode,Reward,R2C,nb_requests,nb_accepted_requests,avr_iteration,nb_rejected_requests,node_rejection,link_rejection,nb_vnfs,nb_vls,avr_cpu_used,max_cpu_used,avr_bw_used,max_bw_used,avr_used_nodes,max_used_nodes,avr_used_links,max_used_links,nb_scaling_up,nb_scaling_down,nb_accepted_scaling,nb_rejected_scaling\n")
        f.close()

        f = open(self.placement_file,"w+")
        max_vnfs= ",".join(map(str, range(self.max_vnf)))
        f.write("Episode,Reward,R2C,nb_iter"+","+max_vnfs+"\n")
        f.close()   

            

    def mapping_result(self,success,reward,R2C,cause=None,nb_iter=None):
        """
        This function records the results of a VNR placement attempt, including whether the placement was successful, 
        the associated reward, the R2C value, the cause of rejection (if applicable), and the number 
        of iterations needed for successful placement.

        Parameters:
        - success (bool): Indicates if the VNR placement was successful.
        - reward (float): The reward value associated with the placement.
        - R2C (float): The Revenu to Cost value of the placement.
        - cause (str, optional): Specifies the reason for a rejection, either 'node' or 'edge'. Default is None.
        - nb_iter (int, optional): The number of iterations required for the VNR placement. Only used for successful placements.
        """
        self.nb_requests+=1
        self.R2C.append(R2C)
        self.reward.append(reward)
        if success:
            self.nb_accepted_requests+=1
            if nb_iter:
                self.nb_iteration.append(nb_iter)
        else:
            self.nb_rejected_requests+=1
            if cause =='node':
                self.rejection_cause['node']+=1
            if cause =='edge':
                self.rejection_cause['edge']+=1
        

    def scaling_result(self,scaling_type,success):
        """
        This function records the result of a scaling attempt (either scaling up or scaling down) 
        for a VNR, updating the corresponding scaling statistics.

        Parameters:
        - scaling_type (str): Specifies the type of scaling, either 'up' (scale up) or 'down' (scale down).
        - success (bool): Indicates whether the scaling attempt was successful (True) or rejected (False).
        """
        if scaling_type=='up':
            self.nb_scaling_requests['up']+=1
            if success:
                self.nb_accepted_scaling+=1
            else:
                self.nb_rejected_scaling+=1
        else :
            self.nb_scaling_requests['down']+=1

    def update_used_ressources(self,sn):
        """
        This function updates the statistics of the used resources based on the current state 
        of a substrate network (SN). Used after placement or scalability 

        Parameters:
        - sn: The substrate network object from which resource usage data will be retrieved.
        """
        result = sn.get_used_ressources()
        self.used_ressources['used_nodes'].append(result['used_nodes'])
        self.used_ressources['used_links'].append(result['used_links'])
        self.used_ressources['used_cpu'].append(result['used_cpu'])
        self.used_ressources['used_bw'].append(result['used_bw'])
    
    def nb_deployed_ressources(self,nb_vnfs,nb_vls):
        self.nb_vnfs.append(nb_vnfs)
        self.nb_vls.append(nb_vls)

    def update_episode(self):

        self.episode_counter+=1
        if self.episode_counter % self.episode_per_file == 0:
            self.file_counter+=1

    def var_reset(self):
        """     
        Resets various metrics and statistics for a new episode run.
        This function clears previous results, including rewards, requests, 
        and resource usage, ensuring a clean slate for the next episode.
        """
        self.R2C= []
        self.reward =[]
        
        self.nb_requests = 0
        self.nb_accepted_requests =0
        self.nb_iteration = []
        self.nb_rejected_requests =0
        self.rejection_cause = {'node': 0, 'edge': 0}

        self.nb_vnfs =[]
        self.nb_vls =[]

        self.used_ressources = {'used_nodes': [0], 'used_links': [0],'used_cpu':[0],'used_bw':[0]}
        
        self.nb_scaling_requests = {'up': 0, 'down': 0}
        self.nb_accepted_scaling =0
        self.nb_rejected_scaling =0
    
    def save_placement(self,episode,reward,r2c,nb_iter,mapping):
        f= open(self.placement_file,"a")
        result= ",".join(map(str, mapping))
        f.write(str(episode)+","+str(reward)+","+str(r2c)+","+str(nb_iter)+","+result+"\n")
        f.close()

    def save_vnr_state(self,vnr):
        file=self.vnr_state_folder+"/file"+str(self.file_counter)+".json"
        f = open (file, "a") 
        f.write(json.dumps(vnr.vnr_state())+ '\n')
        f.close()

    def sn_state(self,sn):
        file=self.sn_state_folder+"/file"+str(self.file_counter)+".json"
        f = open (file, "a") 
        f.write(json.dumps(sn.sn_state())+ '\n')
        f.close()

    def save_results(self):
        f = open(self.results_file, 'a')
        row = f"{self.episode_counter},{np.mean(self.reward)},{np.mean(self.R2C)},{self.nb_requests},{self.nb_accepted_requests},{np.mean(self.nb_iteration)},{self.nb_rejected_requests},{self.rejection_cause['node']},{self.rejection_cause['edge']},{np.sum(self.nb_vnfs)},{np.sum(self.nb_vls)},{np.mean(self.used_ressources['used_cpu'])},{np.max(self.used_ressources['used_cpu'])},{np.mean(self.used_ressources['used_bw'])},{np.max(self.used_ressources['used_bw'])},{np.mean(self.used_ressources['used_nodes'])},{np.max(self.used_ressources['used_nodes'])},{np.mean(self.used_ressources['used_links'])},{np.max(self.used_ressources['used_links'])},{self.nb_scaling_requests['up']},{self.nb_scaling_requests['down']},{self.nb_accepted_scaling},{self.nb_rejected_scaling}\n"
        f.write(row)
        f.close()

        





class Global_controller():
    """ 
    A controller that saves results for all solvers.
    
    This class is responsible for managing the results from multiple solvers 
    in the simulation. 
    """
    def __init__(self,solvers,sns,env,episode_duration,results_location,episode_per_file,max_vnfs):
        self.solvers= solvers
        """ List of solvers names """
        self.sns=sns
        """ List of Substrate Network"""
        self.env=env
        """ Simpy env"""
        self.episode_duration=episode_duration
        self.results_location= results_location
        self.result_savers = []
        self.episode_per_file = episode_per_file
        self.episode_start_time = None
        """ Track episode start time for duration calculation """
        self.episode_times = []
        """ Store episode durations for averaging """
        for i in range(len(self.solvers)):
            self.result_savers.append(Result_saver(self.solvers[i],max_vnfs,self.results_location,self.episode_per_file))
        for i in range(len(self.solvers)):
                self.result_savers[i].create_directories()
                self.result_savers[i].init_files()
                


    def simulation_controller(self):
        """
        Controls the simulation episodes.

        This method runs in an infinite loop, yielding the specified episode duration
        for each iteration. At the end of each episode, it saves the results for each 
        solver, updates the episode count, and resets the variables for the next episode.
        """
        import time
        
        while True:
            # Record episode start time
            if self.episode_start_time is None:
                self.episode_start_time = time.time()
            
            yield self.env.timeout(self.episode_duration)
            
            # Calculate episode duration
            episode_elapsed = time.time() - self.episode_start_time
            self.episode_times.append(episode_elapsed)
            avg_episode_time = np.mean(self.episode_times)
            
            # Display episode summary
            print(f"\n{'='*80}")
            print(f"📊 Episode {self.result_savers[0].episode_counter + 1} Complete | Time: {self.env.now}")
            print(f"⏱️  Episode Duration: {episode_elapsed:.2f}s | Avg: {avg_episode_time:.2f}s")
            print(f"{'='*80}")
            
            for i in range(len(self.solvers)):
                rs = self.result_savers[i]
                acceptance_rate = (rs.nb_accepted_requests / rs.nb_requests * 100) if rs.nb_requests > 0 else 0
                avg_reward = np.mean(rs.reward) if len(rs.reward) > 0 else 0
                avg_r2c = np.mean(rs.R2C) if len(rs.R2C) > 0 else 0
                
                print(f"\n🔧 {self.solvers[i]}:")
                print(f"  ├─ Requests: {rs.nb_accepted_requests}/{rs.nb_requests} accepted ({acceptance_rate:.1f}%)")
                print(f"  ├─ Avg Reward: {avg_reward:.3f} | Avg R2C: {avg_r2c:.3f}")
                print(f"  ├─ Rejections: Node={rs.rejection_cause['node']}, Edge={rs.rejection_cause['edge']}")
                if len(rs.used_ressources['used_cpu']) > 0:
                    print(f"  └─ Resources: CPU={np.mean(rs.used_ressources['used_cpu']):.1f}, BW={np.mean(rs.used_ressources['used_bw']):.1f}")
                
                self.result_savers[i].save_results()
                self.result_savers[i].update_episode()
                self.result_savers[i].var_reset()
            
            print(f"{'='*80}\n")
            
            # Reset for next episode
            self.episode_start_time = time.time()

    def mapping_result(self,results,sns):
        """ 
        Updates the results for each solver based on the mapping results.

        This method processes the results of the current episode, mapping them to the
        corresponding solver. It updates various metrics, saves placement details,
        and records the state of the substrate network.
        """
        for i in range(len(self.solvers)):
            self.result_savers[i].mapping_result(results[i]['success'],results[i]['reward'],results[i]['R2C'],results[i]['cause'],results[i]['nb_iter'])
            self.result_savers[i].save_placement(self.result_savers[i].episode_counter,results[i]['reward'],results[i]['R2C'],results[i]['nb_iter'],results[i]['nodemapping'])
            self.result_savers[i].sn_state(sns[i])
            self.result_savers[i].update_used_ressources(sns[i])
            self.result_savers[i].nb_deployed_ressources(results[i]['nb_vnfs'],results[i]['nb_vls'])

    def scaling_result(self,i,results):
        """
        Updates the scaling results for a specific solver based on the scaling operation.

        This method records the outcome of a scaling operation (either scaling up or scaling down)
        """
        self.result_savers[i].scaling_result(results['scaling_type'],results['success'])


    def save_vnr_state(self,vnrs):
        for i in range(len(self.solvers)):
            self.result_savers[i].save_vnr_state(vnrs[i])


            

