#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:01:26 2022

@author: kaouther
"""



import numpy as np

class ReplayBufferDQN(object):
    """Used in the case of DQN"""
    def __init__(self,max_size):   
        self.mem_size = max_size
        """ Maximum memory size allocated to store Deep Reinforcement Learning (DRL) transitions. """
        self.mem_cntr = 0
        """ A counter tracking the total number of transitions stored in memory since the beginning of the simulation. """
        self.memory = [0]*max_size
        """ A fixed-size memory buffer to store transitions, with a maximum capacity defined by max_size. """


        
    def store(self, transition):
        """ Stores a transition in memory. Replaces the oldest entry when memory is full. """
        index =  self.mem_cntr % self.mem_size
        self.memory[index] = transition
        self.mem_cntr += 1
        
    def sample_buffer(self,batch_size):
        """ 
        Samples a batch of transitions from memory. 
        Collects states, actions, rewards, dones, next states, and the length of each sampled transition.
        The transition length is important in the learning process because VNRs vary in the number of VNFs they contain. 
        This results in different numbers of steps required for placement. Additionally, some transitions may involve placement failures, 
        meaning not all VNFs in a VNR are successfully placed.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        indices = np.random.choice(max_mem,batch_size,replace= False)
        states = []
        actions = []
        rewards = []
        dones = [] 
        next_states=[]
        transition_lens =[]
        for idx in indices: 
            
            states.extend(self.memory[idx].states) 
            actions.extend(self.memory[idx].actions) 
            rewards.extend(self.memory[idx].rewards) 
            dones.extend(self.memory[idx].dones)
            next_states.extend(self.memory[idx].next_states)
            transition_lens.append(len(self.memory[idx].dones))
            
        return states, actions,rewards, dones ,next_states,transition_lens


        
class ReplayBuffer(object):
    """Used in the case of A2C"""
    def __init__(self,max_size):   
        self.mem_size = max_size
        """ Maximum memory size allocated to store Deep Reinforcement Learning (DRL) transitions. """
        self.mem_cntr = 0
        """ A counter tracking the total number of transitions stored in memory since the beginning of the simulation. """
        self.memory = [0]*max_size
        """ A fixed-size memory buffer to store transitions, with a maximum capacity defined by max_size. """


        
    def store(self, transition):
        """ Stores a transition in memory. Replaces the oldest entry when memory is full. """
        index =  self.mem_cntr % self.mem_size
        self.memory[index] = transition
        self.mem_cntr += 1
    
    def store_transition(self, transition):
        """ Alias for store() method for compatibility with PPOAgent. """
        self.store(transition)
        
    def sample_buffer(self,batch_size):
        """ 
        Samples a batch of transitions from memory. 
        Collects states, actions, rewards, dones, next values, and the length of each sampled transition.
        The transition length is important in the learning process because VNRs vary in the number of VNFs they contain. 
        This results in different numbers of steps required for placement. Additionally, some transitions may involve placement failures, 
        meaning not all VNFs in a VNR are successfully placed.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        indices = np.random.choice(max_mem,batch_size,replace= False)
        batch = []
        states = []
        actions = []
        rewards = []
        dones = [] 
        next_values=[]
        transition_lens =[]
        for idx in indices: 
            
            states.extend(self.memory[idx].states) 
            actions.extend(self.memory[idx].actions) 
            rewards.extend(self.memory[idx].rewards) 
            dones.extend(self.memory[idx].dones)
            next_values.extend(self.memory[idx].next_value)
            transition_lens.append(len(self.memory[idx].dones))
            
        return states, actions,rewards, dones ,next_values,transition_lens

 

