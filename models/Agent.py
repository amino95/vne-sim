#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:55:48 2022

@author: kaouther
"""
from models.A2C import GNNA2C
import numpy as np
import torch
from models.replayBuffer import ReplayBuffer
from copy import deepcopy as dc

class Transition(object):
    """ 
    A class to represent a transition in the context of VNR placement.

    A transition captures a sequence of steps taken during the placement of 
    a Virtual Network Request (VNR). It stores the states, actions, 
    rewards, and whether the episode has ended at each step.
    """
    def __init__(self):
        self.states= []
        """ A list of states encountered during the transition. """
        self.actions = []
        """ A list of actions taken at each state. """
        self.rewards= []
        """ A list of rewards received after each action. """
        self.dones = []
        """ A list indicating whether each step was terminal. """
        self.next_value =None
        """ The value of the next state after the placement or the failure (After the last step) """

        
    def store_step(self,state,action,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
   
    
    

class Agent(object):
    
    def __init__(self,gamma,learning_rate,epsilon,memory_size,batch_size,num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,eps_min = 0.01, eps_dec = 5.5e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        """ Discount Factor """
        self.epsilon = epsilon
        """
        Used to balance exploration and exploitation in reinforcement learning.
        """
        self.num_actions=num_actions
        """ The number of actions, representing the total number of nodes in the substrate network (SN). """
        self.memory = ReplayBuffer(memory_size)
        """ A replay buffer for storing transitions. """
        self.batch_size = batch_size
        """ The size of the batch of transitions used during learning. """
        self.memory_init = 3000 
        """ The number of transitions to store before starting the learning process; this value must always be greater than the batch size. """
        self.eps_min = eps_min
        """ The minimum value for the exploration rate (epsilon) during training. """
        self.eps_dec = eps_dec
        """ The amount by which the exploration rate (epsilon) decreases with each step in the learning process. """
        self.learn_step_counter = 0
        self.A2C=GNNA2C(num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate,num_actions, device=self.device)
        
        # Mixed precision training
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 4
        self.accumulation_counter = 0
        
        # Compile model for faster execution (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                self.A2C.model = torch.compile(self.A2C.model, mode='reduce-overhead')
        except Exception:
            pass  # Fallback if compilation not supported
 
        
    def store_transition(self,transition):
        self.memory.store(transition)
        
    def choose_action(self,observation,vnf_cpu,sn_cpu):
            """
            Chooses an action based on the current observation using an epsilon-greedy strategy.

            Args:
                observation: An Observation object that contains the current state of the system.
                vnf_cpu: The CPU required by the VNF being placed.
                sn_cpu: A list containing the available CPU resources of each node in the substrate network.

            Returns:
                action: The chosen action (node index) for placing the VNF.
                value: The estimated value of the current state from the A2C model.
                log_prob: The logarithm of the probability of the chosen action.
            """
            # Get the illegal actions from the observation's node mapping
            illegal_actions=dc(observation.node_mapping)
            # Get the estimated value and policy distribution from the A2C model
            value,policy_dist= self.A2C([observation])
            # Clone the probability distribution for manipulation
            probs = policy_dist[0].clone()
 
            # Epsilon-greedy action selection
            if  np.random.random() < self.epsilon:
                # Select legal actions based on available CPU resources
                legal_actions = np.array([index for index, element in enumerate(sn_cpu) if element > vnf_cpu])
                legal_actions=legal_actions[~np.isin(legal_actions,illegal_actions)]
                
                # Randomly choose an action from the legal actions if available
                if len(legal_actions)>0:
                    action = np.random.choice(legal_actions)
                else: 
                    action = -1  # No legal actions available

            else:
                # Set the probabilities of illegal actions to negative infinity to avoid selection
                probs[illegal_actions] = -float('Inf')
                action = torch.argmax(probs).item() # Choose the action with the highest probability

            # Calculate the log probability of the chosen action
            log_prob = torch.log(probs[action])

            return action, value, log_prob
            

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        


    def learn(self):
        # Return early if we haven't collected enough experiences
        if self.memory.mem_cntr < self.memory_init:
            return
        
        # Sample a batch of experiences from memory
        states, actions,rewards, dones, next_values,transition_lens = self.memory.sample_buffer(self.batch_size)
        
        # Use automatic mixed precision for faster training
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Get predicted values and policy distributions from the A2C model
            values, policy_dists = self.A2C(states)

            # Compute log probabilities of the actions taken using log_softmax for numerical stability
            log_probs = torch.stack([torch.log(policy_dist[actions[i]].clamp(min=1e-8)) for i, policy_dist in enumerate(policy_dists)])
            
            # Initialize target Q-value from the last next value
            q_val = next_values[len(next_values)-1]
            if isinstance(q_val, torch.Tensor):
                q_val = q_val.detach().to(self.device).squeeze()
            else:
                q_val = torch.tensor(q_val, device=self.device, dtype=torch.float32).squeeze()

            lenght = len(rewards) 
            q_vals = torch.zeros(lenght, device=self.device)
            # Pre-create tensors outside loop to avoid repeated allocations
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
              
            # Optimized: target values are calculated backward with episode boundary handling
            j = len(next_values) - 1
            k = 0
            for i in range(lenght - 1, -1, -1):
                reward = rewards_t[i]
                done = dones_t[i]
                q_val = reward + self.gamma * q_val * (1.0 - done)
                q_vals[i] = q_val
                k += 1
                if k == transition_lens[j]:
                    j -= 1
                    k = 0
                    if j >= 0:
                        q_val = next_values[j]
                        if isinstance(q_val, torch.Tensor):
                            q_val = q_val.detach().to(self.device).squeeze()
                        else:
                            q_val = torch.tensor(q_val, device=self.device, dtype=torch.float32).squeeze()

            advantage = q_vals - values.squeeze()
            critic_loss = advantage.pow(2).mean()
            actor_loss = (-log_probs * advantage.detach()).mean()
            actor_critic_loss = (critic_loss + actor_loss) / self.gradient_accumulation_steps

        # Gradient accumulation: only update weights every N steps
        if self.use_amp:
            self.scaler.scale(actor_critic_loss).backward()
        else:
            actor_critic_loss.backward()
        
        self.accumulation_counter += 1
        
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            if self.use_amp:
                self.scaler.step(self.A2C.optimizer)
                self.scaler.update()
            else:
                self.A2C.optimizer.step()
            
            self.A2C.optimizer.zero_grad()
            self.accumulation_counter = 0
            
        self.decrement_epsilon()