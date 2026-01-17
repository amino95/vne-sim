#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:12:48 2022

@author: kaouther
"""

import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from models.GCN import GCNModule
from models.attention import Vnr_AttentionLayer, Sn_AttentionLayer



class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size):
        super(DQN,self).__init__()

        self.layer1 = nn.Linear(num_inputs,hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size)
        self.layer3 = nn.Linear(hidden_size,num_actions)
    
    def forward(self,state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        return self.layer3(state)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()


        self.num_actions = num_actions
        
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)

        self.critic_linear2 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear3 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(num_inputs)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, num_actions)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.ln4 = nn.LayerNorm(num_actions)

    
    def forward(self, state):
        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)
        
        norm_state = self.ln1(state)
        policy_dist = F.tanh(self.actor_linear1(norm_state))
        policy_dist  = self.ln2(policy_dist)
        policy_dist = F.tanh(self.actor_linear2(policy_dist))
        policy_dist = self.ln3(policy_dist)
        policy_dist = F.softmax(self.ln4(self.actor_linear3(policy_dist)), dim=1)
        # Clipping probabilities
        # You can set a threshold below which probabilities will be clipped
        # For example, setting mib=1e-4 will clip all probabilities below 1e-4 to 1e-4
        policy_dist = torch.clamp(policy_dist, min=1e-4)
        return value, policy_dist


class GNNDQN(nn.Module):   

    def __init__(self, num_inputs_sn,num_inputs_vnr, hidden_size,GCN_out,learning_rate,num_actions, device=None):
        super(GNNDQN,self).__init__()
        
        self.learning_rate = learning_rate
        self.device = device if device is not None else torch.device("cpu")

        
        self.gcn_vnr = GCNModule(num_inputs_vnr, hidden_size, GCN_out)
        self.gcn_sn = GCNModule(num_inputs_sn, hidden_size, GCN_out)

        
        self.att_vnr = Vnr_AttentionLayer(GCN_out, torch.tanh)
        self.att_sn =  Sn_AttentionLayer(GCN_out, torch.tanh)


        
        self.actor_critic = DQN(2*GCN_out,num_actions,hidden_size)

        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        self.loss = nn.MSELoss()

        self.to(self.device)


    def forward(self,observation):
        
        # Batch graphs - they're already on device via our cache optimization
        sn_graph = dgl.batch([el.sn.graph for el in observation])
        vnr_graph = dgl.batch([el.vnr.graph for el in observation])
        
        # Ensure graphs are on correct device (avoid redundant .to() calls)
        if sn_graph.device.type != self.device.type:
            sn_graph = sn_graph.to(self.device)
        if vnr_graph.device.type != self.device.type:
            vnr_graph = vnr_graph.to(self.device)
            
        h_sn = sn_graph.ndata['features']
        h_vnr = vnr_graph.ndata['features']
        
        h_sn = self.gcn_sn(sn_graph, h_sn)
        h_vnr = self.gcn_vnr(vnr_graph, h_vnr)

        sn_graph.ndata['h'] = h_sn
        vnr_graph.ndata['h'] = h_vnr

        sn_rep = [el.ndata['h'] for el in dgl.unbatch(sn_graph)]
        vnr_rep = [el.ndata['h'] for el in dgl.unbatch(vnr_graph)]
        
        state = [torch.cat([self.att_sn(sn_rep[i]), self.att_vnr(vnr_rep[i], observation[i].idx)]) for i in range(len(observation))]
        state = torch.cat(state).view(len(observation), -1)

        values = self.actor_critic(state)
        return values

class GNNA2C(nn.Module):
    
    def __init__(self, num_inputs_sn,num_inputs_vnr, hidden_size,GCN_out,learning_rate,num_actions, device=None):
        super(GNNA2C,self).__init__()
        
        self.learning_rate = learning_rate
        self.device = device if device is not None else torch.device("cpu")

        
        self.gcn_vnr = GCNModule(num_inputs_vnr, hidden_size, GCN_out)
        self.gcn_sn = GCNModule(num_inputs_sn, hidden_size, GCN_out)

        
        self.att_vnr = Vnr_AttentionLayer(GCN_out, torch.tanh)
        self.att_sn =  Sn_AttentionLayer(GCN_out, torch.tanh)


        
        self.actor_critic = ActorCritic(2*GCN_out,num_actions,hidden_size)

        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        self.loss = nn.MSELoss()

        self.to(self.device)


    def forward(self,observation):
        
        # Batch graphs - they're already on device via our cache optimization
        sn_graph = dgl.batch([el.sn.graph for el in observation])
        vnr_graph = dgl.batch([el.vnr.graph for el in observation])
        
        # Ensure graphs are on correct device (avoid redundant .to() calls)
        if sn_graph.device.type != self.device.type:
            sn_graph = sn_graph.to(self.device) 
        if vnr_graph.device.type != self.device.type:
            vnr_graph = vnr_graph.to(self.device)
            
        h_sn = sn_graph.ndata['features']
        h_vnr = vnr_graph.ndata['features']
        
        h_sn = self.gcn_sn(sn_graph, h_sn)
        h_vnr = self.gcn_vnr(vnr_graph, h_vnr)

        sn_graph.ndata['h'] = h_sn
        vnr_graph.ndata['h'] = h_vnr

        sn_rep = [el.ndata['h'] for el in dgl.unbatch(sn_graph)]
        vnr_rep = [el.ndata['h'] for el in dgl.unbatch(vnr_graph)]
        
        state = [torch.cat([self.att_sn(sn_rep[i]), self.att_vnr(vnr_rep[i], observation[i].idx)]) for i in range(len(observation))]
        state = torch.cat(state).view(len(observation), -1)

        value, policy_dist = self.actor_critic(state)
        return value, policy_dist
    

        

        
