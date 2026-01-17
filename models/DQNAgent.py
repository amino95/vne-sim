from models.A2C import GNNDQN
import numpy as np
import torch
import torch.nn as nn
from models.replayBuffer import ReplayBufferDQN as ReplayBuffer
from copy import deepcopy as dc


class DQNTransition(object):
    
    def __init__(self):
        self.states= []
        self.actions = []
        self.rewards= []
        self.dones = []
        self.next_states =[]

        
    def store_step(self,state,action,reward,done,next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)


class DQNAgent(object):
    
    def __init__(self,gamma,learning_rate,epsilon,memory_size,batch_size,num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,eps_min = 0.01, eps_dec = 5.5e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = 0.01
        self.num_actions=num_actions

        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.memory_init = 3000
        self.eps_min = eps_min
        self.eps_dec = eps_dec
   
        self.policy_net=GNNDQN(num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate,num_actions, device=self.device)
        self.target_net=GNNDQN(num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate,num_actions, device=self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
 
        


        
    def store_transition(self,transition):
        self.memory.store(transition)
        
    def choose_action(self,observation,vnf_cpu,sn_cpu):
   
            illegal_actions=dc(observation.node_mapping)
            

            if  np.random.random() < self.epsilon:
                legal_actions = np.array([index for index, element in enumerate(sn_cpu) if element > vnf_cpu])
                legal_actions=legal_actions[~np.isin(legal_actions,illegal_actions)]
                if len(legal_actions)>0:
                    action = np.random.choice(legal_actions)
                else: 
                    action = -1

            else:
                with torch.no_grad():
                    values= self.policy_net([observation])
  
                values[0][illegal_actions] = -float('Inf')
                action = torch.argmax(values[0]).item()
                    

            return action
            

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        


    def learn(self):
        torch.autograd.set_detect_anomaly(True)
        if self.memory.mem_cntr < self.memory_init:
            return
        states, actions,rewards, dones ,next_states,transition_lens = self.memory.sample_buffer(self.batch_size)
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.long)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.bool)

        state_action_values = self.policy_net(states).gather(1, action_batch.unsqueeze(1))


     
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma)*(~done_batch) + reward_batch
        loss =  nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
 
     

        # Optimize the model
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        #torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.policy_net.optimizer.step()
        self.decrement_epsilon()
        self.soft_update(self.policy_net, self.target_net)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)



