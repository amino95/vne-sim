#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 3 2026

PPO (Proximal Policy Optimization) Agent for VNR Placement
"""
from models.A2C import GNNA2C
import numpy as np
import torch
from models.replayBuffer import ReplayBuffer
from copy import deepcopy as dc


class PPOAgent(object):
    """
    Proximal Policy Optimization Agent for VNR placement.
    
    PPO is an on-policy algorithm that uses a clipped surrogate objective
    to prevent large policy updates, making training more stable than vanilla
    policy gradient methods.
    """
    
    def __init__(self, gamma, learning_rate, epsilon, memory_size, batch_size, 
                 num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,
                 eps_min=0.01, eps_dec=5.5e-5, clip_ratio=0.2, ppo_epochs=4, 
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        """ Discount Factor """
        
        self.clip_ratio = clip_ratio
        """ PPO clipping parameter (typically 0.2) """
        
        self.ppo_epochs = ppo_epochs
        """ Number of epochs to train on each batch """
        
        self.entropy_coef = entropy_coef
        """ Entropy bonus coefficient for exploration """
        
        self.value_loss_coef = value_loss_coef
        """ Value function loss coefficient """
        
        self.max_grad_norm = max_grad_norm
        """ Maximum gradient norm for clipping """
        
        self.memory_init = batch_size
        """ Minimum memory size before learning starts """
        
        self.batch_size = batch_size
        """ Batch size for training """
        
        self.epsilon = epsilon
        """ Epsilon for epsilon-greedy exploration """
        
        self.eps_min = eps_min
        """ Minimum epsilon value """
        
        self.eps_dec = eps_dec
        """ Epsilon decay rate """
        
        # Performance optimizations
        self.use_amp = torch.cuda.is_available()
        """ Use automatic mixed precision for faster training """
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        """ Gradient scaler for mixed precision training """
        
        self.gradient_accumulation_steps = 4
        """ Number of steps to accumulate gradients """
        
        self.accumulation_counter = 0
        """ Counter for gradient accumulation """
        
        # Initialize the GNN-based Actor-Critic network
        self.A2C = GNNA2C(num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, 
                         learning_rate, num_actions, device=self.device)
        
        # Try to compile model for better performance (PyTorch 2.0+)
        try:
            self.A2C.model = torch.compile(self.A2C.model, mode='reduce-overhead')
        except Exception:
            pass  # torch.compile not available, continue without it
        
        # Replay buffer for storing transitions
        self.memory = ReplayBuffer(memory_size)

    def store_transition(self, transition):
        """
        Store a transition in the replay buffer.
        
        Args:
            transition: A Transition object containing states, actions, rewards, dones
        """
        self.memory.store_transition(transition)

    def choose_action(self, observation, vnf_cpu, sn_cpu):
        """
        Choose an action using the current policy with epsilon-greedy exploration.
        
        Args:
            observation: The current observation (state)
            vnf_cpu: The CPU required by the VNF being placed
            sn_cpu: A list containing the available CPU resources of each node
        
        Returns:
            action: The chosen action (node index)
            value: The estimated value of the current state
            log_prob: The logarithm of the probability of the chosen action
        """
        # Get the illegal actions from the observation's node mapping
        illegal_actions = dc(observation.node_mapping)
        
        # Get the estimated value and policy distribution from the A2C model
        value, policy_dist = self.A2C([observation])
        
        # Clone the probability distribution for manipulation
        probs = policy_dist[0].clone()
 
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Select legal actions based on available CPU resources
            legal_actions = np.array([index for index, element in enumerate(sn_cpu) 
                                     if element > vnf_cpu])
            legal_actions = legal_actions[~np.isin(legal_actions, illegal_actions)]
            
            # Randomly choose an action from the legal actions if available
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else: 
                action = -1  # No legal actions available
        else:
            # Set the probabilities of illegal actions to negative infinity
            probs[illegal_actions] = -float('Inf')
            action = torch.argmax(probs).item()

        # Calculate the log probability of the chosen action
        log_prob = torch.log(probs[action].clamp(min=1e-8))

        return action, value, log_prob

    def decrement_epsilon(self):
        """Decay epsilon for exploration"""
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def compute_gae(self, rewards, values, dones, next_values, transition_lens):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: Predicted values
            dones: Terminal flags
            next_values: Values of next states
            transition_lens: Length of each transition
        
        Returns:
            advantages: Computed advantages
            returns: Target returns
        """
        advantages = []
        gae = 0
        
        # Convert to tensors
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        
        j = len(next_values) - 1
        k = 0
        next_val = next_values[j]
        if isinstance(next_val, torch.Tensor):
            next_val = next_val.detach().to(self.device).squeeze()
        else:
            next_val = torch.tensor(next_val, device=self.device, dtype=torch.float32).squeeze()
        
        # Compute advantages backward
        for i in range(len(rewards) - 1, -1, -1):
            reward = rewards_t[i]
            done = dones_t[i]
            value = values[i].detach()
            
            delta = reward + self.gamma * next_val * (1.0 - done) - value
            gae = delta + self.gamma * 0.95 * (1.0 - done) * gae  # lambda = 0.95 for GAE
            advantages.insert(0, gae)
            
            next_val = value
            k += 1
            
            # Check episode boundary
            if k == transition_lens[j]:
                j -= 1
                k = 0
                if j >= 0:
                    next_val = next_values[j]
                    if isinstance(next_val, torch.Tensor):
                        next_val = next_val.detach().to(self.device).squeeze()
                    else:
                        next_val = torch.tensor(next_val, device=self.device, dtype=torch.float32).squeeze()
                gae = 0
        
        advantages = torch.stack(advantages)
        returns = advantages + values.squeeze().detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def learn(self):
        """
        Train the PPO agent using collected experiences.
        Uses multiple epochs and clipped surrogate objective.
        """
        # Return early if we haven't collected enough experiences
        if self.memory.mem_cntr < self.memory_init:
            return
        
        # Sample a batch of experiences from memory
        states, actions, rewards, dones, next_values, transition_lens = self.memory.sample_buffer(self.batch_size)
        
        # Use automatic mixed precision for the initial forward pass
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Get predicted values and policy distributions
            values, policy_dists = self.A2C(states)
            
            # Compute old log probabilities
            old_log_probs = torch.stack([torch.log(policy_dist[actions[i]].clamp(min=1e-8)) 
                                         for i, policy_dist in enumerate(policy_dists)])
        
        # Compute advantages using GAE (outside autocast for numerical stability)
        advantages, returns = self.compute_gae(rewards, values, dones, next_values, transition_lens)
        
        # PPO training loop - multiple epochs on the same batch
        for epoch in range(self.ppo_epochs):
            # Use automatic mixed precision for forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Get new predictions
                new_values, new_policy_dists = self.A2C(states)
                
                # Compute new log probabilities
                new_log_probs = torch.stack([torch.log(new_policy_dist[actions[i]].clamp(min=1e-8)) 
                                             for i, new_policy_dist in enumerate(new_policy_dists)])
                
                # Compute probability ratio
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                
                # Compute surrogate losses
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                
                # Policy loss (PPO clipped objective)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((new_values.squeeze() - returns) ** 2).mean()
                
                # Entropy bonus for exploration
                entropy = -(new_policy_dists * torch.log(new_policy_dists.clamp(min=1e-8))).sum(dim=1).mean()
                
                # Total loss with gradient accumulation scaling
                loss = (actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy) / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            self.accumulation_counter += 1
            
            # Update weights only after accumulating gradients
            if self.accumulation_counter >= self.gradient_accumulation_steps:
                if self.use_amp:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.A2C.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.A2C.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    self.scaler.step(self.A2C.optimizer)
                    self.scaler.update()
                else:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.A2C.parameters(), self.max_grad_norm)
                    self.A2C.optimizer.step()
                
                self.A2C.optimizer.zero_grad()
                self.accumulation_counter = 0
        
        # Decay epsilon
        self.decrement_epsilon()
