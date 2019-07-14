import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork,WeightedLoss
from buffers import ReplayBuffer, PER_ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.5
BETA = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, framework, buffer_type):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.framework = framework
        self.buffer_type = buffer_type

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        # def __init__(self, device, buffer_size, batch_size, alpha, beta):
        if self.buffer_type =='PER_ReplayBuffer':
            self.memory = PER_ReplayBuffer(device, BUFFER_SIZE, BATCH_SIZE, ALPHA, BETA)
        if self.buffer_type == 'ReplayBuffer':
            self.memory = ReplayBuffer(device, action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if self.buffer_type == 'ReplayBuffer':
                    experiences = self.memory.sample()
                    is_weights = None
                    idxs = None
                if self.buffer_type =='PER_ReplayBuffer':
                    experiences, is_weights, idxs = self.memory.sample()
                    self.criterion = WeightedLoss()
                    #print('debugging:', experiences )
                self.learn(experiences, is_weights, idxs, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # use local network

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, is_weights, idxs, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        if self.framework == 'DQN':
        # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) #target network: which is uploaded slower than the local network
            #print('Q_targets_next is ', Q_targets_next)
        if self.framework == "DDQN":
            #print("DDQN")
            max_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            #print('max_actions is ', max_actions)
            #print('self.qnetwork_target(next_states).detach() is ',self.qnetwork_target(next_states).detach())
            #Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, max_actions).squeeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, max_actions)
            #print('DDQN, Q', Q_targets_next)
            #print('rewards is ', rewards)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
      
    #------------------------------------------------------------------------------  
        #Huber Loss provides better results than MSE
        if is_weights is None:
            loss = F.smooth_l1_loss(Q_expected, Q_targets)

        #Compute Huber Loss manually to utilize is_weights with Prioritization
        else:
            loss, td_errors = self.criterion.huber(Q_expected, Q_targets, is_weights)
            self.memory.batch_update(idxs, td_errors)

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
