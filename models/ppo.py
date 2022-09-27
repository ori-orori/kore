import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.unified_features_dim = cfg['unified_features_dim']
        self.encoded_input_dim = cfg['encoded_input_dim']
        self.encoder = nn.Sequential(
                        nn.Linear(self.unified_features_dim, self.encoded_input_dim)
                        )

    def forward(self, unified_features):
        return self.encoder(unified_features)


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoded_input_dim = cfg['encoded_input_dim']
        self.critic = nn.Sequential(
                    nn.Linear(self.encoded_input_dim, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)
                )
    
    def forward(self, batch_state):
        return self.critic(batch_state)

class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg['encoded_input_dim']
        self.rnn_input_dim = cfg['rnn_input_dim']
        self.rnn_hidden_dim = cfg['rnn_hidden_dim']
        self.rnn_num_layers = cfg['rnn_num_layers']
        self.action1_1_dim = cfg['action1_1_dim']
        self.action1_2_dim = cfg['action1_2_dim']
        self.action2_1_dim = cfg['action2_1_dim']
        self.action2_2_dim = cfg['action2_2_dim']
        self.max_action_len = cfg['max_action_len']

        self.encoder = nn.Linear(self.input_dim+2, self.rnn_hidden_dim)
        self.actor1_1 = nn.Linear(self.input_dim, 3)
        self.actor1_2 = nn.Linear(self.input_dim+1, 1)
        self.actor2 = nn.GRU(input_size=self.rnn_input_dim, hidden_size=self.rnn_hidden_dim, num_layers=self.rnn_num_layers, bias=False, batch_first=True)
        self.actor2_1 = nn.Linear(self.rnn_hidden_dim, self.action2_1_dim)
        self.actor2_2 = nn.Linear(self.rnn_hidden_dim, self.action2_2_dim)

    def forward(self, batch_state, batch_action=None, evaluate=False):
        self.batch_size = batch_state.size(0)
        
        action_output = [] # for train
        action_logprob = torch.zeros((self.batch_size, 1))

        action1_1_output = self.actor1_1(batch_state)
        action_output.append(action1_1_output)
        action1_1_prob = F.softmax(action1_1_output, dim=1)
        action1_1_dist = Categorical(action1_1_prob)
        if not evaluate: # supervised learning
            action1_1 = action1_1_dist.sample().to(torch.float).unsqueeze(0)
            action1_1_logprob = action1_1_dist.log_prob(action1_1)
        else:
            action1_1 = batch_action[:, 0]
            action1_1_logprob = action1_1_dist.log_prob(action1_1)
        action_logprob += action1_1_logprob

        action1_2_mean = self.actor1_2(torch.cat((batch_state, action1_1), dim=1))
        action_output.append(action1_2_mean)
        std = torch.full(size=(self.batch_size, self.action1_2_dim), fill_value=0.1)
        action1_2_dist = Normal(action1_2_mean, std)
        if not evaluate: # supervised learning
            action1_2 = action1_2_dist.sample()
            action1_2_logprob = action1_2_dist.log_prob(action1_2)
        else:
            action1_2 = batch_action[:, 1]
            action1_2_logprob = action1_2_dist.log_prob(action1_2)
        action_logprob += action1_2_logprob

        action1 = torch.cat((action1_1, action1_2), dim=1)

        rnn_input, action2_hidden = self.create_input_action(batch_state, batch_action, action1)

        if batch_action is None: # reinforcement learning 
            action2 = None
            action_len = 2
            while action_len < self.max_action_len:
                action2_output, action2_hidden = self.actor2(rnn_input, action2_hidden)
                action2_output = action2_output.squeeze(0)
                if action_len % 2 == 0: # direction
                    action2_1_output = self.actor2_1(action2_output)
                    
                    action2_1_prob = F.softmax(action2_1_output, dim=1)
                    action2_1_dist = Categorical(action2_1_prob)
                    action2_1 = action2_1_dist.sample().to(torch.float).unsqueeze(0)
                    action2_1_logprob = action2_1_dist.log_prob(action2_1)
                    action_logprob += action2_1_logprob
                    if action2 is None: # first prediction
                        action2 = action2_1
                    else:
                        action2 = torch.cat((action2, action2_1), dim=1)

                    if action2_1.detach().item() == 5:
                        break
                    
                else: # step
                    action2_2_mean = self.actor2_2(action2_output)
                    
                    std = torch.full(size=(self.batch_size, 1), fill_value=0.1)
                    action2_2_dist = Normal(action2_2_mean, std)
                    action2_2 = action2_2_dist.sample()
                    action2_2_logprob = action2_2_dist.log_prob(action2_2)
                    action_logprob += action2_2_logprob
                    action2 = torch.cat((action2, action2_2), dim=1)
                action_len += 1
        else: # supervised learning and evaluation
            action2_output, action2_hidden = self.actor2(rnn_input, action2_hidden) # [1, 10, 16], [2, 1, 16]
            action2_1_index = torch.arange(0, self.max_action_len-2, 2)
            action2_1_output = self.actor2_1(action2_output[:, action2_1_index]) # 1, 5, 6
            action_output.append(action2_1_output)
            action2_1_prob = F.softmax(action2_1_output, dim=2)
            action2_1_dist = Categorical(action2_1_prob)
            if not evaluate: # supervised learning
                action2_1 = action2_1_dist.sample().to(torch.float)
                action2_1_logprob = action2_1_dist.log_prob(action2_1)
            else: # evaluation
                action2_1 = batch_action[:, action2_1_index]
                action2_1_logprob = action2_1_dist.log_prob(action2_1)
            # action_logprob += action2_1_logprob

            action2_2_index = torch.arange(1, self.max_action_len-2, 2)
            action2_2_mean = self.actor2_2(action2_output[:, action2_2_index])
            action_output.append(action2_2_mean)
            std = torch.full(size=(self.batch_size, 1), fill_value=0.1)
            action2_2_dist = Normal(action2_2_mean, std)
            if not evaluate: # supervised learning
                action2_2 = action2_2_dist.sample()
                action2_2_logprob = action2_2_dist.log_prob(action2_2)
            else: # evaluation
                action2_2 = batch_action[:, action2_2_index]
                action2_2_logprob = action2_2_dist.log_prob(action2_2)
            # action_logprob += action2_2_logprob

            action2 = torch.zeros((self.batch_size, self.max_action_len-2))
            action2[:, action2_1_index] = action2_1
            action2[:, action2_2_index] = action2_2.squeeze(dim=2)

        action = torch.cat((action1, action2), dim=1)
        return action, action_logprob, action_output


    def create_input_action(self, batch_state, batch_action, action1):
        if batch_action is None:
            init_hidden1 = self.encoder((torch.cat((batch_state, action1), dim=1)).unsqueeze(0))
            rnn_input = F.one_hot(torch.full((self.batch_size, 1), self.rnn_input_dim-1), num_classes=self.rnn_input_dim).to(torch.float)
        else:
            init_hidden1 = self.encoder((torch.cat((batch_state, batch_action[:,:2]), dim=1)).unsqueeze(0))
            rnn_input = torch.zeros(self.batch_size, self.max_action_len-1, self.rnn_input_dim)
            for i, action in enumerate(batch_action):
                if action[0] > 0: # action type is launch fleet
                    rnn_input[i][0][5] = 1.0
                    for j, char in enumerate(action[2:]):
                        if j % 2 == 0: # direction
                            rnn_input[i][j+1][int(char)] = 1.0
                        else: # step
                            rnn_input[i][j+1][:] = char
        init_hidden2 = torch.zeros(self.rnn_num_layers-1, self.batch_size, self.rnn_hidden_dim)
        init_hidden = torch.cat((init_hidden1, init_hidden2), dim=0)
        return rnn_input, init_hidden

class PPO(nn.Module):
    """
        This is the PPO class we will use as our model in main.py
    """
    def __init__(self, cfg):
        """Initializes the PPO model.
            Args:
                input_dim: Input dimension entering the model
				action_dim: Action dimension
            Returns:
                None
        """
        # Extract environment information
        super().__init__()
        self.env = None

        # Initialize actor and critic networks
        self.encoder = Encoder(cfg)
        self.actor = Actor(cfg)
        self.critic = Critic(cfg)

    def forward(self, state, batch_action=None):
        encoded_state = self.encoder(state)
        value = self.critic(encoded_state)
        if  batch_action is None:
            action, action_logprob, action_output = self.actor(encoded_state)
        else:
            action, action_logprob, action_output = self.actor(encoded_state, batch_action)
        
        return value, action, action_output
        

    def get_action(self, state):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                state - the state at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        encoded_state = self.encoder(state)
        action, action_logprob = self.actor(encoded_state)
        # dist = Categorical(action_probs)

        # action = dist.sample()
        # action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, batch_state, batch_action):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_state - the state from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of state)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        encoded_state = self.encoder(batch_state)
        value = self.critic(encoded_state).squeeze()

        action, action_logprobs = self.actor(encoded_state, batch_action)
        # dist = Categorical(action_probs)
        # log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return value, action_logprobs