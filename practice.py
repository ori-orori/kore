import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from models.ppo import PPO
import yaml
from models.encoder import CellFeatureEncoder
from models.model_factory import build_model


# rnn = nn.GRU(input_size=5, hidden_size=3, num_layers=2, batch_first=True, bias=False)
# input = torch.randn(2, 5, 5)
# h0 = torch.randn(2, 1, 3)
# h1 = torch.randn(2, 1, 3)
# hidden = torch.cat((h0, h1), dim=1)
# print(hidden.shape)
# output, hn = rnn(input, hidden)
# print(output, hn)

# std = torch.full(size=(3,2), fill_value=0.5)
# print(std)

# action1_2_dist = Normal(torch.tensor([[[0.1], [0.2]], [[-2.0], [3.8]], [[1.0], [0.5]]]), std)
# action1_2_dist = Normal(torch.tensor([[0.1, 0.2], [-2.0, 3.8], [1.0, 0.5]]), std)
# action1_2 = action1_2_dist.sample()
# print(action1_2_dist.mean)
# print(action1_2)
# action1_2_log_prob = action1_2_dist.log_prob(action1_2)
# print(action1_2_log_prob)

# dist = Categorical(torch.tensor([[0.5, 0.5], [0.1, 0.9], [0.8, 0.2]]))
# action = dist.sample()
# print(action)
# print(dist.log_prob(action))

# batch_size = 3
# max_action_len = 10
# rnn_input_dim = 6
# batch_action = torch.tensor([[0.1, 1, 0.3, 2, 0.5], [-0.3, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])

# def create_input_action(batch_action):
#         rnn_input = torch.zeros(batch_size, max_action_len, rnn_input_dim)
#         for i, action in enumerate(batch_action):
#             if action[0] > 0: # action type is launch fleet
#                 rnn_input[i][0][5] = 1.0
#                 for j, char in enumerate(action[1:]):
#                     if j % 2 == 0: # direction
#                         rnn_input[i][j+1][int(char)] = 1.0
#                     else: # step
#                         rnn_input[i][j+1][:] = char
#         return rnn_input

# print(create_input_action(batch_action))

# print(torch.ones((batch_size, 1), dtype=torch.int))
# print(F.one_hot(torch.ones((batch_size, 1), dtype=torch.int), num_classes=5))

# a = torch.full((3, 1), 5).squeeze(1)
# a = torch.full((2, 3), 3.141592)
# print(a)
# print(F.one_hot(a, num_classes=6))

# a = torch.randn((3, 10))
# b = torch.tensor([1, 3, 5, 7, 9])
# c = torch.randn((3, 5))
# print(torch.arange(0, 10, 2))
# print(torch.arange(1, 10, 2))
# print(a)
# print(torch.select(a, 1, b))
# print(c)
# a[:, b] = c
# print(a)

with open('./config/model_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
# print(cfg['self_features_dim'])
ppo = PPO(cfg)


# print(data)
# a = ppo.actor.actor1_1(data)
# print(a)

batch_size=1


# init_hidden1 = ppo.actor.encoder(torch.randn((batch_size,137)).unsqueeze(0)) # 1,1,16
# init_hidden2 = torch.zeros(1, batch_size, 16) # 1,1,16
# action2_hidden = torch.cat((init_hidden1, init_hidden2), dim=0) # 2, 1, 16
# print(action2_hidden.shape)
# rnn_input = F.one_hot(torch.full((1, 1,), 6-1), num_classes=6).to(torch.float) # 1, 1, 6
# print(rnn_input.shape)
# action2_output, action2_hidden = ppo.actor.actor2(rnn_input, action2_hidden)
# print(action2_output)
# print(action2_hidden)

# batch_state = torch.randn((1, 64))
# a = ppo.actor(batch_state)
# print(a)

encoder = CellFeatureEncoder(cfg)
model = build_model(cfg)
cell_state = torch.randn((1, 6, 21, 21))
scalar_state = torch.randn((1, 3))
self_state = torch.randn((1, 4))

unified_state = [cell_state, scalar_state, self_state]

output = model(unified_state)
print(output)

