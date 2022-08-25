import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from models.ppo import PPO
from models.cnn import CNNEncoder

def build_model(cfg, args):
    return UnifiedModel(cfg, args)

class UnifiedModel(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        self.ppo_model = PPO(cfg, args)
        self.cnn_encoder = CNNEncoder(cfg, args)

    def forward(self, x):
        cell_features, scalar_features, self_features = x
        encoded_cell_features = self.cnn_encoder(cell_features)
        unified_features = torch.cat((cell_features, scalar_features, self_features), dim=1)
        value, action = self.ppo_model(unified_features)
        return value, action
