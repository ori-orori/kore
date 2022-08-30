import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from models.ppo import PPO
from models.encoder import CellFeatureEncoder

def model_build(cfg):
    return UnifiedModel(cfg)

class UnifiedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ppo_model = PPO(cfg)
        self.cnn_encoder = CellFeatureEncoder(cfg)

    def forward(self, state):
        cell_features, scalar_features, self_features = state
        encoded_cell_features = self.cnn_encoder(cell_features)
        unified_features = torch.cat((encoded_cell_features, scalar_features, self_features), dim=1)
        value, action = self.ppo_model(unified_features)
        return value, action
