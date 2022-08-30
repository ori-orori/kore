from torch.nn.modules.batchnorm import BatchNorm2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CNN Encoder which get fleet info, shipyard info, and cell kore as input
class CellFeatureEncoder(nn.Module):
  def __init__(self, cfg):
    super(CellFeatureEncoder, self).__init__()
    
    self.height = cfg['cell_size']
    self.width = cfg['cell_size']
    self.channel = cfg['raw_cell_features_dim']
    self.network = nn.Sequential(
      nn.Conv2d(self.channel, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    self.avgpool = nn.AdaptiveAvgPool2d(1)

  def forward(self, cell_state):
    """Forward propagation to embed x

    Args:
      x: input feature

    Returns:
      x: output feature (Embedded)

    """
    x = self.network(cell_state)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x
