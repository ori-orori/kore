from torch.nn.modules.batchnorm import BatchNorm2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CNN Encoder which get fleet info, shipyard info, and cell kore as input
class VGGNet(nn.Module):
  def __init__(self, channel=6, height=21, width=21, init_weights=True):
    super(VGGNet, self).__init__()
    
    self.height = height
    self.width = width
    self.channel = channel
    self.network = nn.Sequential(
      nn.Conv2d(channel, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
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
      nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    )
    self.avgpool = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    """Forward propagation to embed x

    Args:
      x: input feature

    Returns:
      x: output feature (Embedded)

    """
    x = self.network(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'channel': self.channel,
        'height': self.height,
        'width': self.width,
    })
    return config

  def call(self, spatial_feature):
    spatial_feature_encoded = self.network(spatial_feature)

    return spatial_feature_encoded

model = VGGNet()
input = torch.randn(1, 6, 21, 21)
print(model(input).size())
