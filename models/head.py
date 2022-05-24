import torch.nn as nn


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.num_classes = out_channels

    def forward(self, x):
        return self.conv1(x)
