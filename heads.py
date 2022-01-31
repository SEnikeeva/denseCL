import torch.nn as nn


class GlobalHead(nn.Module):
    def __init__(self, out_channels=128):
        super(GlobalHead, self).__init__()

        self.out_channels = out_channels

    def forward(self, x):
        x = nn.Sequential()

