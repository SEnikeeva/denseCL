import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck


class BackBone(ResNet):

    # TODO: constants to arguments
    def __init__(self):
        super(BackBone, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))
        self.dense_head = nn.Sequential(
            nn.Conv2d(2048, 2048, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 128, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)

        x = self.avgpool(feat)
        x = torch.flatten(x, 1)
        global_out = self.fc(x)
        dense_out = self.dense_head(feat)

        return global_out, dense_out, feat
