import torch
from torch import nn


def cov5_3x3():
    return nn.Conv2d(512, 512,
                     kernel_size=(3, 3),
                     dilation=(2, 2),
                     stride=(1, 1),
                     padding=(2, 2),
                     bias=False)


class FCN(nn.Module):
    def __init__(self, num_classes=2, resnet=None):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layer4[0].conv2 = cov5_3x3()
        self.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4[1].conv2 = cov5_3x3()
        self.layer4[2].conv2 = cov5_3x3()
        self.decoder = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=3, dilation=6, stride=1, padding=6, bias=False),
                                     nn.BatchNorm2d(256), nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, dilation=6, stride=1, padding=6, bias=False),
                                     nn.BatchNorm2d(256), nn.ReLU(),
                                     nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=False))
        self.score_pool4 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=(1, 1))
        self.upsampling2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upsampling16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, bias=False)

    def forward(self, x):
        input_shape = x.shape
        with torch.no_grad():
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x16 = self.layer3(x)
        x32 = self.layer4(x16)
        x32 = self.decoder(x32)
        x32 = self.upsampling2(x32)
        x16 = self.score_pool4(x16)
        x32 = x32[:, :, 1:1 + x16.size()[2], 1:1 + x16.size()[3]]
        y_pred = x32 + x16
        y_pred = self.upsampling16(y_pred)
        cx = int((y_pred.shape[3] - input_shape[3]) / 2)
        cy = int((y_pred.shape[2] - input_shape[2]) / 2)
        y_pred = y_pred[:, :, cy:cy + input_shape[2], cx:cx + input_shape[3]]

        return y_pred
