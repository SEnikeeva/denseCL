import torchvision.transforms as transforms
import torch.nn as nn
import torch


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU(inplace=True)
    )


class OriginalUNet(nn.Module):
    def __init__(self):
        super(OriginalUNet, self).__init__()
        channels = [64, 128, 256, 512, 1024]
        num_feat = 128
        self.dconv_down1 = double_conv(3, channels[0])
        self.dconv_down2 = double_conv(channels[0], channels[1])
        self.dconv_down3 = double_conv(channels[1], channels[2])
        self.dconv_down4 = double_conv(channels[2], channels[3])
        self.dconv_down5 = double_conv(channels[3], channels[4])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample4 = nn.ConvTranspose2d(channels[-1], num_feat,
                                            kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(num_feat, num_feat,
                                            kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(num_feat, num_feat,
                                            kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(num_feat, num_feat,
                                            kernel_size=2, stride=2)

        self.dconv_up4 = double_conv(num_feat + channels[3], num_feat)
        self.dconv_up3 = double_conv(num_feat + channels[2], num_feat)
        self.dconv_up2 = double_conv(num_feat + channels[1], num_feat)
        self.dconv_up1 = double_conv(num_feat + channels[0], num_feat)

        self.padding = transforms.Pad(padding=92, padding_mode='reflect')

    def forward(self, x):
        x = self.padding(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)
        x = self.upsample4(x)
        y_crop, x_crop = get_crop_size(x, conv4)
        conv4 = conv4[:, :, y_crop:-y_crop, x_crop:-x_crop]
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample3(x)
        y_crop, x_crop = get_crop_size(x, conv3)
        conv3 = conv3[:, :, y_crop:-y_crop, x_crop:-x_crop]
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample2(x)
        y_crop, x_crop = get_crop_size(x, conv2)
        conv2 = conv2[:, :, y_crop:-y_crop, x_crop:-x_crop]
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample1(x)
        y_crop, x_crop = get_crop_size(x, conv1)
        conv1 = conv1[:, :, y_crop:-y_crop, x_crop:-x_crop]
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        return x


def get_crop_size(feat_map1, feat_map2):
    y_crop = (feat_map2.shape[-2] - feat_map1.shape[-2]) // 2
    x_crop = (feat_map2.shape[-1] - feat_map1.shape[-1]) // 2
    return y_crop, x_crop
