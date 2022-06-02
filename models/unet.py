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


class UNet5(nn.Module):
    def __init__(self):
        super(UNet5, self).__init__()
        channels = [64, 128, 256, 512, 1024]
        num_feat = 128
        self.dconv_down1 = double_conv(3, channels[0])
        self.dconv_down2 = double_conv(channels[0], channels[1])
        self.dconv_down3 = double_conv(channels[1], channels[2])
        self.dconv_down4 = double_conv(channels[2], channels[3])
        self.dconv_down5 = double_conv(channels[3], channels[4])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample4 = nn.ConvTranspose2d(channels[-1], channels[-2],
                                            kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(channels[-2], channels[-3],
                                            kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(channels[-3], channels[-4],
                                            kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(num_feat, num_feat,
                                            kernel_size=2, stride=2)

        self.dconv_up4 = double_conv(channels[-2] + channels[3], channels[-2])
        self.dconv_up3 = double_conv(channels[-3] + channels[2], channels[-3])
        self.dconv_up2 = double_conv(channels[-4] + channels[1], channels[-4])
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


def contract_block(in_channels, out_channels, kernel_size, padding):
    contract = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    return contract


def expand_block(in_channels, out_channels, kernel_size, padding):
    expand = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(),
                           nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                              output_padding=1)
                           )
    return expand


class UNet3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = contract_block(in_channels, 32, 7, 3)
        self.conv2 = contract_block(32, 64, 3, 1)
        self.conv3 = contract_block(64, out_channels, 3, 1)

        self.upconv3 = expand_block(out_channels, 64, 3, 1)
        self.upconv2 = expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = expand_block(32 * 2, out_channels, 3, 1)

    def __call__(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv3 = transforms.CenterCrop(conv2.size(-1))(upconv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv2 = transforms.CenterCrop(conv1.size(-1))(upconv2)

        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1
