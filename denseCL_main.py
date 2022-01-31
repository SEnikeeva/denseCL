import torch
import torchvision

from data_service import DataAugmentation
from backbone import BackBone, ResNet, Bottleneck
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn


# functions to show an image
from denseCL import DenseCL


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(train_loader, model):
    model.train()
    for i, (images, _) in enumerate(train_loader):
        output, target = model(in_q=images[0], in_k=images[1])


if __name__ == "__main__":
    dataset_path = "~/fiftyone/coco-2017/validation/"
    train_set = torchvision.datasets.ImageFolder(root=dataset_path, transform=DataAugmentation())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone_dq = BackBone(dense=True)
    backbone_dk = BackBone(dense=True)

    backbone_gq = BackBone()
    backbone_gk = BackBone()

    model = DenseCL(backbone_gq, backbone_gk, backbone_dq, backbone_dk)
    model2 = ResNet(Bottleneck, [3, 4, 6, 3],)
    train(train_loader, model)




