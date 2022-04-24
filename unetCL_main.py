import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from backbone import BackBone
from data_service import DataAugmentation
from denseCL import DenseCL
from unet import OriginalUNet
from unetCL import UNetCL
from utils import clear_out_folder


def train(train_loader, model, criterion, optimizer, **kwargs):
    with tqdm(train_loader, unit="batch") as t_epoch:
        lmbd = kwargs['lmbd']
        model.train()
        for (images, _) in t_epoch:
            if kwargs['cuda']:
                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)
            else:
                images[0] = images[0]
                images[1] = images[1]

            t_epoch.set_description(f"Epoch {kwargs['epoch']}")
            output, target = model(images[0], images[1])
            # print('OUTPUT: ', output)
            # print('TARGET: ', target)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(output, target)
            t_epoch.set_postfix(loss=loss.item(), accuracy=acc)
    return loss


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# TODO: add gpu and distr
def main(is_cuda=True):
    checkpoints_folder = 'checkpoints'
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    else:
        clear_out_folder(checkpoints_folder)
    clear_out_folder("checkpoints")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    momentum = 0.9
    weight_decay = 1e-4
    lr = 0.3
    batch_size = 4  # 256
    start_epoch = 0
    epochs = 3  # 800
    lmbd = 0.5

    dataset_path = "~/fiftyone/coco-2017/try/"
    train_set = torchvision.datasets.ImageFolder(root=dataset_path, transform=DataAugmentation())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    backbone_q = OriginalUNet()
    backbone_k = OriginalUNet()

    model = UNetCL(backbone_q, backbone_k, is_cuda=is_cuda)
    print(model)

    if is_cuda:
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    writer = SummaryWriter()

    for epoch in range(start_epoch, epochs):
        loss = train(train_loader, model, criterion, optimizer, lmbd=lmbd, epoch=epoch, cuda=False)
        writer.add_scalar("Loss/train", loss, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=f"{checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar")
    writer.flush()
    writer.close()


# TODO: make it exist
def accuracy(output, target):
    return 100


if __name__ == "__main__":
    main(is_cuda=False)
