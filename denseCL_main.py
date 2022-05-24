import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.backbone import BackBone
from data_process.augmentation import DataAugmentation, PretrainAugmentation
from models.denseCL import DenseCL
from scripts.utils import clear_out_folder


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
            output_g, target_g, output_d, target_d = model(images[0], images[1])
            print(output_g)
            loss_g = criterion(output_g, target_g)
            loss_d = criterion(output_d, target_d)
            loss = lmbd * loss_g + (1 - lmbd) * loss_d
            print("LOSS: ", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_epoch.set_postfix(loss=loss.item(), accuracy=100)
    return loss


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
    train_set = torchvision.datasets.ImageFolder(root=dataset_path, transform=PretrainAugmentation())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=True)

    backbone_q = BackBone()
    backbone_k = BackBone()

    model = DenseCL(backbone_q, backbone_k, is_cuda=is_cuda)
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
        torch.save({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f=f"{checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main(is_cuda=False)
