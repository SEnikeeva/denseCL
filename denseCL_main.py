import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from backbone import BackBone
from data_service import DataAugmentation
from denseCL import DenseCL


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def train(train_loader, model, criterion, optimizer, **kwargs):
    with tqdm(train_loader, unit="batch") as t_epoch:
        lmbd = kwargs['lmbd']
        model.train()
        for (images, _) in t_epoch:
            t_epoch.set_description(f"Epoch {kwargs['epoch']}")
            output_g, target_g, output_d, target_d = model(images[0], images[1])
            loss_g = criterion(output_g, target_g)
            loss_d = criterion(output_d, target_d)
            loss = lmbd * loss_g + (1 - lmbd) * loss_d

            loss.backward()
            optimizer.step()

            accuracy = 100
            t_epoch.set_postfix(loss=loss.item(), accuracy=accuracy)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# TODO: add gpu and distr
def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    momentum = 0.9
    weight_decay = 1e-4
    lr = 0.3
    batch_size = 4  # 256
    start_epoch = 0
    epochs = 3  # 800
    lmbd = 0.5

    dataset_path = "~/fiftyone/coco-2017/validation/"
    train_set = torchvision.datasets.ImageFolder(root=dataset_path, transform=DataAugmentation())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    backbone_q = BackBone()
    backbone_k = BackBone()

    model = DenseCL(backbone_q, backbone_k)
    print(model)

    criterion = nn.CrossEntropyLoss()
    # .cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    for epoch in range(start_epoch, epochs):
        train(train_loader, model, criterion, optimizer, lmbd=lmbd, epoch=epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


if __name__ == "__main__":
    main()
