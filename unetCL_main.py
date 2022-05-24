import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data_process.augmentation import DataAugmentation, TargetAugmentation
from data_process.cityscapes import CityscapesDataset
from models.head import Head
from models.unet import OriginalUNet
from models.unetCL import UNetCL
from scripts.train import train_backbone, train_epoch_head


def main(trained_backbone=True):
    is_cuda = torch.cuda.is_available()

    lr = 0.001
    batch_size = 4  # 256
    epochs = 25  # 800

    head_checkpoints_folder = '/content/drive/MyDrive/colab/UNetCL/head_checkpoints'
    model_checkpoints_folder = '/content/drive/MyDrive/colab/UNetCL/model_checkpoints'

    dataset_path = '/content/drive/MyDrive/colab/UNetCL/cityscapes_dataset'
    train_set = CityscapesDataset(mode='fine', split='train', transform=DataAugmentation(),
                                  target_transform=TargetAugmentation(), root_dir=dataset_path)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    backbone_q = OriginalUNet()
    backbone_k = OriginalUNet()

    model = UNetCL(backbone_q, backbone_k, is_cuda=is_cuda)
    head = Head(128, 20)

    if is_cuda:
        model.cuda()
        head.cuda()
        criterion_head = nn.CrossEntropyLoss(ignore_index=255).cuda()
    else:
        criterion_head = nn.CrossEntropyLoss(ignore_index=255)

    optimizer_head = torch.optim.Adam(head.parameters(), lr)

    if not trained_backbone:
        model = train_backbone(model, train_loader, is_cuda=is_cuda)
    else:
        torch.save({
            'state_dict': model.state_dict()
        }, f=f"{model_checkpoints_folder}/checkpoint_{0:04n}.pth.tar")

    writer = SummaryWriter(log_dir='/content/drive/MyDrive/colab/UNetCL/runs')
    for epoch in range(epochs):
        loss = train_epoch_head(train_loader, model, head, criterion_head, optimizer_head, epoch=epoch, cuda=is_cuda)
        writer.add_scalar("Loss/train", loss, epoch)
        torch.save({
            'epoch': epoch + 1,
            'loss': loss,
            'state_dict': head.state_dict(),
            'optimizer': optimizer_head.state_dict(),
        }, f=f"{head_checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
