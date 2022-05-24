import torch
import torch.nn as nn

from scripts.train import train_backbone, train_head
from scripts.utils import weights_median_cityscapes


def load_and_train_backbone(model, train_loader=None, model_name='unetcl', epoch=0, epochs=10,
                            output_folder='/content/drive/MyDrive/colab/UNetCL/output', trained=False, **kwargs):
    checkpoints_folder = f'{output_folder}/{model_name}_checkpoints'
    checkpoint_path = f"{checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar"

    checkpoint = torch.load(checkpoint_path)
    momentum = 0.9
    weight_decay = 1e-4
    lr = 0.003
    is_cuda = torch.cuda.is_available()
    model.load_state_dict(checkpoint['state_dict'])

    if trained:
        return model

    if is_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
        model.cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    optimizer.load_state_dict(checkpoint['optimizer'])
    return train_backbone(model,
                          optimizer,
                          criterion,
                          train_loader,
                          epochs=epochs,
                          start_epoch=epoch + 1,
                          is_cuda=is_cuda,
                          model_name=model_name,
                          output_folder=output_folder, **kwargs)


def load_and_train_head(train_loader, head, model, model_name='unetcl', epoch=0, epochs=25, epoch_backbone=10,
                        output_folder='/content/drive/MyDrive/colab/UNetCL/output', **kwargs):
    checkpoints_folder = f'{output_folder}/{model_name}_head_checkpoints'
    checkpoint_path = f"{checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar"

    checkpoint = torch.load(checkpoint_path)
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")

    lr = 0.001
    head.load_state_dict(checkpoint['state_dict'])

    weights_median = weights_median_cityscapes
    model = load_and_train_backbone(model, model_name='unetcl', epoch=epoch_backbone, trained=True)
    if is_cuda:
        model.cuda()
        head.cuda()
        criterion_head = nn.CrossEntropyLoss(ignore_index=255, weight=weights_median.cuda()).cuda()
    else:
        criterion_head = nn.CrossEntropyLoss(ignore_index=255, weight=weights_median.cuda())

    optimizer_head = torch.optim.Adam(head.parameters(), lr)
    optimizer_head.load_state_dict(checkpoint['optimizer'])

    return train_head(train_loader,
                      model,
                      head,
                      criterion_head,
                      optimizer_head,
                      epochs=epochs,
                      start_epoch=epoch + 1,
                      is_cuda=is_cuda,
                      model_name=model_name,
                      output_folder=output_folder,
                      device=device, **kwargs)
