import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scripts.metrics import accuracy


def train_epoch_backbone(train_loader, model, criterion, optimizer, model_name, **kwargs):
    lmbd = kwargs.get('lmbd')
    with tqdm(train_loader, unit="batch") as t_epoch:
        model.train()
        for (images, _) in t_epoch:
            if kwargs['cuda']:
                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)
            t_epoch.set_description(f"Epoch {kwargs['epoch']}")
            if model_name == 'unetcl':
                output, target = model(images[0], images[1])
                loss = criterion(output, target)
            else:
                output_g, target_g, output_d, target_d = model(images[0], images[1])
                loss_g = criterion(output_g, target_g)
                loss_d = criterion(output_d, target_d)
                loss = lmbd * loss_g + (1 - lmbd) * loss_d
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_epoch.set_postfix(loss=loss.item(), accuracy=100)
    return loss


def train_epoch_head(train_loader, model, head, criterion, optimizer, model_name='', **kwargs):
    device = torch.device("cuda:0" if kwargs['cuda'] else "cpu")

    with tqdm(train_loader, unit="batch") as t_epoch:
        model.eval()
        head.train()
        for (images, labels) in t_epoch:
            if kwargs['cuda']:
                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)
            with torch.no_grad():
                x = torch.cat((model.encoder_q(images[0]), model.encoder_q(images[1])), 0)
            target = torch.cat((labels[0], labels[1]), 0)
            (x, target) = (x.to(device), target.to(device))
            t_epoch.set_description(f"Epoch {kwargs['epoch']}")
            output = head(x)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(output, target)
            t_epoch.set_postfix(loss=loss.item(), accuracy=acc)
    return loss


def train_backbone(model, optimizer, criterion, train_loader, epochs=10, start_epoch=0, is_cuda=True,
                   model_name='unetcl',
                   output_folder='/content/drive/MyDrive/colab/UNetCL/output', **kwargs):
    checkpoints_folder = f'{output_folder}/{model_name}_checkpoints'
    writer = SummaryWriter(log_dir=f'{output_folder}/{model_name}_runs')
    for epoch in range(start_epoch, epochs):
        loss = train_epoch_backbone(train_loader, model, criterion, optimizer, model_name, epoch=epoch, cuda=is_cuda,
                                    **kwargs)
        writer.add_scalar("Loss/train", loss, epoch)
        torch.save({
            'epoch': epoch + 1,
            'arch': model_name,
            'loss': loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f=f"{checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar")
    writer.flush()

    return model


def train_head(head, model, optimizer, criterion, train_loader, epochs=10,
               start_epoch=0, is_cuda=True, model_name='unetcl',
               output_folder='/content/drive/MyDrive/colab/UNetCL/output', **kwargs):
    head_checkpoints_folder = f'{output_folder}/{model_name}_head_checkpoints'
    writer = SummaryWriter(log_dir=f'{output_folder}/{model_name}_head_runs')
    for epoch in range(start_epoch, epochs):
        loss = train_epoch_head(train_loader, model, head, criterion, optimizer,
                                model_name=model_name, epoch=epoch, cuda=is_cuda, **kwargs)
        writer.add_scalar("Loss/train", loss, epoch)
        torch.save({
            'epoch': epoch + 1,
            'arch': f"{model_name}_head",
            'loss': loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f=f"{head_checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar")
    writer.flush()

    return model
