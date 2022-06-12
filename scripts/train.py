import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from scripts.metrics import accuracy


def train_epoch_backbone(train_loader, model, criterion, optimizer, model_name, cuda, **kwargs):
    lmbd = kwargs.get('lmbd')
    dataset_type = 'coco-2017' if kwargs.get('dataset_type') is None else kwargs.get('dataset_type')
    with tqdm(train_loader, unit="batch") as t_epoch:
        model.train()
        loss_sum = 0
        counter = 1
        for (images, labels) in t_epoch:
            if dataset_type == 'coco-2017':
                images_q = images[0]
                images_k = images[1]
            else:
                images_q = images
                images_k = labels
            if cuda:
                images_q = images_q.cuda(non_blocking=True)
                images_k = images_k.cuda(non_blocking=True)
            t_epoch.set_description(f"Epoch {kwargs['epoch']}")
            if model_name == 'densecl':
                output_g, target_g, output_d, target_d = model(images_q, images_k)
                loss_g = criterion(output_g, target_g)
                loss_d = criterion(output_d, target_d)
                loss = lmbd * loss_g + (1 - lmbd) * loss_d
            else:
                output, target = model(images_q, images_k)  # TODO: it seems like broken
                loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            mean_loss = loss_sum / counter
            counter += 1

            t_epoch.set_postfix(loss=loss.item(), mean_loss=mean_loss)
    return loss


def train_epoch_head(train_loader, model, head, criterion, optimizer, model_name='', **kwargs):
    device = torch.device("cuda:0" if kwargs['cuda'] else "cpu")

    with tqdm(train_loader, unit="batch") as t_epoch:
        model.eval()
        head.train()
        acc_sum = 0
        loss_sum = 0
        counter = 1
        for (images, labels, paths) in t_epoch:
            if kwargs['cuda']:
                if kwargs.get('active_learning') is None:
                    images[0] = images[0].cuda(non_blocking=True)
                    images[1] = images[1].cuda(non_blocking=True)
                else:
                    images = images.cuda(non_blocking=True)
            with torch.no_grad():
                if model_name == 'densecl':
                    x = torch.cat((images[0], images[1]), 0)
                else:
                    if kwargs.get('active_learning') is not None:
                        x = model.encoder_q(images)
                    else:
                        x = torch.cat((model.encoder_q(images[0]), model.encoder_q(images[1])), 0)
            if kwargs.get('active_learning') is not None:
                target = labels
            else:
                target = torch.cat((labels[0], labels[1]), 0)
            (x, target) = (x.to(device), target.to(device))
            t_epoch.set_description(f"Epoch {kwargs['epoch']}")
            output = head(x)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_classes = 19 if kwargs.get('num_classes') is None else kwargs.get('num_classes')
            acc = accuracy(output, target, num_classes)
            acc_sum += acc
            loss_sum += loss.item()
            mean_loss = loss_sum / counter
            mean_acc = acc_sum / counter
            counter += 1
            t_epoch.set_postfix(loss=loss.item(), accuracy=acc, mean_acc=mean_acc, mean_loss=mean_loss)
    return mean_loss, mean_acc


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
        loss, acc = train_epoch_head(train_loader, model, head, criterion, optimizer,
                                model_name=model_name, epoch=epoch, cuda=is_cuda, **kwargs)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        torch.save({
            'epoch': epoch + 1,
            'arch': f"{model_name}_head",
            'loss': loss,
            'state_dict': head.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f=f"{head_checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar")
    writer.flush()

    return head
