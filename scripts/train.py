import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ignite.metrics import ConfusionMatrix, mIoU
from ignite.engine import Engine


def train(train_loader, model, criterion, optimizer, **kwargs):
    with tqdm(train_loader, unit="batch") as t_epoch:
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
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(output, target)
            t_epoch.set_postfix(loss=loss.item(), accuracy=acc)
    return loss


def train_head(train_loader, model, head, criterion, optimizer, **kwargs):

    device = kwargs['device']
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


# TODO: make it exist
def accuracy(output, target):
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)
    cm = ConfusionMatrix(num_classes=19)
    metric = mIoU(cm, ignore_index=0)
    metric.attach(default_evaluator, 'miou')
    state = default_evaluator.run([[output, target]])
    return state.metrics['miou']


def train_backbone(model, train_loader, is_cuda=True):
    checkpoints_folder = '/content/drive/MyDrive/colab/UNetCL/model_checkpoints'
    momentum = 0.9
    weight_decay = 1e-4
    lr = 0.3
    start_epoch = 0
    epochs = 3  # 800

    if is_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    writer = SummaryWriter()

    for epoch in range(start_epoch, epochs):
        loss = train(train_loader, model, criterion, optimizer, epoch=epoch, cuda=False)
        writer.add_scalar("Loss/train", loss, epoch)
        torch.save({
            'epoch': epoch + 1,
            'arch': 'resnet50',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f=f"{checkpoints_folder}/checkpoint_{epoch:04n}.pth.tar")
    writer.flush()

    return model
