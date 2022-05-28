import torch
import torch.nn as nn
import torchvision

from data_process.augmentation import PretrainAugmentation, DataAugmentation, TargetAugmentation
from data_process.cityscapes import CityscapesDataset
from data_process.coco import COCO
from models.backbone import BackBone
from models.denseCL import DenseCL
from models.head import Head
from models.unet import OriginalUNet
from models.unetCL import UNetCL
from scripts.evaluate import evaluate
from scripts.train import train_backbone, train_head
from scripts.load import load_and_train_backbone, load_and_train_head
from scripts.utils import weights_median_cityscapes


def main(task='train', model_name='unetcl', model_type='backbone',
         dataset_type='cityscapes_dataset', batch_size=4, dict_size=6553,
         num_classes=19, epoch_backbone=10, trained_backbone=True, epoch=0):
    """
    :param epoch:
    :param trained_backbone:
    :param epoch_backbone:
    :param num_classes: number of classes in dataset
    :param dict_size:
    :param batch_size:
    :param task: train resume_train evaluate
    :param model_name: unetcl densecl
    :param model_type: backbone head
    :param dataset_type: coco or cityscapes_dataset
    """
    dataset_path = f'/content/drive/MyDrive/colab/UNetCL/{dataset_type}'
    output_folder = '/content/drive/MyDrive/colab/UNetCL/output'
    batch_size = batch_size
    is_cuda = torch.cuda.is_available()
    head = None
    if model_name == 'unetcl':
        backbone_q = OriginalUNet()
        backbone_k = OriginalUNet()
        model = UNetCL(backbone_q, backbone_k, is_cuda=is_cuda, dict_size=dict_size)
        if model_type == 'head':
            head = Head(128, num_classes)
    else:
        backbone_q = BackBone()
        backbone_k = BackBone()
        model = DenseCL(backbone_q, backbone_k, is_cuda=is_cuda, dict_size=dict_size)
        if model_type == 'head':
            head = Head(128, num_classes)

    if 'train' in task:
        if model_type == 'backbone':
            lmbd = 0.5
            momentum = 0.9
            weight_decay = 1e-4
            lr = 0.03
            if dataset_type == 'cityscapes_dataset':
                train_set = CityscapesDataset(mode='fine', split='train', transform=PretrainAugmentation(),
                                              root_dir=dataset_path)
            else:
                train_set = torchvision.datasets.ImageFolder(root=f'{dataset_path}/train/',
                                                             transform=PretrainAugmentation(), )
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

            if task == 'train':
                if is_cuda:
                    model.cuda()
                    criterion = nn.CrossEntropyLoss().cuda()
                else:
                    criterion = nn.CrossEntropyLoss()

                optimizer = torch.optim.SGD(model.parameters(), lr,
                                            momentum=momentum,
                                            weight_decay=weight_decay)
                model = train_backbone(model, optimizer, criterion, train_loader, epochs=10, start_epoch=0,
                                       is_cuda=is_cuda,
                                       model_name=model_name,
                                       output_folder=output_folder, lmbd=lmbd)
            elif task == 'resume_train':
                model = load_and_train_backbone(model, train_loader=train_loader, model_name=model_name,
                                                epoch=epoch, epochs=10 - epoch,
                                                output_folder=output_folder, lmbd=lmbd)
            return model
        else:
            if dataset_type == 'cityscapes_dataset':
                train_set = CityscapesDataset(mode='fine', split='train', transform=DataAugmentation(292),
                                              target_transform=TargetAugmentation(292), root_dir=dataset_path)
            else:
                train_set = COCO(mode='fine', split='train', transform=DataAugmentation(292),
                                 target_transform=TargetAugmentation(292), root_dir=dataset_path)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

            if task == 'train':
                lr = 0.001
                model_checkpoints_folder = f'{output_folder}/{model_name}_checkpoints'
                if not trained_backbone:
                    model = load_and_train_backbone(model, model_name=model_name, epoch=epoch_backbone, trained=True)
                else:
                    torch.save({
                        'state_dict': model.state_dict()
                    }, f=f"{model_checkpoints_folder}/checkpoint_{0:04n}.pth.tar")

                weights_median = weights_median_cityscapes
                if is_cuda:
                    model.cuda()
                    head.cuda()
                    criterion_head = nn.CrossEntropyLoss(ignore_index=255, weight=weights_median.cuda()).cuda()
                else:
                    criterion_head = nn.CrossEntropyLoss(ignore_index=255, weight=weights_median)

                optimizer_head = torch.optim.Adam(head.parameters(), lr)

                head = train_head(head, model, optimizer_head, criterion_head, train_loader, epochs=25,
                                  is_cuda=is_cuda, model_name=model_name,
                                  output_folder=output_folder)
            elif task == 'resume_train':
                head = load_and_train_head(train_loader, head, model, model_name=model_name, epoch=epoch,
                                           epochs=25 - epoch, epoch_backbone=epoch_backbone,
                                           output_folder=output_folder)
            return head
    elif task == 'evaluate':
        if dataset_type == 'cityscapes_dataset':
            test_set = CityscapesDataset(mode='fine', split='val', transform=DataAugmentation(292),
                                         target_transform=TargetAugmentation(292), root_dir=dataset_path)
        else:
            test_set = COCO(mode='fine', split='val', transform=DataAugmentation(292),
                            target_transform=TargetAugmentation(292), root_dir=dataset_path)
        miou, labels_unique = evaluate(model, head, test_set, epoch_head=epoch, epoch_model=epoch_backbone,
                                       model_name=model_name,
                                       output_folder=output_folder, is_cuda=is_cuda)
        return miou, labels_unique
