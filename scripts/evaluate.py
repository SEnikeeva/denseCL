import os

import torch
import torch.nn as nn
from cityscapesscripts.helpers.labels import id2label as i2l
from tqdm import tqdm

from scripts.metrics import accuracy
from scripts.utils import imshow, labels_show, clear_out_folder


def evaluate(model, head, test_set, epoch_head, epoch_model, model_name='unetcl',
             output_folder='/content/drive/MyDrive/colab/UNetCL/output',
             is_cuda=True, dataset_type='cityscapes_dataset', save_images=False):

    head_checkpoints_folder = f'{output_folder}/{model_name}_head_checkpoints'
    head_checkpoints_path = f"{head_checkpoints_folder}/checkpoint_{epoch_head:04n}.pth.tar"

    checkpoint = torch.load(head_checkpoints_path)
    head.load_state_dict(checkpoint['state_dict'])

    model_checkpoints_folder = f'{output_folder}/{model_name}_checkpoints'
    model_checkpoints_path = f"{model_checkpoints_folder}/checkpoint_{epoch_model:04n}.pth.tar"

    checkpoint = torch.load(model_checkpoints_path)
    model.load_state_dict(checkpoint['state_dict'])

    if is_cuda:
        model.cuda()
        head.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    miou = []
    labels_unique = []

    head.eval()
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_set)):
            images, labels = batch
            if is_cuda:
                images[0] = images[0].cuda(non_blocking=True)
                images[1] = images[1].cuda(non_blocking=True)
            for i in range(len(images)):
                images[i] = torch.unsqueeze(images[i], dim=0)
                if model_name == 'densecl':
                    X = images[i]
                else:
                    X = model.encoder_q(images[i])
                y = labels[i]
                y = torch.unsqueeze(y, dim=0)
                X, y = X.to(device), y.to(device)

                predictions = head(X)
                miou.append(accuracy(predictions, y, num_classes=head.num_classes))

                predictions = nn.functional.softmax(predictions, dim=1)
                pred_labels = torch.argmax(predictions, dim=1)
                pred_labels = pred_labels.int()
                if save_images:
                    pred_labels = pred_labels.to('cpu')
                    labels[i] = labels[i].to('cpu')
                    images[i] = images[i].to('cpu')
                    if dataset_type == 'cityscapes_dataset':
                        # Remapping the labels
                        pred_labels.apply_(lambda x: i2l[x].id)
                    if len(pred_labels.unique()) == 4:
                        pred_type = 'perfect'
                    elif len(pred_labels.unique()) == 3:
                        pred_type = 'good'
                    else:
                        pred_type = 'okay'
                    out_dir = f"{output_folder}/evaluation/{pred_type}/{idx}_{i}"
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    else:
                        clear_out_folder(out_dir)
                    imshow(images[i].squeeze(0), save=True, out_dir=f"{out_dir}/img.jpg")
                    labels_show(labels[i].squeeze(0), save=True, out_dir=f"{out_dir}/ground.png")
                    labels_show(pred_labels.squeeze(0), save=True, out_dir=f"{out_dir}/pred.png")
                    pred_labels = pred_labels.to(device)
                labels_unique.append(pred_labels.unique())
    return miou, labels_unique
