import torch
import torch.nn as nn
from cityscapesscripts.helpers.labels import id2label as i2l
from tqdm import tqdm

from scripts.metrics import accuracy


def evaluate(model, head, test_set, epoch_head, epoch_model, model_name='unetcl',
             output_folder='/content/drive/MyDrive/colab/UNetCL/output', is_cuda=True):

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
                X = model.encoder_q(images[i])
                y = labels[i]
                y = torch.unsqueeze(y, dim=0)
                X, y = X.to(device), y.to(device)

                predictions = head(X)
                miou.append(accuracy(predictions, y, num_classes=head.num_classes))

                predictions = nn.functional.softmax(predictions, dim=1)
                pred_labels = torch.argmax(predictions, dim=1)
                pred_labels = pred_labels.int()
                # Remapping the labels
                pred_labels = pred_labels.to('cpu')
                pred_labels.apply_(lambda x: i2l[x].id)
                pred_labels = pred_labels.to(device)
                labels_unique.append(pred_labels.unique())
    return miou, labels_unique
