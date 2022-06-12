import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from scripts.train import train_head


def active_learning_iteration(model, head, x_pool, labeled_pixels=None, is_cuda=True, n_instances=10, **kwargs):
    if labeled_pixels is None:
        labeled_pixels = [[] for _ in x_pool]
    model_name = kwargs['model_name']
    X_pool = x_pool
    device = torch.device("cuda:0" if is_cuda else "cpu")

    head.eval()
    model.eval()
    with torch.no_grad():
        to_label_list = []
        for idx, batch in enumerate(tqdm(X_pool)):
            lps = labeled_pixels[idx]
            images, labels, path = batch
            if is_cuda:
                images = images.cuda(non_blocking=True)
            images = torch.unsqueeze(images, dim=0)
            if model_name == 'densecl':
                X = images
            else:
                X = model.encoder_q(images)
            y = labels
            y = torch.unsqueeze(y, dim=0)
            X, y = X.to(device), y.to(device)

            predictions = head(X)
            predictions = nn.functional.softmax(predictions, dim=1)
            predictions = torch.squeeze(predictions)
            pred_labels = torch.max(predictions, dim=0).values
            N = pred_labels.size(0)
            pred_labels = pred_labels.flatten()
            pred_labels, indices = torch.sort(pred_labels)
            for lp in lps:
                indices = indices[indices != lp]
            to_label = indices[:n_instances]
            labeled_pixels[idx].extend(to_label)
            to_label = list(zip((to_label // N).tolist(), (to_label % N).tolist()))
            to_label_list.append(to_label)
    return to_label_list, labeled_pixels


def active_learning_init(model, head, train_set, criterion_head, optimizer_head,
                         output_folder='/content/drive/MyDrive/colab/UNetCL/output',
                         is_cuda=True, **kwargs):
    n_initial = 100
    batch_size = kwargs['batch_size']
    epochs = kwargs['epochs']
    model_name = kwargs['model_name']
    num_classes = kwargs['num_classes']
    initial_idx = np.random.choice(range(len(train_set)), size=n_initial, replace=False)
    X_initial = []
    for idx in initial_idx:
        X_initial.append(train_set[idx])
    X_pool = np.delete(train_set, initial_idx, axis=0)
    initial_train_loader = torch.utils.data.DataLoader(X_initial, batch_size=batch_size, shuffle=True, pin_memory=True)

    head = train_head(head, model, optimizer_head, criterion_head, initial_train_loader, epochs=epochs,
                      is_cuda=is_cuda, model_name=model_name,
                      output_folder=output_folder, num_classes=num_classes,
                      active_learning=kwargs.get('active_learning'))

    return head, X_pool
