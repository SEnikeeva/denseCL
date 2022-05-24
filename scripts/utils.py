import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
import zipfile


def replace_slash(file_path):
    platform = sys.platform
    slash_map = {'win32': '\\',
                 'cygwin': '\\',
                 'darwin': '/',
                 'linux2': '/'}
    if platform not in slash_map.keys():
        platform = 'linux2'
    return file_path.replace('\\', slash_map[platform])


def clear_out_folder(output_folder):
    files = os.listdir(output_folder)
    for f_ in files:
        path_dir = replace_slash(output_folder + "\\" + f_)
        os.remove(path_dir)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def labels_show(labels):
    plt.matshow(labels)
    plt.colorbar()
    plt.show()


def extract_zip(zip_pah, directory_to_extract_to):
    with zipfile.ZipFile(zip_pah, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


weights_median_cityscapes = torch.tensor([0.0238, 0.1442, 0.0384, 1.3385, 1.0000, 0.7148, 4.2218, 1.5914, 0.0551,
                                          0.7577, 0.2183, 0.7197, 6.4924, 0.1254, 3.2801, 3.7300, 3.7667, 8.8921,
                                          2.1195])
