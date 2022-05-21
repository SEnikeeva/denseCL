import os
import sys

import numpy as np
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
