import os
import sys


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
