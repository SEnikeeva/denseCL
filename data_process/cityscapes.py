import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, mode='fine', transform=None, target_transform=None, is_eval=False):
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'fine':
            self.mode = 'gtFine'
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = is_eval

        self.label_path = os.path.join(os.getcwd(), root_dir + '/' + self.mode + '/' + self.split)
        self.rgb_path = os.path.join(os.getcwd(), root_dir + '/leftImg8bit/' + self.split)
        city_list = os.listdir(self.rgb_path)
        # pdb.set_trace()
        for city in city_list:
            temp = os.listdir(self.rgb_path + '/' + city)
            list_items = temp.copy()

            # defining paths
            images_list = ['/' + city + '/' + path for path in list_items]
            labels_list = [li.replace('leftImg8bit', 'gtFine_labelIds') for li in images_list]

            self.yLabel_list.extend(labels_list)
            self.XImg_list.extend(images_list)

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):

        mapping = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
                   7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
                   16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
                   27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
                   }

        def encode_labels(mask):
            label_mask = np.zeros_like(mask)
            for k in mapping:
                label_mask[mask == k] = mapping[k]
            return label_mask

        image = Image.open(self.rgb_path + self.XImg_list[index])
        if self.target_transform is not None:
            y = Image.open(self.label_path + self.yLabel_list[index])
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)
            torch.manual_seed(seed)
            image1 = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            y1 = self.target_transform(y)

            seed2 = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed2)
            torch.manual_seed(seed2)
            image2 = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            y2 = self.target_transform(y)

            image1 = transforms.ToTensor()(image1)
            y1 = np.array(y1)
            y1 = encode_labels(y1)
            y1 = torch.from_numpy(y1)
            y1 = y1.type(torch.LongTensor)

            image2 = transforms.ToTensor()(image2)
            y2 = np.array(y2)
            y2 = encode_labels(y2)
            y2 = torch.from_numpy(y2)
            y2 = y2.type(torch.LongTensor)

            if self.eval:
                return [image1, image2], [y1, y2], self.XImg_list[index]
            else:
                return [image1, image2], [y1, y2]
        else:
            image1, image2 = self.transform(image)
            return [image1, image2]
