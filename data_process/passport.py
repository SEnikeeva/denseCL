import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PassportDataset(Dataset):
    def __init__(self, data_dir, task='train', labels_dir='', active_learning=False):

        self.XImg_list = []
        self.yLabel_list = []

        self.label_path = os.path.join(os.getcwd(), labels_dir)
        self.rgb_path = os.path.join(os.getcwd(), data_dir)
        self.XImg_list = sorted(os.listdir(self.rgb_path))
        if task == 'train':
            self.images_q = sorted(list(filter(lambda img_path: '_1' in img_path, self.XImg_list)))
            self.images_k = sorted(list(filter(lambda img_path: '_2' in img_path, self.XImg_list)))
        else:
            self.yLabel_list = sorted(os.listdir(self.label_path))
        self.task = task
        self.active_learning = active_learning

    def __len__(self):
        if self.task == 'train':
            length = len(self.images_q)
        else:
            length = len(self.XImg_list)
        return length

    def __getitem__(self, index):

        if self.task != 'train':
            image = Image.open(self.rgb_path + self.XImg_list[index])
            y = Image.open(self.label_path + self.yLabel_list[index])
            image1 = transforms.ToTensor()(image)
            image1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image1)
            y1 = transforms.PILToTensor()(y)
            y1 = torch.squeeze(y1, 0)
            y1 = y1.type(torch.LongTensor)
            if self.active_learning:
                return image1, y1, self.rgb_path + self.XImg_list[index]
            image2 = transforms.RandomHorizontalFlip(1)(image)
            image2 = transforms.ToTensor()(image2)
            image2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image2)
            y2 = transforms.RandomHorizontalFlip(1)(y)
            y2 = transforms.PILToTensor()(y2)
            y2 = torch.squeeze(y2, 0)
            y2 = y2.type(torch.LongTensor)
            return [image1, image2], [y1, y2]
        else:
            image_q = Image.open(self.rgb_path + self.images_q[index])
            image_k = Image.open(self.rgb_path + self.images_k[index])
            image_q = transforms.ToTensor()(image_q)
            image_k = transforms.ToTensor()(image_k)
            image_q = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_q)
            image_k = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_k)
            return [image_q, image_k]
