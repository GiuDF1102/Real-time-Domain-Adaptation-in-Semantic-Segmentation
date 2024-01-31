#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import numpy as np
from torch import float32

# Dataset is in data\Cityscapes
class CityScapes(Dataset):
    def __init__(self, mode):
        super(CityScapes, self).__init__()
        # Parameters initialization
        self.mode = mode
        self.data_path = 'data/Cityscapes/'
        # Normalization
        if self.mode != 'fda':
          self.transform = v2.Compose([
              v2.ToImage(), 
              v2.ToDtype(float32, scale=True), 
              v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
          ])
        else: 
          self.transform = v2.Compose([
              v2.ToImage(), 
              v2.ToDtype(float32, scale=True)
          ])

        # Loading data and labels
        if self.mode == 'train' or self.mode=="fda" or self.mode=="ssl":
            self.data = open(self.data_path + 'images/train/train_cs.txt', 'r').read().splitlines()
            self.labels = open(self.data_path + 'gtFine/train/train_gT_cs.txt', 'r').read().splitlines()
        elif self.mode == 'val':
            self.data = open(self.data_path + 'images/val/val_cs.txt', 'r').read().splitlines()
            self.labels = open(self.data_path + 'gtFine/val/val_gT_cs.txt', 'r').read().splitlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path_split = self.data[index].split('\\');
        image_path = "/".join(image_path_split)
        label_path_split = self.labels[index].split('\\');
        label_path = "/".join(label_path_split)
        if self.mode == "fda" or self.mode == "train" or self.mode=="ssl":
            image = Image.open('data/Cityscapes/images/train/' + image_path).convert('RGB')
            label = Image.open('data/Cityscapes/gtFine/train/' + label_path)
        else:
            image = Image.open('data/Cityscapes/images/val/' + image_path).convert('RGB')
            label = Image.open('data/Cityscapes/gtFine/val/' + label_path)

        if self.mode == "train" or self.mode == "fda" or self.mode=="ssl":
            image = image.resize((1024, 512), Image.BILINEAR)
            label = label.resize((1024, 512), Image.NEAREST)

        # Normalization
        image = self.transform(image)
        label = np.array(label).astype(np.int32)[np.newaxis, :]

        if self.mode == 'train' or self.mode == 'train_full' or self.mode=='val':
            return image, label
        elif self.mode == 'ssl':
            return image, label, self.data[index]


# Dataset is in data\Cityscapes
class CityScapesSSL(Dataset):
    def __init__(self, mode):
        super(CityScapesSSL, self).__init__()
        # Parameters initialization
        self.mode = mode
        self.data_path = 'data/Cityscapes/'
        # Normalization
        if self.mode != 'fda':
          self.transform = v2.Compose([
              v2.ToImage(), 
              v2.ToDtype(float32, scale=True), 
              v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
          ])
        else: 
          self.transform = v2.Compose([
              v2.ToImage(), 
              v2.ToDtype(float32, scale=True)
          ])

        # Loading data and labels
        if self.mode == 'train' or self.mode=="fda":
            self.data = open(self.data_path + 'images/train/train_cs.txt', 'r').read().splitlines()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path_split = self.data[index].split('\\');
        image_path = "/".join(image_path_split)
        image_path_split_ssl = self.data[index].split('\\')[-1];
        if self.mode == "fda" or self.mode == "train":
            image = Image.open('data/Cityscapes/images/train/' + image_path).convert('RGB')
            label = Image.open('data/sudo_labels/' + image_path_split_ssl)

        if self.mode == "train" or self.mode == "fda":
            image = image.resize((1024, 512), Image.BILINEAR)

        # Normalization
        image = self.transform(image)
        label = np.array(label).astype(np.int32)[np.newaxis, :]

        if self.mode == 'train' or self.mode == 'train_full' or self.mode=='fda':
            return image, label



