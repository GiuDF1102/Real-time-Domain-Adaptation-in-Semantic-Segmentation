#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import numpy as np
from torch import float32, tensor
import random

# Dataset is in data\Cityscapes
class GTA5(Dataset):
    def __init__(self, mode, aug_type=None):
        super(GTA5, self).__init__()
        # Parameters initialization
        self.mode = mode
        self.data_path = 'data/GTA5/'
        self.aug_type = aug_type
        self.aug_transform = None
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        
        # Loading data and labels
        if self.mode == 'train':
            self.data = open(self.data_path + 'train_gta5.txt', 'r').read().splitlines()
            self.labels = open(self.data_path + 'train_gta5.txt', 'r').read().splitlines()
        elif self.mode == 'val':
            self.data = open(self.data_path + 'val_gta5.txt', 'r').read().splitlines()
            self.labels = open(self.data_path + 'val_gta5.txt', 'r').read().splitlines()
        elif self.mode == 'train_full':
            self.data = open(self.data_path + 'full_gta5.txt', 'r').read().splitlines()
            self.labels = open(self.data_path + 'full_gta5.txt', 'r').read().splitlines()

        bright_t = v2.ColorJitter(brightness=[1,2])
        contrast_t = v2.ColorJitter(contrast=[2,5])
        saturation_t = v2.ColorJitter(saturation=[1,3])
        hue_t = v2.ColorJitter(hue=0.2)
        gs_t = v2.RandomGrayscale()
        hflip_t = v2.RandomHorizontalFlip()
        rp_t = v2.RandomPerspective(distortion_scale=0.5)
        blur_t = v2.GaussianBlur(kernel_size=7, sigma=(0.3, 0.7))
        sol_t = v2.RandomSolarize(threshold = 0.4)

        aug_transformations = {
            "CS-HF": v2.Compose([contrast_t, saturation_t, hflip_t]),
            "H-RP": v2.Compose([hue_t, rp_t]),
            "B-GS": v2.Compose([bright_t, gs_t]),
            "HS-HF": v2.Compose([hue_t, saturation_t, hflip_t]),
            "S-BL-HF": v2.Compose([sol_t, blur_t, hflip_t]),
            "B" : v2.Compose([bright_t]),
            "C" : v2.Compose([contrast_t]),
            "S" : v2.Compose([saturation_t]),
            "H" : v2.Compose([hue_t]),
            "GS" : v2.Compose([gs_t]),
            "HF" : v2.Compose([hflip_t]),
            "RP" : v2.Compose([rp_t]),
            "BL" : v2.Compose([blur_t]),
            "SOL" : v2.Compose([sol_t]),
        }

        lab_transformation = {
            "CS-HF": v2.Compose([hflip_t]),
            "H-RP": v2.Compose([rp_t]),
            "B-GS": v2.Compose([]),
            "HS-HF": v2.Compose([hflip_t]),
            "S-BL-HF": v2.Compose([hflip_t]),
            "HF" : v2.Compose([hflip_t]),
            "RP" : v2.Compose([rp_t]),
        }
        
        if self.aug_type is not None:
            self.aug_transform = aug_transformations[self.aug_type]
            self.lab_transform = lab_transformation[self.aug_type]

        # Normalization
        self.transform = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(float32, scale=True), 
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = Image.open('data/GTA5/images/' + self.data[index]).convert('RGB')
        label = Image.open('data/GTA5/labels/' + self.labels[index])

        if self.mode == "train":
            image = image.resize((1280, 720), Image.BILINEAR)
            label = label.resize((1280, 720), Image.NEAREST)

        # Mapping classes and returning label to a tensor object
        label = np.asarray(label, np.float32)
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = tensor(label_copy.copy(), dtype=float32)
        label = np.array(label).astype(np.int32)[np.newaxis, :]

        # Augmentation
        if self.aug_transform is not None and self.mode == 'train_full' and random.random() > 0.5:
            image = self.aug_transform(image)
            image = self.transform(image)
            label = self.lab_transform(label)
        else:
            image = self.transform(image)

        return image, label


