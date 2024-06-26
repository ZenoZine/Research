import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from torch import from_numpy
import torchvision.transforms.functional as TF

from PIL import Image
from torchvision.transforms.transforms import CenterCrop
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
import random

class pascalVOCLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    """

    def __init__(
        self,
        root,
        split="train",
        img_size=512,
        augmentations=None,
        train_percent = 0.8,
    ):
        self.root = root
        self.split = split
        self.n_classes = 21
        self.train_percent = train_percent
        

        path = os.path.join(self.root, "ImageSets/Segmentation", split + ".txt")
        data_list = open(path, "r")
        file_list = data_list.readlines()
        self.files = [file_list[i].strip() for i in range(0,len(file_list))]

        random.shuffle(self.files)
        
        # Split data into training and test sets based on training percentage
        split_idx = int(len(self.files) * self.train_percent)
        if self.split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]
            
        print(f"{self.split} set size: {len(self.files)}")
            
        self.image_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(512),
                # ToTensor will convert pixel values to 0 to 1 range
                # So, mean and std, need to be in that range
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        
        self.lbl_tf= transforms.Compose(
            [
                transforms.CenterCrop(512),
            ]
        )


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        im_name = self.files[index]
        im_path = os.path.join(self.root, "JPEGImages", im_name + ".jpg")
        lbl_path = os.path.join(self.root, "SegmentationClass", im_name + ".png")
        im = Image.open(im_path)
        lbl =  np.array(Image.open(lbl_path))
        lbl[lbl == 255] = 0
        # Converting to tensor without scaling to a range of 0 to 1, which is automatic in ToTensor()
        lbl = from_numpy(lbl).long()

        # You can use these two lines to check that your labels are in the expected range
        # print((lbl >= 0).all())
        # print((lbl <= 20).all())
        
        im = self.image_tf(im)
        lbl = self.lbl_tf(lbl)
        '''
        if self.split == 'train' and augmentations is not None:
            lbl = torch.unsqueeze(lbl, 0)
            #lbl = lbl.view([1, 512, 512, 1])
            if random.randint(0, 2) > 0:
                im = TF.hflip(im)
                lbl = TF.hflip(lbl)
            if random.randint(0, 2) > 0:
                angle = random.randint(0, 4) * 90
                im = TF.rotate(im, angle)
                lbl = TF.rotate(lbl, angle)
            lbl = torch.squeeze(lbl, 0)
        '''
        return im, lbl
