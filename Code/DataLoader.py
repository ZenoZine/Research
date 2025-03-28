# Training Script with notes to myself (Maybe I should remove those later...):

# Dr. Ahana gave some code that goes over the training model.
# Note to self: Check Outlook more, and try to remember that Github exists lol.

import argparse
import os
import scipy.misc
import numpy as np
import sys
import torch
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet50
from torchmetrics import JaccardIndex
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from torch import argmax
from torch import nn
from torch import optim
from torchsummary import summary
from torch import no_grad
from torch import FloatTensor
from torch import save
from torch import load
from torch import from_numpy
from PIL import Image
import matplotlib.pyplot as plt
import ssl
import random

from datetime import datetime

# from DataLoader import pascalVOCLoader
class pascalVOCLoader(Dataset):
    """
    Data loader for the Pascal VOC semantic segmentation dataset.
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
        print (len(self.files))
        '''

        Split data into training and test sets based on training percentage

        split_idx = int(len(self.files) * self.train_percent)
        if self.split == "train":
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

        print(f"{self.split} set size: {len(self.files)}")

        '''

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
        
        # Code for adding augmentations
        if self.split == 'train':
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

        return im, lbl
    
    
# Parameters (Is there a reason why the parameters are defined outside?)
batch_size = 8
params = {'batch_size': batch_size , 'shuffle': True}

# This part I want to analyze later
timestamp = str(datetime.now().timestamp())
writer = SummaryWriter('runs/VOCSegment.' +timestamp)

base_dir = '/home/josh_reed/Desktop/Reed_Project/Research/VOCdevkit/VOC2012'

# Defining the Dataloader
training_set = pascalVOCLoader(base_dir , 'train')
training_gen = DataLoader(training_set , **params)

# Defining the network, optimizer, and loss
model = deeplabv3_resnet50(pretrained=False, progress=False)
optimizer = optim.SGD(model.parameters() , lr = 0.001 , momentum = 0.9)
# We will use weights later
criterion = nn.CrossEntropyLoss(ignore_index = 255)

#Looping
max_epochs = 1

for epoch in range(max_epochs):
  train_loss = 0.0

  # Training mode: Online.
  model.train()

  # Visuals for iteration
  tbar = tqdm(training_gen)
  print (epoch)
  for i, sample in enumerate(tbar):
    image, target = sample[0] , sample[1]

    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output['out'] , target)

    # This is where back propagation takes place
    loss.backward()

    # Weights are updated
    optimizer.step()
    train_loss += loss.item()

  #Prints out loss after every epoch and saves the loss in SummaryWriter
  ## FIXME: You will get an error due to use of tab and space.
  ## Make sure the next 3 lines have same indentation as line 64 (for i, sample in ....)
  print('[Epoch: %d]' % (epoch))
  print('Train loss: %.3f' % (train_loss / (i + 1)))
  # svae loss of every epoch in the summary writer object
  writer.add_scalar('training loss', train_loss / (i + 1), epoch)

# save model weights
save(model.state_dict(), f'{base_dir}/trained_models/final'+timestamp+'.pt')
