# Training Script with notes to myself (Maybe I should remove those later...):

# Dr. Ahana gave some code that goes over the training model.
# Note to self: Check Outlook more, and try to remember that Github exists lol.

import argparse
import os
import scipy.misc
import numpy as np
import sys
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
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
import matplotlib.pyplot as plt
import ssl
 
from datetime import datetime
 
from data_loader import pascalVOCLoader

# Parameters (Is there a reason why the parameters are defined outside?)
params = {'batch_size': batch_size , 'shuffle': True, 'num_workers': 1}

# This part I want to analyze later
writer = SummaryWriter('runs/VOCSegment.' +timestamp)

# Defining the Dataloader
training_set = pascalVOCLoader(partition['train'] , labels)
training_gen = DataLoader(training_set , **params)

# Defining the network, optimizer, and loss
model = deeplabv3_resnet50(pretrained: False, progress: False)
optimizer = optim.SGD(model.parameters() , lr = 0.001 , momentum = 0.9)
criterion = nn.CrossEntropyLoss(weight = weights, ignore_index = 255)

#Looping
max_epochs = 10

for epoch in range(max_epochs):
  train_loss = 0.0

  # Training mode: Online.
  model.train()

  # Visuals for iteration
  tbar = tqdm(training_gen)

  for i, sample in enumerate(tbar):
    image, target = sample[0] , sample[1]

    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output['out'] , target)

    # This is where back propagation takes place
    loss.backward()

    # Weights are updated
    optimizer.step()
    train_oss += loss.item()

# This part I just copied
print('[Epoch: %d]' % (epoch))
    	print('Train loss: %.3f' % (train_loss / (i + 1)))
        # svae loss of every epoch in the summary writer object
        writer.add_scalar('training loss',
                            train_loss / (i + 1),
                            epoch)
 	
	# save model weights  
    save(model.state_dict(), f'{base_dir}/trained_models/final.pt-'+timestamp)
