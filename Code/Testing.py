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

test_set = pascalVOCLoader(base_dir, 'test')
test_loader = DataLoader(test_set, batch_size=1,
                shuffle=False, num_workers=1)
   # Define netwoirk, optimizer and loss
model = deeplabv3_resnet50(pretrained=False, progress=False)
    
metric = JaccardIndex(num_classes=21, ignore_index=0, reduction='none')
model.load_state_dict(load(f'{base_dir}/trained_models/final.pt-1649029005.91684'))

model.eval()
accuracy = []
tbar = tqdm(test_loader)
for i, sample in enumerate(tbar):
        image, target = sample[0], sample[1]
        
        with no_grad():
            output = model(image)
        pred = output['out'].cpu()
        target = target.cpu()
        pred = argmax(pred, dim=1)
        accuracy.append(metric(pred, target).numpy())

    
IoUs = np.array(accuracy)
Per_class_IoU = np.mean(IoUs, axis=0)
miou = np.mean(Per_class_IoU, axis=0)
print("\nTest set result")
print(per_class_IoU, mIoU)
