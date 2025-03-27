import argparse
import os
import scipy.misc
import numpy as np
import sys
from tqdm import tqdm
import torch
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
import time

from datetime import datetime

from DataLoader import pascalVOCLoader

parser = argparse.ArgumentParser(description="Test a trained DeepLabV3 model on Pascal VOC dataset.")
parser.add_argument("--model-name", type=str, required=True, help="Path to the trained model .pt file")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = "/home/josh_reed/Desktop/Reed_Project/Research/VOCdevkit/VOC2012"
test_set = pascalVOCLoader(base_dir, 'val')
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = deeplabv3_resnet50(pretrained=False, progress=False).to(device)
model.load_state_dict(load(f'{base_dir}/trained_models/{args.model_name}', map_location=device))

# Move metric to GPU
metric = JaccardIndex(task="multiclass", num_classes=21, ignore_index=0).to(device)

# Set model to evaluation mode
model.eval()

accuracy = []
total_batches = len(test_loader)

# Start timing
start_time = time.time()

# Testing loop with batch timing
tbar = tqdm(test_loader, desc="Testing")
for i, sample in enumerate(tbar):
    batch_start = time.time()  # Start batch timer

    image, target = sample[0].to(device), sample[1].to(device)

    with no_grad():
        output = model(image)

    pred = argmax(output['out'], dim=1)
    accuracy.append(metric(pred, target).cpu().numpy())

    # Measure batch time
    batch_time = time.time() - batch_start
    tbar.set_description(f"Batch {i+1}/{total_batches} | Time per batch: {batch_time:.3f}s")

# Final timing
total_time = time.time() - start_time
avg_batch_time = total_time / total_batches

# Print results
print(f"\nTotal inference time: {total_time:.2f} seconds")
print(f"Average time per batch: {avg_batch_time:.3f} seconds")

# Compute final IoU scores
IoUs = np.array(accuracy)
Per_class_IoU = np.mean(IoUs, axis=0)
miou = np.mean(Per_class_IoU, axis=0)

# Print results
print("\nTest set results:")
print("Per-class IoU:", Per_class_IoU)
print("Mean IoU:", miou)
