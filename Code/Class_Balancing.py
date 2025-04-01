import numpy as np
import torch
from DataLoader import pascalVOCLoader
from torch.utils.data import DataLoader

def compute_class_weights(dataloader, num_classes=21, batch_size=8):
    dataset = pascalVOCLoader(root, split="train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    print("Computing class weights...")

    # Iterate over dataset to compute class frequencies
    for _, mask in dataloader:
        mask = mask.numpy().flatten()  # Flatten mask to 1D array

        mask = mask[mask != 255]  # Ignore void class

        unique, counts = np.unique(mask, return_counts=True)
        pixel_counts[unique] += counts  # Accumulate pixel counts

    # Compute median frequency without zero check
    median_frequency = np.median(pixel_counts)

    # Compute class weights
    class_weights = median_frequency / pixel_counts

    return torch.FloatTensor(class_weights)
