import argparse
import os
import numpy as np
import sys
import torch
from torchmetrics import JaccardIndex
from torch import no_grad
from torch import argmax
from torch import load
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import time
from datetime import datetime
from DataLoader import pascalVOCLoader

# Argument parser
parser = argparse.ArgumentParser(description="Test a trained DeepLabV3 model on Pascal VOC dataset.")
parser.add_argument("--model-name", type=str, required=True, help="Path to the trained model .pt file")
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and data loading
base_dir = "/home/josh_reed/Desktop/Reed_Project/Research/VOCdevkit/VOC2012"
test_set = pascalVOCLoader(base_dir, 'val')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = deeplabv3_resnet50(pretrained=False, progress=False).to(device)
print(f"Loading model from {base_dir}/trained_models/{args.model_name}")
model.load_state_dict(load(f'{base_dir}/trained_models/{args.model_name}', map_location=device))
print("Model loaded successfully.")

# Setup IoU metric
metric = JaccardIndex(task="multiclass", num_classes=21, ignore_index=0).to(device)

# Set model to evaluation mode
model.eval()

# Initialize variables
accuracy = []
total_batches = len(test_loader)

# Start timing
start_time = time.time()

# Testing loop
tbar = tqdm(test_loader, desc="Testing")
for i, sample in enumerate(tbar):
    batch_start = time.time()  # Start batch timer

    image, target = sample[0].to(device), sample[1].to(device)

    with no_grad():
        output = model(image)

    pred = argmax(output['out'], dim=1)

    batch_per_class_iou = metric(pred, target).cpu().numpy()

    accuracy.append(batch_per_class_iou)

    # Measure batch time
    batch_time = time.time() - batch_start
    tbar.set_description(f"Batch {i+1}/{total_batches} | Time per batch: {batch_time:.3f}s")

# Final time calculation
accuracy = np.array(accuracy)
total_time = time.time() - start_time
avg_batch_time = total_time / total_batches

# Print results
print(f"\nTotal inference time: {total_time:.2f} seconds")
print(f"Average time per batch: {avg_batch_time:.3f} seconds")

# Compute final IoU scores
Per_class_IoU = np.mean(accuracy, axis=0)
miou = np.mean(Per_class_IoU)

# Print per-class and mean IoU
print("\nTest set results:")
print("Per-class IoU:", Per_class_IoU)
print("Mean IoU:", miou)

# Save the results to a text file
results_dir = "test_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results = {
    "Per-class IoU": Per_class_IoU,
    "Mean IoU": miou,
    "Total Inference Time": total_time,
    "Average Time per Batch": avg_batch_time
}

# Set up the timestamps for the results file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = os.path.join(results_dir, f"results_{timestamp}.txt")

# Save the results to the file
with open(results_file, "w") as f:
    f.write(f"Results for model: {args.model_name}\n\n")
    for key, value in results.items():
        f.write(f"{key}: {value}\n")

print(f"Results saved to {results_file}")

# Plotting the per-class IoU as a bar graph and saving it as an image
classes = np.arange(0, 21)  # 21 classes in Pascal VOC dataset
plt.figure(figsize=(10, 6))

# Create the bar graph
plt.bar(classes, Per_class_IoU, color='skyblue')

# Labeling the axes and the title
plt.xlabel('Class')
plt.ylabel('IoU')
plt.title('Per-class IoU for the model')

# Set x-axis ticks to show class indices
plt.xticks(classes)

# Display grid for better visibility
plt.grid(True, linestyle='--', alpha=0.6)

# Make the layout tight so labels aren't cut off
plt.tight_layout()

# Save the plot as an image
graph_filename = f'{base_dir}/trained_models/per_class_iou_{args.model_name}.png'
plt.savefig(graph_filename)
print(f"Graph saved to {graph_filename}")

# Close the plot to avoid memory buildup
plt.close()
