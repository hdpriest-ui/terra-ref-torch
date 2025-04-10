import os
import sys
import torch
from torch.utils.data import DataLoader
# Absolute imports
from dataset import StitchingDataset
from models.StitchingModel import StitchingModel
from utils import load_checkpoint, save_heatmap
import constant

# Initialize device
device = torch.device(f"cuda:{constant.GPU}" if torch.cuda.is_available() else "cpu")

# Initialize dataset and dataloader
test_dataset = StitchingDataset(constant.TEST_DATA_DIRECTORY, resize_height=constant.HEIGHT, resize_width=constant.WIDTH)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3)

# Initialize model
model = StitchingModel(height=constant.HEIGHT, width=constant.WIDTH).to(device)

# Load checkpoint
if constant.CHECKPOINT:
    load_checkpoint(constant.CHECKPOINT, model)

# Inference loop
model.eval()
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        input1, input2 = inputs
        input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)

        # Forward pass
        outputs, coarse_stitched = model(input1, input2)

        # Save outputs and heatmaps
        save_heatmap(outputs, targets, idx, constant.OUTPUT_ROOT)