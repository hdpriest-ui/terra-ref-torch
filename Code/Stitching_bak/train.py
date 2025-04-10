import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Absolute imports
from dataset import StitchingDataset
from models.StitchingModel import StitchingModel
from utils import save_checkpoint, load_checkpoint, create_output_dirs
from constant import const

def main():
    # Initialize directories

    print(f"summary dir is: {const.SUMMARY_DIR}")
    create_output_dirs(const.OUTPUT_ROOT)

    # Initialize device
    device = torch.device(f"cuda:{const.GPU}" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloaders
    train_dataset = StitchingDataset(const.TRAINING_DATA_DIRECTORY, resize_height=const.HEIGHT, resize_width=const.WIDTH)
    train_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=3)

    # Initialize model, loss, and optimizer
    model = StitchingModel.StitchingModel(height=const.HEIGHT, width=const.WIDTH).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE)

    # Load checkpoint if specified
    if const.CHECKPOINT:
        load_checkpoint(const.CHECKPOINT, model, optimizer)

    # Training loop
    best_loss = float("inf")
    for epoch in range(const.ITERATION):
        model.train()
        epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            input1, input2 = inputs
            input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)

            # Forward pass
            outputs, coarse_stitched = model(input1, input2)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if const.GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), const.GRADIENT_CLIP)
            optimizer.step()

            epoch_loss += loss.item()

            # Log progress every 1000 iterations
            if (batch_idx + 1) % 1000 == 0:
                print(f"Epoch [{epoch + 1}/{const.ITERATION}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{const.ITERATION}] Average Loss: {epoch_loss:.4f}")

        # Save the model if it has the best loss so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch, best_loss)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()