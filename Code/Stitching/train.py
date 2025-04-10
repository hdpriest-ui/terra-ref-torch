import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# Absolute imports
from dataset import StitchingDataset
from models.StitchingModel import StitchingModel
from utils import save_checkpoint, load_checkpoint, create_output_dirs
from constant import const
from stitch_logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def get_device():
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_device('cuda')
        the_device = torch.device('cuda')
        return the_device
    else:
        the_device = torch.device('cpu')
        return the_device

def save_images_to_tensorboard(writer, images, tag, step):
    """
    Save a batch of images to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        images (torch.Tensor): Batch of images to save.
        tag (str): Tag for the images in TensorBoard.
        step (int): Current step or epoch.
    """
    grid = make_grid(images, normalize=True, scale_each=True)
    writer.add_image(tag, grid, step)

def main():
    logger.info("Starting training process...")
    logger.debug(f"Summary directory: {const.SUMMARY_DIR}")
    create_output_dirs(const.OUTPUT_ROOT)
    checkpoint_dir = os.path.join(const.OUTPUT_ROOT, "checkpoints")

    device = torch.device(f"cuda:{const.GPU}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    target_height, target_width = 304, 304  # Hard-coded dimensions
    train_dataset = StitchingDataset(const.TRAINING_DATA_DIRECTORY, resize_height=target_height, resize_width=target_width)
    val_dataset = StitchingDataset(const.TEST_DATA_DIRECTORY, resize_height=target_height, resize_width=target_width)
    train_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE, generator=torch.Generator(get_device()), shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=const.BATCH_SIZE, generator=torch.Generator(get_device()), shuffle=False, num_workers=3)
    logger.info(f"Loaded training dataset with {len(train_dataset)} samples.")
    logger.info(f"Loaded validation dataset with {len(val_dataset)} samples.")

    model = StitchingModel(target_height=target_height,
                           target_width=target_width,
                           batch_size=const.BATCH_SIZE,
                           original_height=const.HEIGHT, 
                           original_width=const.WIDTH,
                           homography_model=const.HOMOGRAPHY_CHECKPOINT).to(device)  # Pass the Homography Estimator to the Stitching Model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
    writer = SummaryWriter(log_dir=str(const.SUMMARY_DIR))
    if const.CHECKPOINT:
        logger.info(f"Loading checkpoint from {const.CHECKPOINT}")
        load_checkpoint(const.CHECKPOINT, model, optimizer)

    overall_iteration = 0
    best_loss = float("inf")
    for epoch in range(const.ITERATION):
        logger.info(f"Starting epoch {epoch + 1}/{const.ITERATION}")
        model.train()
        epoch_loss = 0
        set_size = len(train_loader)
        # Training Loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            input1, input2 = inputs
            input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)

            outputs, coarse_stitched = model(input1, input2, targets)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            if const.GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), const.GRADIENT_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            overall_iteration += 1
            if (batch_idx + 1) % 1000 == 0:
                logger.info(f"Epoch [{epoch + 1}/{const.ITERATION}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                writer.add_scalar('Loss/train', loss.item(), overall_iteration)

        epoch_loss /= set_size
        logger.info(f"Epoch [{epoch + 1}/{const.ITERATION}] Average Training Loss: {epoch_loss:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                input1, input2 = inputs
                input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)

                outputs, coarse_stitched = model(input1, input2, targets)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Save images to TensorBoard
                if batch_idx == 0:  # Save only the first batch for visualization
                    save_images_to_tensorboard(writer, input1, "Validation/Input1", epoch + 1)
                    save_images_to_tensorboard(writer, input2, "Validation/Input2", epoch + 1)
                    save_images_to_tensorboard(writer, coarse_stitched[:, 6:8, :, :], "Validation/CoarseStitched", epoch + 1)
                    save_images_to_tensorboard(writer, outputs, "Validation/FinalStitched", epoch + 1)
                    save_images_to_tensorboard(writer, targets, "Validation/Targets", epoch + 1)

        val_loss /= len(val_loader)
        logger.info(f"Epoch [{epoch + 1}/{const.ITERATION}] Validation Loss: {val_loss:.4f}")
        writer.add_scalar('Loss/validation', val_loss, epoch + 1)

        # Save checkpoint if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            logger.info(f"New best validation loss: {best_loss:.4f}. Saving checkpoint...")
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_dir)

    logger.info("Training process completed.")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()