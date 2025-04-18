import os
import sys
import torch
from torch.utils.data import DataLoader
# Absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # Adjust the path to your project structure
from dataset import StitchingDataset
from models.StitchingModel import StitchingModel
from utils import load_checkpoint, save_heatmap
import constant
from stitch_logger import get_logger
from Homography.models import HomographyEstimator

# Initialize logger
logger = get_logger(__name__)

logger.info("Starting inference process...")

# Initialize device
device = torch.device(f"cuda:{constant.GPU}" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize dataset and dataloader
test_dataset = StitchingDataset(constant.TEST_DATA_DIRECTORY, resize_height=constant.HEIGHT, resize_width=constant.WIDTH)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3)
logger.info(f"Loaded test dataset with {len(test_dataset)} samples.")

# Initialize model
model = StitchingModel(height=constant.HEIGHT, width=constant.WIDTH, homography_model=constant.HOMOGRAPHY_CHECKPOINT).to(device)

# Load checkpoint
if constant.CHECKPOINT:
    logger.info(f"Loading checkpoint from {constant.CHECKPOINT}")
    load_checkpoint(constant.CHECKPOINT, model)

# Inference loop
model.eval()
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(test_loader):
        input1, input2 = inputs
        input1, input2, targets = input1.to(device), input2.to(device), targets.to(device)
        # homography_output = self.homography_model(torch.cat([input1, input2], dim=1))
        # Forward pass
        outputs, coarse_stitched = model(input1, input2)

        # Save outputs and heatmaps
        output_path = os.path.join(constant.RESULT_DIR, f"output_{idx + 1}.png")
        coarse_path = os.path.join(constant.RESULT_DIR, f"coarse_{idx + 1}.png")
        target_path = os.path.join(constant.RESULT_DIR, f"target_{idx + 1}.png")

        # Save stitched outputs and coarse images
        save_heatmap(outputs, targets, idx, constant.RESULT_DIR)
        torch.save(outputs.cpu(), output_path)
        torch.save(coarse_stitched.cpu(), coarse_path)
        torch.save(targets.cpu(), target_path)

        logger.debug(f"Processed sample {idx + 1}/{len(test_loader)}")

logger.info("Inference process completed.")