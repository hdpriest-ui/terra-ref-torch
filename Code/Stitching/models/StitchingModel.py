import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # Adjust the path to your project structure
from Stitching.models.unet import UNet
from Stitching.models.SSL import StructureStitchingLayer
from Homography.tensorDLT import solve_DLT
from Homography.models import HomographyEstimator
from stitch_logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# class HomographyEstimator(nn.Module):
#     """
#     Homography Estimator to predict the shift for computing the homography matrix.
#     """
#     def __init__(self, in_channels=6, features=64):
#         super(HomographyEstimator, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(features, features * 2, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1)
#         self.fc = nn.Linear(features * 4 * 16 * 16, 8)  # Predict 8 values for the 4 corner points

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)  # Flatten
#         shift = self.fc(x)
#         return shift


class StitchingModel(nn.Module):
    """
    Full stitching model integrating homography estimation, SSL, and U-Net.
    """
    def __init__(self, target_height=None, target_width=None, homography_model=None, batch_size=None, original_height=128, original_width=128):
        super(StitchingModel, self).__init__()
        self.height = original_height
        self.width = original_width
        self.stitched_height = target_height
        self.stitched_width = target_width
        self.batch_size = batch_size
        self.h_estimator = self._load_homography_estimator(model_path=homography_model)
        self.unet = UNet(in_channels=9, out_channels=3)
        
    def _load_homography_estimator(self, model_path=None):
        # Load Homography Estimator model
        if model_path is not None:
            logger.info(f"Loading homography model from checkpoint: {model_path}")
            homography_model = HomographyEstimator(batch_size=self.batch_size)
            homography_checkpoint = torch.load(model_path)
            homography_model.load_state_dict(homography_checkpoint["model_state_dict"])
            homography_model.eval()  # Set to evaluation mode
            for param in homography_model.parameters():
                param.requires_grad = False  # Freeze parameters
            return homography_model
        else:
            homography_model = HomographyEstimator(batch_size=self.batch_size)
            return homography_model
        

    def forward(self, input1, input2, targets, is_stitching=True):
        # Step 1: Concatenate input1 and input2 along the channel dimension
        inputs = torch.cat([input1, input2], dim=1)

        # Step 2: Predict shift using the homography estimator
        # HomographyModel(image_a, image_b, labels)
        shift, warp_gt = self.h_estimator(input1, input2, targets, is_stitching)

        # Step 3: Compute the homography matrix
        H = solve_DLT(shift, patch_size=max(self.height, self.width))

        # Step 4: Apply the homography transformation using SSL
        # coarse_stitched = StructureStitchingLayer.apply(inputs, H, self.stitched_height, self.stitched_width)
        coarse_stitched = StructureStitchingLayer(inputs, H, self.stitched_height, self.stitched_width)

        # Step 5: Pass the coarse-stitched output and input2 to the U-Net
        refined_output = self.unet(coarse_stitched)

        return refined_output, coarse_stitched