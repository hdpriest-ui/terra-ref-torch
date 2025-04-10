import torch
import torch.nn as nn
from models.unet import UNet
from models.SSL import StructureStitchingLayer
from models.tensorDLT import solve_DLT


class HomographyEstimator(nn.Module):
    """
    Homography Estimator to predict the shift for computing the homography matrix.
    """
    def __init__(self, in_channels=6, features=64):
        super(HomographyEstimator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(features * 4 * 16 * 16, 8)  # Predict 8 values for the 4 corner points

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        shift = self.fc(x)
        return shift


class StitchingModel(nn.Module):
    """
    Full stitching model integrating homography estimation, SSL, and U-Net.
    """
    def __init__(self, height=128, width=128):
        super(StitchingModel, self).__init__()
        self.height = height
        self.width = width
        self.h_estimator = HomographyEstimator(in_channels=6)
        self.unet = UNet(in_channels=6, out_channels=3)

    def forward(self, input1, input2):
        # Step 1: Concatenate input1 and input2 along the channel dimension
        inputs = torch.cat([input1, input2], dim=1)

        # Step 2: Predict shift using the homography estimator
        shift = self.h_estimator(inputs)

        # Step 3: Compute the homography matrix
        H = solve_DLT(shift, scale=max(self.height, self.width))

        # Step 4: Apply the homography transformation using SSL
        coarse_stitched = StructureStitchingLayer.apply(input1, H)

        # Step 5: Pass the coarse-stitched output and input2 to the U-Net
        refined_output = self.unet(torch.cat([coarse_stitched, input2], dim=1))

        return refined_output, coarse_stitched