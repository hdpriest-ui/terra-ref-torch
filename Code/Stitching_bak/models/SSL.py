import torch
import torch.nn.functional as F


class StructureStitchingLayer:
    """
    Structure Stitching Layer (SSL) for applying homography transformations
    to input images and generating coarse-stitched outputs.
    """

    @staticmethod
    def _transform(image, homography, height, width):
        """
        Apply a homography transformation to the input image.

        Args:
            image (torch.Tensor): Input image of shape (batch_size, channels, height, width).
            homography (torch.Tensor): Homography matrix of shape (batch_size, 3, 3).
            height (int): Height of the output image.
            width (int): Width of the output image.

        Returns:
            torch.Tensor: Transformed image of shape (batch_size, channels, height, width).
        """
        batch_size, channels, _, _ = image.shape

        # Create a meshgrid of pixel coordinates
        y, x = torch.meshgrid(
            torch.linspace(0, height - 1, height, device=image.device),
            torch.linspace(0, width - 1, width, device=image.device),
        )
        ones = torch.ones_like(x)
        grid = torch.stack([x, y, ones], dim=0).unsqueeze(0)  # Shape: (1, 3, height, width)
        grid = grid.repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, 3, height, width)

        # Flatten the grid and apply the homography
        grid = grid.view(batch_size, 3, -1)  # Shape: (batch_size, 3, height * width)
        transformed_grid = torch.bmm(homography, grid)  # Shape: (batch_size, 3, height * width)

        # Normalize the transformed grid
        x_t = transformed_grid[:, 0, :] / (transformed_grid[:, 2, :] + 1e-8)
        y_t = transformed_grid[:, 1, :] / (transformed_grid[:, 2, :] + 1e-8)

        # Reshape to (batch_size, height, width)
        x_t = x_t.view(batch_size, height, width)
        y_t = y_t.view(batch_size, height, width)

        # Stack and normalize to [-1, 1] for grid_sample
        x_t = 2.0 * x_t / (width - 1) - 1.0
        y_t = 2.0 * y_t / (height - 1) - 1.0
        grid = torch.stack([x_t, y_t], dim=-1)  # Shape: (batch_size, height, width, 2)

        # Apply the grid transformation
        transformed_image = F.grid_sample(image, grid, align_corners=True)
        return transformed_image

    @staticmethod
    def apply(inputs, homography):
        """
        Perform structure stitching using homography transformations.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, 6, height, width),
                                   where the first 3 channels are `input1` and the next 3 are `input2`.
            homography (torch.Tensor): Homography matrix of shape (batch_size, 3, 3).

        Returns:
            torch.Tensor: Stitched output of shape (batch_size, 3, height, width).
        """
        batch_size, _, height, width = inputs.shape

        # Split inputs into input1 and input2
        input1 = inputs[:, :3, :, :] + 1.0  # Add 1 for normalization
        input2 = inputs[:, 3:, :, :] + 1.0

        # Step 1: Transform input1 with identity homography
        identity_homography = torch.eye(3, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)
        transformed_input1 = StructureStitchingLayer._transform(input1, identity_homography, height, width)

        # Step 2: Transform input2 with the provided homography
        transformed_input2 = StructureStitchingLayer._transform(input2, homography, height, width)

        # Step 3: Create masks for valid regions
        mask1 = (transformed_input1.abs() > 1e-6).float()
        mask2 = (transformed_input2.abs() > 1e-6).float()
        overlap_mask = mask1 * mask2
        mask1 = mask1 - overlap_mask
        mask2 = mask2 - overlap_mask

        # Combine the transformed images using the masks
        stitched_output = (
            transformed_input1 * mask1
            + transformed_input2 * mask2
            + 0.5 * (transformed_input1 + transformed_input2) * overlap_mask
        )

        # Normalize back to the original range
        stitched_output = stitched_output - 1.0

        return stitched_output