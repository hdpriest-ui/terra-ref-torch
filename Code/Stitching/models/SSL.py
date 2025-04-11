import torch
import torch.nn.functional as F


class StructureStitchingLayer:
    """
    Structure Stitching Layer (SSL) for applying homography transformations
    to input images and generating coarse-stitched outputs.
    """
    
    @staticmethod
    def _repeat(x, n_repeats):
        """Repeats a tensor along a new dimension."""
        rep = torch.ones(n_repeats, 1, device=x.device)
        rep = rep.float()
        x = x.float()
        x_flat = x.reshape([-1, 1])
        repeated_flattened_x = torch.matmul(x_flat, rep.t())
        repeated_flattened_x = repeated_flattened_x.int()
        return repeated_flattened_x.reshape([-1])

    @staticmethod
    def _interpolate(im, x, y, out_size):
        """Performs bilinear interpolation on the input image."""
        num_batch, channels, height, width = im.shape
        out_height, out_width = out_size
        
        x = x.float()
        y = y.float()
        height_f = float(height)
        width_f = float(width)
        
        # Scale indices from [-1, 1] to [0, width/height]
        # x = (x + 1.0) * (width_f) / 2.0
        # y = (y + 1.0) * (height_f) / 2.0
        
        # Do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        
        # Clip to valid range
        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)
        
        # Calculate base indices
        dim2 = width
        dim1 = width * height
        base = StructureStitchingLayer._repeat(torch.arange(num_batch, device=im.device) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        
        # Reshape image for gathering
        im_flat = im.reshape(-1, channels)
        im_flat = im_flat.float()
        
        # Gather values
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]
        
        # Calculate weights
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)
        
        # Calculate interpolated values
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    @staticmethod
    def _meshgrid(image, height, width):
        """Creates a grid of coordinates."""
        # Calculate the shift needed to center the image
        shift = (width - image.shape[3]) / 2.0
        device = image.device
        
        # Create x coordinates with centering
        x_t = torch.matmul(torch.ones(height, 1, device=device),
                        torch.linspace(-shift, width - shift, width, device=device).unsqueeze(0))
        
        # Create y coordinates with centering
        y_t = torch.matmul(torch.linspace(-shift, height - shift, height, device=device).unsqueeze(1),
                        torch.ones(1, width, device=device))
        
        x_t_flat = x_t.reshape([1, -1])
        y_t_flat = y_t.reshape([1, -1])
        
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    @staticmethod
    def _transform(image, H, height, width):
        """Applies homography transformation to the input image."""
        num_batch = image.shape[0]
        original_height = image.shape[2]
        original_width = image.shape[3]
        num_channels = image.shape[1]
        
        H = H.reshape([-1, 3, 3])
        H = H.float()
        
        # Create grid
        out_height = height
        out_width = width
        grid = StructureStitchingLayer._meshgrid(image, out_height, out_width)
        grid = grid.unsqueeze(0)
        grid = grid.reshape(-1)
        grid = grid.repeat(num_batch)
        grid = grid.reshape(num_batch, 3, -1)
        
        # Apply transformation
        T_g = torch.matmul(H, grid)
        x_s = T_g[:, 0:1, :]
        y_s = T_g[:, 1:2, :]
        t_s = T_g[:, 2:3, :]
        
        t_s_flat = t_s.reshape(-1)
        
        # Avoid division by zero
        small = torch.tensor(1e-7, device=image.device)
        smallers = 1e-6 * (1.0 - (torch.abs(t_s_flat) >= small).float())
        t_s_flat = t_s_flat + smallers
        
        # Normalize coordinates
        x_s_flat = x_s.reshape(-1) / t_s_flat
        y_s_flat = y_s.reshape(-1) / t_s_flat
        
        # Perform interpolation
        input_transformed = StructureStitchingLayer._interpolate(image, x_s_flat, y_s_flat, (out_height, out_width))
        
        output = input_transformed.reshape([num_batch, num_channels, out_height, out_width])
        return output
    
    @staticmethod
    def apply(inputs, H, target_height, target_width):
        """
        Perform structure stitching using homography transformations.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, 6, height, width),
                                   where the first 3 channels are `input1` and the next 3 are `input2`.
            homography (torch.Tensor): Homography matrix of shape (batch_size, 3, 3).

        Returns:
            torch.Tensor: Stitched output of shape (batch_size, 3, height, width).
        """
        batch_size = H.shape[0]
    
        # Step 1: Transform first image with identity homography
        H_one = torch.eye(3, device=inputs.device)
        H_one = H_one.unsqueeze(0).repeat(batch_size, 1, 1)

        img1 = inputs[:, :3, :, :] + 1
        img1_tf = StructureStitchingLayer._transform(img1, H_one, target_height, target_width)
        
        warp = inputs[:, 3:, :, :] + 1
        warp_tf = StructureStitchingLayer._transform(warp, H, target_height, target_width)
        
        # Step 3: Create masks and combine images
        # Create masks based on valid pixel regions
        mask1 = (torch.sum(torch.abs(img1_tf), dim=1, keepdim=True) > 1e-6).float()
        mask2 = (torch.sum(torch.abs(warp_tf), dim=1, keepdim=True) > 1e-6).float()
        
        # Calculate overlap region
        overlap = mask1 * mask2
        
        # Calculate non-overlapping regions
        mask1_only = mask1 * (1 - overlap)
        mask2_only = mask2 * (1 - overlap)
        
        # Combine images using masks with proper blending
        structureStitching = (
            img1_tf * mask1_only +  # First image only
            warp_tf * mask2_only +  # Second image only
            0.5 * (img1_tf + warp_tf) * overlap  # Overlap region
        )
        
        img1_tf = img1_tf - 1
        warp_tf = warp_tf - 1
        structureStitching = structureStitching - 1
        
        # Step 4: Concatenate results
        output = torch.cat([img1_tf, warp_tf, structureStitching], dim=1)
        
        return output 