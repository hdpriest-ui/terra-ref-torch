import torch
import torch.nn.functional as F


class StructureStitchingLayer:
    """
    Structure Stitching Layer (SSL) for applying homography transformations
    to input images and generating coarse-stitched outputs.
    """

    @staticmethod
    def _repeat(x, n_repeats):
            rep = torch.ones([1, n_repeats], dtype=torch.float32)
            x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
            return torch.reshape(x, [-1])

    @staticmethod
    def _interpolate(im, x, y, out_size):
        num_batch = im.shape[0]
        height = im.shape[2]
        width = im.shape[3]
        channels = im.shape[1]
        height_f = torch.tensor(data=height, dtype=torch.float32)
        width_f = torch.tensor(data=width, dtype=torch.float32)
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([], dtype=torch.int32)
        max_y = torch.tensor(data=im.shape[2] - 1, dtype=torch.int32)
        max_x = torch.tensor(data=im.shape[3] - 1, dtype=torch.int32)

        # # scale indices from [-1, 1] to [0, width/height]
        # x = (x + 1.0) * (width_f) / 2.0
        # y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clip(x0, min=zero, max=max_x)
        x1 = torch.clip(x1, min=zero, max=max_x)
        y0 = torch.clip(y0, min=zero, max=max_y)
        y1 = torch.clip(y1, min=zero, max=max_y)

        dim2 = width
        dim1 = width * height
        base = StructureStitchingLayer._repeat(torch.arange(0, end=num_batch, dtype=torch.float32) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim

        im_flat = torch.reshape(input=im.permute(0, 2, 3, 1), shape=[-1, channels])
        # im_flat = torch.cast(im_flat, 'float32')
        Ia = torch.gather(im_flat, dim=0, index=torch.tile(idx_a, (3, 1)).permute((1, 0)).long())
        Ib = torch.gather(im_flat, dim=0, index=torch.tile(idx_b, (3, 1)).permute((1, 0)).long())
        Ic = torch.gather(im_flat, dim=0, index=torch.tile(idx_c, (3, 1)).permute((1, 0)).long())
        Id = torch.gather(im_flat, dim=0, index=torch.tile(idx_d, (3, 1)).permute((1, 0)).long())

        # and finally calculate interpolated values
        x0_f = x0.to(torch.float32)
        x1_f = x1.to(torch.float32)
        y0_f = y0.to(torch.float32)
        y1_f = y1.to(torch.float32)

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
        return output
        
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
        batch_size, channels, original_height, original_width = image.shape
        shift = (height - original_height) / 2
        # Create a meshgrid of pixel coordinates
        x_lin = torch.transpose(torch.linspace(-shift, width - shift - 1, width, device=image.device).unsqueeze(1), 1, 0)
        x_ones = torch.ones([x_lin.shape[1], 1], device=image.device)
        
        y_lin = torch.linspace(-shift, height - shift - 1, height, device=image.device).unsqueeze(1)
        y_ones = torch.ones([1, y_lin.shape[0]], device=image.device)
        
        x_t = torch.matmul(x_ones, x_lin.unsqueeze(0))  # Shape: (batch_size, 1, width)
        y_t = torch.matmul(y_lin.unsqueeze(0), y_ones)  # Shape: (batch_size, 1, height)
        
        # y, x = torch.meshgrid(
        #     torch.linspace(0, height - 1, height, device=image.device),
        #     torch.linspace(0, width - 1, width, device=image.device),
        # )
        ones = torch.ones_like(x_t)
        grid = torch.stack([x_t, y_t, ones], dim=1)  # Shape: (1, 3, height, width)
        grid = grid.repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, 3, height, width)

        # Flatten the grid and apply the homography
        grid = grid.view(batch_size, 3, -1)  # Shape: (batch_size, 3, height * width)
        transformed_grid = torch.bmm(homography, grid)  # Shape: (batch_size, 3, height * width)

        # Normalize the transformed grid
        x_t_flat = torch.reshape(transformed_grid[:, 0, :] / (transformed_grid[:, 2, :] + 1e-8), [-1])
        y_t_flat = torch.reshape(transformed_grid[:, 1, :] / (transformed_grid[:, 2, :] + 1e-8), [-1])

        # # Reshape to (batch_size, height, width)
        # y_reshape = torch.reshape(y_t, [batch_size, height, width])
        # x_reshape = torch.reshape(x_t, [batch_size, height, width])
        # x_t = x_t.view(batch_size, height, width)
        # y_t = y_t.view(batch_size, height, width)

        # # Stack and normalize to [-1, 1] for grid_sample
        # x_t = 2.0 * x_t / (width - 1) - 1.0
        # y_t = 2.0 * y_t / (height - 1) - 1.0
        # grid = torch.stack([x_t, y_t], dim=-1)  # Shape: (batch_size, height, width, 2)

        # Apply the grid transformation
        # transformed_image = _interpolate(image, grid, align_corners=True)
        transformed_image = StructureStitchingLayer._interpolate(image, x_t_flat, y_t_flat, (height, width))  # Shape: (batch_size, channels, height, width)
        output = torch.reshape(transformed_image, [batch_size, channels, height, width])
        return output

    @staticmethod
    def apply(inputs, homography, target_height, target_width):
        """
        Perform structure stitching using homography transformations.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, 6, height, width),
                                   where the first 3 channels are `input1` and the next 3 are `input2`.
            homography (torch.Tensor): Homography matrix of shape (batch_size, 3, 3).

        Returns:
            torch.Tensor: Stitched output of shape (batch_size, 3, height, width).
        """
        batch_size, _, input_height, input_width = inputs.shape
        height = target_height
        width = target_width

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