import torch
import torch.nn.functional as F
import numpy as np

def StructureStitchingLayer(inputs, H_tf, target_height, target_width):
    def _repeat(x, n_repeats):
        rep = torch.ones(n_repeats, 1,  device=x.device).t()
        x = x.float()
        rep = rep.float()
        x = torch.matmul(x.reshape(-1, 1), rep)
        x = x.int()
        return x.reshape(-1)

    def _interpolate(im, x, y, out_size):
        num_batch = im.size(0)
        height = im.size(2)
        width = im.size(3)
        channels = im.size(1)

        x = x.float()
        y = y.float()
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([], dtype=torch.int32, device=im.device)
        max_y = torch.tensor(height - 1, dtype=torch.int32, device=im.device)
        max_x = torch.tensor(width - 1, dtype=torch.int32, device=im.device)

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch, device=im.device) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
        im_flat = im.permute(0, 2, 3, 1)
        im_flat = im_flat.reshape(-1, channels)
        im_flat = im_flat.float()
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        output = output.reshape(num_batch, out_height, out_width, channels)
        output = output.permute(0, 3, 1, 2)
        return output

    def _meshgrid(height, width, image_tf):
        device = image_tf.device
        shift = (height - image_tf.shape[2]) / 2.
        x_t = torch.matmul(torch.ones(height, 1, device=device),
                         torch.linspace(0. - shift, 1.0 * width - shift, width, device=device).unsqueeze(0))
        y_t = torch.matmul(torch.linspace(0. - shift, 1.0 * height - shift, height, device=device).unsqueeze(1),
                         torch.ones(1, width, device=device))

        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def _transform(image_tf, H_tf):
        num_batch = image_tf.size(0)
        num_channels = image_tf.size(1)
        H_tf = H_tf.reshape(-1, 3, 3)
        H_tf = H_tf.float()

        out_height = target_height
        out_width = target_width
        grid = _meshgrid(out_height, out_width, image_tf)
        grid = grid.unsqueeze(0)
        grid = grid.reshape(-1)
        grid = grid.repeat(num_batch)
        grid = grid.reshape(num_batch, 3, -1)

        T_g = torch.matmul(H_tf, grid)
        x_s = T_g[:, 0:1, :]
        y_s = T_g[:, 1:2, :]
        t_s = T_g[:, 2:3, :]
        t_s_flat = t_s.reshape(-1)

        one = torch.tensor(1.0, dtype=torch.float32, device=image_tf.device)
        small = torch.tensor(1e-7, dtype=torch.float32, device=image_tf.device)
        smallers = 1e-6 * (one - (torch.abs(t_s_flat) >= small).float())

        t_s_flat = t_s_flat + smallers
        x_s_flat = x_s.reshape(-1) / t_s_flat
        y_s_flat = y_s.reshape(-1) / t_s_flat

        output = _interpolate(image_tf, x_s_flat, y_s_flat, (out_height, out_width))
        # output = input_transformed.reshape(num_batch, num_channels, out_height, out_width)
        return output

    batch_size = H_tf.size(0)
    mask_one = torch.ones_like(inputs[:, 0:3, :, :], dtype=torch.float32)
    
    # Step 1
    H_one = torch.eye(3, device=inputs.device)
    H_one = H_one.unsqueeze(0).repeat(batch_size, 1, 1)
    img1_tf = inputs[:, 0:3, :, :] + 1.
    img1_tf = _transform(img1_tf, H_one)
    
    # Step 2
    warp_tf = inputs[:, 3:6, :, :] + 1.
    warp_tf = _transform(warp_tf, H_tf)
    
    # Step 3
    one = torch.ones_like(img1_tf, dtype=torch.float32)
    zero = torch.zeros_like(img1_tf, dtype=torch.float32)
    mask1 = torch.where(torch.abs(img1_tf) < 1e-6, zero, one)
    mask2 = torch.where(torch.abs(warp_tf) < 1e-6, zero, one)
    mask = torch.mul(mask1, mask2)
    mask1 = mask1 - mask
    mask2 = mask2 - mask
    structureStitching = zero + torch.mul(warp_tf, mask2) + torch.mul(img1_tf, mask1) \
              + 0.5*torch.mul(img1_tf, mask) + 0.5*torch.mul(warp_tf, mask)
    img1_tf = img1_tf - 1.
    warp_tf = warp_tf - 1.
    structureStitching = structureStitching - 1.
    
    # Step 4
    output = torch.cat([img1_tf, warp_tf, structureStitching], dim=1)
    
    return output