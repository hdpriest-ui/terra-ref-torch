import torch

###### Method copied + updated from Nie nielang@bjtu.edu.cn
# @article{nie2020view,
#   title={A view-free image stitching network based on global homography},
#   author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Meiqin and Zhao, Yao},
#   journal={Journal of Visual Communication and Image Representation},
#   volume={73},
#   pages={102950},
#   year={2020},
#   publisher={Elsevier}
# }
#
# updated and adapted for PyTorch
#


def transform(image2_tensor, H_tf):

    def _repeat(x, n_repeats):
            rep = torch.ones([1, n_repeats], dtype=torch.float32)
            x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
            return torch.reshape(x, [-1])

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

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

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
        base = _repeat(torch.arange(0, end=num_batch, dtype=torch.float32) * dim1, out_height * out_width)
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

    def _meshgrid(height, width):
        x_t = torch.matmul(torch.ones([height, 1]),
                           torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                           torch.ones([1, width]))

        x_t_flat = torch.reshape(x_t, (1, -1))
        y_t_flat = torch.reshape(y_t, (1, -1))

        ones = torch.ones_like(x_t_flat)
        grid = torch.concat([x_t_flat, y_t_flat, ones], 0)
        return grid

    def _transform(image2_tensor, H_tf):
        num_batch = image2_tensor.shape[0]
        height = image2_tensor.shape[2]
        width = image2_tensor.shape[3]
        num_channels = image2_tensor.shape[1]

        H_tf = torch.reshape(H_tf, (-1, 3, 3))
        H_tf_shape = list(H_tf.shape)
        # initial

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = 128
        out_width = 128
        grid = _meshgrid(out_height, out_width)
        grid = torch.unsqueeze(grid, 0)
        grid = torch.reshape(grid, [-1])
        grid = torch.tile(grid, [num_batch])
        grid = torch.reshape(grid, [num_batch, 3, -1])

        T_g = torch.matmul(H_tf, grid)
        x_s = T_g[0:, 0:1, 0:]
        y_s = T_g[0:, 1:2, 0:]
        t_s = T_g[0:, 2:3, 0:]
        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        t_s_flat = torch.reshape(t_s, [-1])
        x_s_flat = torch.reshape(x_s, [-1]) / t_s_flat
        y_s_flat = torch.reshape(y_s, [-1]) / t_s_flat

        input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (128, 128))     ### Identical to here
        output = torch.reshape(input=input_transformed, shape=[num_batch, out_height, out_width, num_channels])
        # output = torch.reshape(input=input_transformed, shape=[num_batch, num_channels, out_height, out_width])
        output = output.permute(0, 3, 1, 2)
        return output

    output = _transform(image2_tensor, H_tf)
    return output