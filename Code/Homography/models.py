import torch.nn as nn
import torch as torch

from tensorDLT import solve_DLT
from spatial_transform import transform
import numpy as np

def device():
    cuda = torch.cuda.is_available()
    if cuda:
        the_device = torch.device('cuda')
        torch.set_default_device('cuda')
        return the_device
    else:
        the_device = torch.device('cpu')
        return the_device


class HomographyFeatureExtractor(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        # block 1
        kernel_size = 5
        channel_base = 64
        padding = int((kernel_size - 1)/2)
        self.batch_size = batch_size
        self.conv1a = nn.Conv2d(1, channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act1a = nn.ReLU()
        self.conv1b = nn.Conv2d(channel_base, channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act1b = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.conv2a = nn.Conv2d(channel_base, channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act2a = nn.ReLU()
        self.conv2b = nn.Conv2d(channel_base, channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act2b = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3
        self.conv3a = nn.Conv2d(channel_base, 2*channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act3a = nn.ReLU()
        self.conv3b = nn.Conv2d(2*channel_base, 2*channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act3b = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4
        self.conv4a = nn.Conv2d(2*channel_base, 2*channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act4a = nn.ReLU()
        self.conv4b = nn.Conv2d(2*channel_base, 2*channel_base, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act4b = nn.ReLU()

        # weight init was brought in to work on the poor training performance prior to major pytorch overhaul.
        # consider reintroducing after you see performance of training without it, after pytorch overhaul
        # this has not been seen yet as of 3/11 afternoon.
        # i believe torch initializes weights and biases to torch.empty; tf inits to xavier normal. trying this
        def _weights_init(m):
            # print("confirming feature extractor weight init function is called.")
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print("initializing weights using xavier normal in homography estimator...")
                nn.init.xavier_normal_(m.weight)

        self.apply(_weights_init)

    def forward(self, x):
        x = self.act1a(self.conv1a(x))
        x = self.act1b(self.conv1b(x))
        x = self.pool1(x)
        x = self.act2a(self.conv2a(x))
        x = self.act2b(self.conv2b(x))
        x = self.pool2(x)
        x = self.act3a(self.conv3a(x))
        x = self.act3b(self.conv3b(x))
        x = self.pool3(x)
        x = self.act4a(self.conv4a(x))
        x = self.act4b(self.conv4b(x))
        return x

class HomographyEstimator(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        this_device = device()
        self.batch_size = batch_size
        self.feature_extractor_a = HomographyFeatureExtractor(batch_size).to(device())
        self.feature_extractor_b = HomographyFeatureExtractor(batch_size).to(device())
        search_range = 16
        input_dims = (search_range * 2 + 1)**2
        kernel_size = 5
        padding = int((kernel_size - 1)/2)
        self.conv1 = nn.Conv2d(in_channels=input_dims, out_channels=512, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding) ## elhan suggests removing this layer
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(kernel_size, kernel_size), stride=1, padding=padding) ## EE: reduce down to orig. input dimensions
        self.act3 = nn.ReLU()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=131072, out_features=1024)
        self.fc_act = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=8)

        # weight init was brought in to work on the poor training performance prior to major pytorch overhaul.
        # consider reintroducing after you see performance of training without it, after pytorch overhaul
        # this has not been seen yet as of 3/11 afternoon.
        def _weights_init(m):
            # print("confirming homography estimator weight init function is called.")
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print("initializing weights using xavier normal in homography estimator...")
                nn.init.xavier_normal_(m.weight)

        self.apply(_weights_init)

    def cost_volume(self, c1, warp, search_range):
        """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
        Args:
            c1: Level of the feature pyramid of Image1
            warp: Warped level of the feature pyramid of image22
            search_range: Search range (maximum displacement)
        """
        padded_lvl = nn.functional.pad(warp, (search_range, search_range, search_range, search_range, 0, 0, 0, 0))
        _, h, w = torch.unbind(c1)[0].size()
        max_offset = search_range * 2 + 1
        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                # this slices out all elements of dim 1 and 4, and from y-h and from x-w on dims 2, 3
                # slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
                slice = padded_lvl[:, :, y:y + h, x:x + w]
                # cost = tf.reduce_mean(input_tensor=c1 * slice, axis=3, keepdims=True)
                cost = torch.mean(input=c1 * slice, dim=1, keepdim=True)
                cost_vol.append(cost)
        # cost_vol = tf.concat(cost_vol, axis=3)
        cost_vol = torch.cat(cost_vol, dim=1)
        m = nn.LeakyReLU(0.1)
        cost_vol = m(cost_vol)

        return cost_vol

    def build_model(self, inputs_a, inputs_b):
        # extractor_a = HomographyFeatureExtractor(batch_size=self.batch_size).to(device())
        # extractor_b = HomographyFeatureExtractor(batch_size=self.batch_size).to(device())
        extractor_a = self.feature_extractor_a
        extractor_b = self.feature_extractor_b
        gs_a = torch.mean(input=inputs_a, dim=1, keepdim=True)
        gs_b = torch.mean(input=inputs_b, dim=1, keepdim=True)
        # feature_a = extractor_a.forward(torch.mean(input=inputs_a, dim=1, keepdim=True))
        # feature_b = extractor_b.forward(torch.mean(input=inputs_b, dim=1, keepdim=True))
        # feature_a = extractor_a.forward(gs_a)
        # feature_b = extractor_b.forward(gs_b)
        feature_a = extractor_a(gs_a)
        feature_b = extractor_b(gs_b)
        search_range = 16
        # compute overlap
        global_correlation = self.cost_volume(nn.functional.normalize(feature_a, p=2, dim=1),
                                              nn.functional.normalize(feature_b, p=2, dim=1), search_range)

        # similarity metric needed - cosine sim.
        ## look at the dist. of deltas between A/B to understand how we will need to transform the similarity back to scoring space
        ## xgboost - model category here

        x = self.act1(self.conv1(global_correlation))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc_act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

    def forward(self, inputs_a, inputs_b, label, patch_size=128.):
        # unsure what patch size parameterization supports:
        # i expect it represents the input size and the area of the overlaid two inputs for homography estimation
        batch_size = inputs_a.shape[0]
        predicted_shift = self.build_model(inputs_a, inputs_b)
        # predicted_shift = predicted_shift.unsqueeze(2)

        homography_gt = solve_DLT(label, patch_size)
        homography_gt = homography_gt.to(device())
        homography_gt_inverse = homography_gt

        # high risk operation here. implementing linear alg. methods i don't fully understand.
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                      [0., patch_size / 2.0, patch_size / 2.0],
                      [0., 0., 1.]]).astype(np.float32)

        M_tensor = torch.from_numpy(M)
        M_tile = torch.tile(torch.unsqueeze(M_tensor, 0), dims=[batch_size, 1, 1])
        M_tensor_inverse = torch.inverse(M_tensor)
        M_tile_inverse = torch.tile(torch.unsqueeze(M_tensor_inverse, 0), dims=[batch_size, 1, 1])
        M_tile_inverse=M_tile_inverse.to(device())
        M_tile=M_tile.to(device())
        homography_gt_inverse=homography_gt_inverse.to(device())
        step1 = torch.matmul(M_tile_inverse, homography_gt_inverse)
        step1.to(device())
        homography_gt_mat = torch.matmul(step1, M_tile)
        warp_gt = transform(inputs_b, homography_gt_mat) # validated identical
        return predicted_shift, warp_gt

