import torch.nn as nn
import torch as torch

def device():
    the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return the_device


class HomographyFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # block 1
        self.conv1a = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act1a = nn.ReLU()
        self.conv1b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act1b = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.conv2a = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act2a = nn.ReLU()
        self.conv2b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act2b = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3
        self.conv3a = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act3a = nn.ReLU()
        self.conv3b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act3b = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 4
        self.conv4a = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act4a = nn.ReLU()
        self.conv4b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act4b = nn.ReLU()

    def forward(self, x):
        # input 3x32x32, output 32x32x32
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
    def __init__(self):
        super().__init__()
        search_range = 16
        input_dims = (search_range * 2 + 1)**2
        self.conv1 = nn.Conv2d(in_channels=input_dims, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.act3 = nn.ReLU()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=131072, out_features=1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=8)

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
        # cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1)
        m = nn.LeakyReLU(0.1)
        cost_vol = m(cost_vol)

        return cost_vol

    def forward(self, inputs_a, inputs_b):
        extractor_a = HomographyFeatureExtractor().to(device())
        extractor_b = HomographyFeatureExtractor().to(device())
        # greyscale, i think
        # feature extraction
        feature_a = extractor_a.forward(torch.mean(input=inputs_a, dim=1, keepdim=True).expand(1, 3, 128, 128))
        feature_b = extractor_b.forward(torch.mean(input=inputs_b, dim=1, keepdim=True).expand(1, 3, 128, 128))
        search_range = 16
        # compute overlap
        global_correlation = self.cost_volume(nn.functional.normalize(feature_a, p=2, dim=1), nn.functional.normalize(feature_b, p=2, dim=1), search_range)

        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(global_correlation))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.flat(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)

        # x = self.drop1(x)
        # # input 32x32x32, output 32x32x32
        # x = self.act2(self.conv2(x))
        # # input 32x32x32, output 32x16x16
        # x = self.pool2(x)
        # # input 32x16x16, output 8192
        # x = self.flat(x)
        # # input 8192, output 512
        # x = self.act3(self.fc3(x))
        # x = self.drop3(x)
        # # input 512, output 10
        # x = self.fc4(x)
        return x

