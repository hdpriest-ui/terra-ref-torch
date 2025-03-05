import os
from random import random, seed

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional
from constant import const
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models import HomographyEstimator
from utilities import HomographyInputLoader

# C:\Users\hdpriest\Large_datasets\terra-ref\RGB

# Device will determine whether to run the training on GPU or CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

def device():
    the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return the_device

this_device = device()
print(f"Running in context: {this_device}")

train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER
batch_size = const.BATCH_SIZE
iterations = const.ITERATIONS
train_dataset_tile_height = 128
train_dataset_tile_width = 128

# utilize for testing/dev
remove_randomness = False
if remove_randomness:
    torch.manual_seed(12345)
    np.random.seed(12345)
    seed(12345)
    torch.backends.cudnn.benchmark = False

# C+P with updates from Digital Ocean
###################
#  DATA LOGISTICS #
###################
# all_transforms = transforms.Compose([transforms.Resize((32,32)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
#                                                           std=[0.2023, 0.1994, 0.2010])
#                                      ])
# Create Training dataset
train_dataset = HomographyInputLoader(train_folder, resize_height=train_dataset_tile_height, resize_width=train_dataset_tile_width)
test_dataset = HomographyInputLoader(test_folder, resize_height=train_dataset_tile_height, resize_width=train_dataset_tile_width)

# Make data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


###################
#    TRAINING     #
###################

HomographyModel = HomographyEstimator(batch_size=batch_size)
HomographyModel.to(device())
# Set Loss function with criterion
# h_loss = tf.reduce_mean(input_tensor=tf.abs((train_outputs - train_h) ** 2))
criterion = nn.MSELoss()

# g_lrate = tf.compat.v1.train.piecewise_constant(g_step, boundaries=[500000], values=[0.00005, 0.000005])

# Set optimizer with optimizer
learning_rate_set=0.00005
optimizer = torch.optim.Adam(HomographyModel.parameters(), lr=learning_rate_set)
writer = SummaryWriter(log_dir=str(const.SUMMARY_DIR))

total_step = len(train_dataloader)
epoch_save_number = 500
epoch_update_number = 100
# iter_save_number = 500
iter_save_number = int(const.ITERATIONS / 2)
iter_update_number = 100
epoch_max = 10
current_iter = 0
for epoch in range(epoch_max):
    if current_iter >= const.ITERATIONS:
        break
    print(f"Beginning epoch: {epoch}")
    for i, (images, labels) in enumerate(train_dataloader):
        # Move tensors to the configured device
        if current_iter >= const.ITERATIONS:
            break
        image_a = images[0]
        image_b = images[1]
        labels = torch.mean(input=labels, dim=2, keepdim=False)
        image_a = image_a.to(device())
        image_b = image_b.to(device())
        labels = labels.to(device())

        # Forward pass
        outputs = HomographyModel(image_a, image_b)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if current_iter % iter_update_number == 0:
            print(f"Iteration [{current_iter}/{const.ITERATIONS}] :: Loss: {loss.item()}")
            writer.add_scalar('Loss/train', loss.item(), current_iter)

        if current_iter % iter_save_number == 0:
            print(f"Saving model at iteration: {current_iter}")
            torch.save({
                'iteration': current_iter,
                'model_state_dict': HomographyModel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, str(Path(const.SNAPSHOT_DIR, "homography_checkpoint.pth")))
        current_iter = current_iter + 1
# const.SUMMARY_DIR = Path(get_dir(Path(const.OUTPUT_ROOT, "summary", "homography")))
# const.SNAPSHOT_DIR = Path(get_dir(Path(const.OUTPUT_ROOT, "snapshot", "homography")))