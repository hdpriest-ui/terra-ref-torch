import os
from random import random, seed

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from constant import const
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from models import HomographyEstimator
from utilities import HomographyInputLoader

# Device will determine whether to run the training on GPU or CPU.
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

def device():
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_device('cuda')
        the_device = torch.device('cuda')
        return the_device
    else:
        the_device = torch.device('cpu')
        return the_device

this_device = device()
print(f"Running in context: {this_device}")

train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER
batch_size = const.BATCH_SIZE # param
iterations = const.ITERATIONS
learning_rate_set = const.LEARNING_RATE  # param
epsilon_set = const.EPSILON # param
gradient_clip_level = const.GRADIENT_CLIP
train_dataset_tile_height = 128 # param
train_dataset_tile_width = 128 # param

# utilized for testing/dev
remove_randomness = False
if remove_randomness:
    torch.manual_seed(12345)
    np.random.seed(12345)
    seed(12345)
    torch.backends.cudnn.benchmark = False

# torch.backends.cudnn.enabled = False

def main():
    ###################
    #  DATA LOGISTICS #
    ###################
    # Create Training dataset
    train_dataset = HomographyInputLoader(train_folder, resize_height=train_dataset_tile_height, resize_width=train_dataset_tile_width)
    test_dataset = HomographyInputLoader(test_folder, resize_height=train_dataset_tile_height, resize_width=train_dataset_tile_width)

    # Make data loaders

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device()), num_workers=3, prefetch_factor=64)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device()), num_workers=3, prefetch_factor=64)

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, prefetch_factor=64)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=3, prefetch_factor=64)


    ###################
    #    TRAINING     #
    ###################

    HomographyModel = HomographyEstimator(batch_size=batch_size)
    HomographyModel.to(device())

    # criterion = nn.MSELoss()
    def criterion(output, label):
        # i can't even describe how much this shouldn't matter.
        interstitial = output - label
        interstitial1 = interstitial ** 2
        interstitial2 = torch.abs(interstitial1)
        this_loss = torch.mean(interstitial2)
        return this_loss

    # Set optimizer with optimizer
    optimizer = torch.optim.Adam(HomographyModel.parameters(), lr=learning_rate_set, eps=epsilon_set, weight_decay=1e-5)
    writer = SummaryWriter(log_dir=str(const.SUMMARY_DIR))

    total_step = len(train_dataloader)
    iter_save_number = 25000
    iter_update_number = 1000
    epoch_max = 5000
    current_iter = 0
    HomographyModel.train()
    w_previous = None
    # print(HomographyModel.state_dict())
    for epoch in range(epoch_max):
        if current_iter >= const.ITERATIONS:
            break
        print(f"Beginning epoch: {epoch}")
        # print(list(HomographyModel.parameters()))
        for i, (images, labels) in enumerate(train_dataloader):
            # Move tensors to the configured device
            if current_iter >= const.ITERATIONS:
                break
            image_a = images[1]
            image_b = images[0]
            # labels = torch.mean(input=labels, dim=2, keepdim=False)
            image_a = image_a.to(device())
            image_b = image_b.to(device())
            labels = labels.to(device())

            # optimizer.zero_grad() #### ADDED THIS I HAVE NO IDEA IF IT RIGHT
            # Forward pass
            outputs, training_warp_gt = HomographyModel(image_a, image_b, labels)

            labels = torch.mean(labels, dim=2)
            loss = criterion(outputs.flatten(), labels.flatten())

            # Backward and optimize
                 ### REMOVED THIS
            loss.backward()

            # w = HomographyModel.conv1._parameters['weight'].detach()
            # if w_previous is None:
            #     w_previous = w.clone()
            # print(f"Conv1: {(w-w_previous).abs().sum()}")
            # w_previous = w.clone()
            # gradient clipping present in nie approach to (by their annotation) avoid dividing by zero.
            # necessary here?
            # for param in HomographyModel.parameters():
            #     print(param.grad)
            # for p in HomographyModel.parameters():
            #     print('gradient param:{}'.format(p.grad))
            # torch.nn.utils.clip_grad_norm_(HomographyModel.parameters(), gradient_clip_level)
            # torch.nn.utils.clip_grad_value_(HomographyModel.parameters(), gradient_clip_level)
            optimizer.step()
            optimizer.zero_grad()

            if (current_iter % iter_update_number == 0) or (current_iter == const.ITERATIONS):
                print(f"Iteration [{current_iter}/{const.ITERATIONS}] :: Loss: {loss.item()}")
                writer.add_scalar('Loss/train', loss.item(), current_iter)

            if (current_iter % iter_save_number == 0) or (current_iter == const.ITERATIONS):
                print(f"Saving model at iteration: {current_iter}")
                torch.save({
                    'iteration': current_iter,
                    'model_state_dict': HomographyModel.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, str(Path(const.SNAPSHOT_DIR, "homography_checkpoint.pth")))
            current_iter = current_iter + 1

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()