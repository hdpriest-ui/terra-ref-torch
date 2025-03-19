import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
import numpy as np
import cv2
from collections import OrderedDict
import os
import glob
####
# Lifted shamelessly and with alterations from
# https://github.com/nie-lang/DeepImageStitching-1.0
# (and updated to use pytorch, etc)
#####

def np_load_frame(filename, resize_height, resize_width):
    # image_decoded = decode_image(filename)
    image_decoded = cv2.imread(filename)
    image_decoded = (image_decoded / 127.5) - 1.0
    image_decoded = torch.from_numpy(image_decoded)
    image_decoded = image_decoded.permute(2, 0, 1)
    image_decoded = v2.ToDtype(torch.float32)(image_decoded)
    return image_decoded


class HomographyInputLoader(Dataset):
    def __init__(self, video_folder, resize_height=128, resize_width=128):
        self.dir = video_folder
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self.setup()
        self.labels = HLoader(self.dir)

    def __len__(self):
        video_info_list = list(self.videos.values())
        return len(video_info_list[0]['frame'])

    def __getitem__(self, idx):
        # video_name = list(self.videos.keys())[idx]
        # assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        batch = []
        video_info_list = list(self.videos.values())
        # the 'video' here is the entire set of images in 'input1' and 'input2'
        # so, we are essentially slicing out a single from input 1, and the same single frame from input2
        for i in range(0, 2):
            # print(f"Using path for HIL: {video_info_list[i]['frame'][idx]}")
            image = np_load_frame(video_info_list[i]['frame'][idx], self._resize_height, self._resize_width)
            batch.append(image)

        label = self.labels.__getitem__(idx)
        return [batch[0], batch[1]], label

    def setup(self):
        videos = [os.path.normpath(i) for i in glob.glob(os.path.join(self.dir, '*'))]
        for video in videos:
            path_sep = os.path.sep
            video_name = video.split(path_sep)[-1]
            if video_name == 'input1' or video_name == 'input2':
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

        print(self.videos.keys())


def np_load_h(filename):
    shift = np.load(filename)
    return shift


class HLoader(object):
    def __init__(self, video_folder):
        self.dir = video_folder
        self.videos = OrderedDict()
        self.setup()

    def __getitem__(self, idx):
        # assert video_name in self.videos.keys(), 'video = {} is not in {}!'.format(video_name, self.videos.keys())
        batch = []
        video_info_list = list(self.videos.values())
        for i in range(0, 1):
            image = np_load_h(video_info_list[i]['frame'][idx])
            # print(f"Using path for label: {video_info_list[i]['frame'][idx]}")
            batch.append(image)
        return np.concatenate(batch, axis=1)

    def setup(self):
        videos = [os.path.normpath(i) for i in glob.glob(os.path.join(self.dir, '*'))]
        for video in videos:
            path_sep = os.path.sep
            video_name = video.split(path_sep)[-1]
            if video_name == 'shift':
                self.videos[video_name] = {}
                self.videos[video_name]['path'] = video
                self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.npy'))
                self.videos[video_name]['frame'].sort()
                self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
        print(self.videos.keys())

    def get_video_clips(self, index):
        batch = []
        video_info_list = list(self.videos.values())
        for i in range(0, 1):
            image = np_load_h(video_info_list[i]['frame'][index])
            batch.append(image)
        return np.concatenate(batch, axis=1)


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




