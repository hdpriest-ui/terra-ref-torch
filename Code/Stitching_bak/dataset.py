import glob
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
from collections import OrderedDict


def np_load_frame(filename, resize_height, resize_width):
    """
    Load and preprocess an image frame.

    Args:
        filename (str): Path to the image file.
        resize_height (int): Height to resize the image to.
        resize_width (int): Width to resize the image to.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image_decoded = cv2.imread(filename)
    image_decoded = (image_decoded / 127.5) - 1.0  # Normalize to [-1, 1]
    image_decoded = cv2.resize(image_decoded, (resize_width, resize_height))
    image_decoded = torch.from_numpy(image_decoded).permute(2, 0, 1)  # HWC -> CHW
    return image_decoded.float()


def np_load_h(filename):
    """
    Load a homography shift label.

    Args:
        filename (str): Path to the `.npy` file.

    Returns:
        np.ndarray: Homography shift array.
    """
    try:
        label = np.load(filename)
    except Exception as e:
        raise ValueError(f"Error loading label file {filename}: {e}")
    return label


class StitchingDataset(Dataset):
    def __init__(self, data_directory, resize_height=128, resize_width=128):
        """
        Dataset for stitching tasks.

        Args:
            data_directory (str): Path to the directory containing the dataset.
            resize_height (int): Height of the input images.
            resize_width (int): Width of the input images.
        """
        self.dir = data_directory
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self.setup()

    def setup(self):
        """
        Set up the dataset by organizing paths to input frames and labels.
        """
        videos = [os.path.normpath(i) for i in os.listdir(self.dir)]
        for video in videos:
            video_path = os.path.join(self.dir, video)
            if os.path.isdir(video_path):
                if video == 'input1' or video == 'input2':
                    self.videos[video] = {
                        'path': video_path,
                        'frame': sorted(glob.glob(os.path.join(video_path, '*.jpg')))
                    }
                elif video == 'shift':
                    self.videos[video] = {
                        'path': video_path,
                        'frame': sorted(glob.glob(os.path.join(video_path, '*.npy')))
                    }

        required_dirs = ['input1', 'input2', 'shift']
        for req_dir in required_dirs:
            if req_dir not in self.videos:
                raise FileNotFoundError(f"Required directory '{req_dir}' not found in {self.dir}")

        # Ensure all directories have the same number of frames
        input1_len = len(self.videos['input1']['frame'])
        input2_len = len(self.videos['input2']['frame'])
        shift_len = len(self.videos['shift']['frame'])
        assert input1_len == input2_len == shift_len, \
            f"Mismatch in frame counts: input1={input1_len}, input2={input2_len}, shift={shift_len}"

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.videos['input1']['frame'])

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: ([input1, input2], label) where inputs are tensors and label is a numpy array.
        """
        # Load input frames
        input1_path = self.videos['input1']['frame'][idx]
        input2_path = self.videos['input2']['frame'][idx]
        input1 = np_load_frame(input1_path, self._resize_height, self._resize_width)
        input2 = np_load_frame(input2_path, self._resize_height, self._resize_width)

        # Load label
        label_path = self.videos['shift']['frame'][idx]
        label = np_load_h(label_path)

        return [input1, input2], label