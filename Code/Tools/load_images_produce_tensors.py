import logging
from PIL import Image
import numpy as np
import glob
import os.path
import random as rd
import cv2
from pathlib import Path
import argparse
from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2, ToTensor

def setup_logging():
    logger = logging.getLogger()  # Gets the root logger
    fh = logging.FileHandler('image_dataset.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


# def save_to_file(index, image1, image2, label, path_dest, gt_4pt_shift):
#     if not os.path.exists(path_dest):
#         os.makedirs(path_dest)
#     input1_path = Path(path_dest,'input1')
#     input2_path = Path(path_dest, 'input2')
#     label_path = Path(path_dest, 'label')
#     pt4_shift_path = Path(path_dest, 'shift')
#
#     if not os.path.exists(input1_path):
#         os.makedirs(input1_path)
#     if not os.path.exists(input2_path):
#         os.makedirs(input2_path)
#     if not os.path.exists(label_path):
#         os.makedirs(label_path)
#     if not os.path.exists(pt4_shift_path):
#         os.makedirs(pt4_shift_path)
#
#     input1_path = Path(path_dest,'input1' ,index + '.jpg')
#     input2_path = Path(path_dest, 'input2', index + '.jpg')
#     label_path = Path(path_dest, 'label', index + '.jpg')
#     pt4_shift_path = Path(path_dest, 'shift', index + '.npy')
#     image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
#     image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
#     label = Image.fromarray(label.astype('uint8')).convert('RGB')
#     image1.save(input1_path)
#     image2.save(input2_path)
#     label.save(label_path)
#     np.save(str(pt4_shift_path), gt_4pt_shift)

def np_load_frame(filename, resize_height, resize_width):
    # print(f"Loading file: {filename}")
    image_decoded = decode_image(filename)
    # image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    # image_resized = image_decoded.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0
    image_decoded = (image_decoded / 127.5) - 1.0
    image_decoded = v2.ToDtype(torch.float32)(image_decoded)
    return image_decoded

# Function to generate dataset
def generate_dataset(path_source, path_dest):

    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    image_suffixes = [".jpg", ".tif"]
    image_file_list = glob.glob(os.path.join(path_source, '*'))
    for image_file in image_file_list:
        image_path = Path(path_source, image_file)
        if image_path.suffix in image_suffixes:
            image_decoded = []
            if image_path.suffix == ".tif":
                # Open the TIFF image
                image = Image.open(image_path)
                # Convert the image to a tensor
                convert_tensor = ToTensor()
                image_decoded = convert_tensor(image)
            else:
                image_decoded = decode_image(image_path)
                # image_decoded = (image_decoded / 127.5) - 1.0
                # image_decoded = v2.ToDtype(torch.float32)(image_decoded)
            image_stem = image_path.stem
            output_path = Path(path_dest, image_stem + "_tensor.npy")
            np.save(str(output_path), image_decoded)
    return True

        
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input parameters for the script.")
    parser.add_argument("--raw", type=str, required=True, help="Path to directory of raw images to create datasets from.")
    parser.add_argument("--output", type=str, required=True, help="directory to place output tensors.")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    setup_logging()
    raw_image_path = Path(args.raw)
    output_image_path = Path(args.output)

    ### generate training dataset
    print("Loading files for tensors...")
    # generate_image_path =  "C:\\Users\\hdpriest\\Large_datasets\\terra-ref\\RGB\\training"
    generate_dataset(raw_image_path, output_image_path)



