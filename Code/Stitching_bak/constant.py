import os
import argparse
from pathlib import Path
from PIL import Image
import glob


# def get_dir(directory):
#     """
#     Create the directory if it does not exist.
#
#     Args:
#         directory (str): The directory path.
#
#     Returns:
#         str: The directory path.
#     """
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     return directory


def parser_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Options to run the network.')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='The device ID of the GPU.')
    parser.add_argument('-i', '--iteration', type=int, default=1,
                        help='Number of iterations (epochs). Default is 1.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('--training_data_directory', type=str, required=True,
                        help='Path to the training data directory.')
    parser.add_argument('--test_data_directory', type=str, required=True,
                        help='Path to the testing data directory.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Path to the output root directory.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to load (optional).')
    parser.add_argument('--checkpoint_number', type=str, default=None,
                        help='Checkpoint number to automatically identify a file to load (optional).')
    parser.add_argument('--vgg19_folder', type=str, default=None,
                        help='Path to the folder containing the VGG19 model (optional).')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer. Default is 0.001.')
    parser.add_argument('--gradient_clip', type=float, default=None,
                        help='Gradient clipping value. Default is None (no clipping).')
    return parser.parse_args()


def determine_image_size(data_directory, num_samples=50):
    """
    Dynamically determine the image size by peeking at the first few images in the training directory.

    Args:
        data_directory (str): Path to the training data directory.
        num_samples (int): Number of images to sample for determining the size.

    Returns:
        tuple: (height, width) of the images.
    """
    input1_dir = os.path.join(data_directory, "input1")
    image_paths = glob.glob(os.path.join(input1_dir, "*.*"))[:num_samples]

    if not image_paths:
        raise ValueError(f"No images found in the directory: {input1_dir}")

    heights, widths = [], []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            heights.append(img.height)
            widths.append(img.width)

    # Return the most common height and width
    return int(sum(heights) / len(heights)), int(sum(widths) / len(widths))


class Const(object):
    """
    A class to define constant variables.
    """
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError(f"Cannot change constant '{name}'.")
        if not name.isupper():
            raise self.ConstCaseError(f"Constant name '{name}' is not all uppercase.")
        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants Information ================>\n'
        for name, value in self.__dict__.items():
            _str += f'\t{name}: {value}\n'
        return _str


# Parse arguments
args = parser_args()
const = Const()

# Input directories
const.TRAINING_DATA_DIRECTORY = args.training_data_directory
const.TEST_DATA_DIRECTORY = args.test_data_directory

# GPU settings
const.GPU = args.gpu

# Training parameters
const.BATCH_SIZE = args.batch_size
const.ITERATION = args.iteration
const.LEARNING_RATE = args.learning_rate
const.GRADIENT_CLIP = args.gradient_clip

# Dynamically determine image size
const.HEIGHT, const.WIDTH = determine_image_size(const.TRAINING_DATA_DIRECTORY)
print("LOADING CONSTANTS FROM STITCHING CONSTANTS")
# Output directories
const.OUTPUT_ROOT = Path(str(args.output_root))
const.SUMMARY_DIR = Path(str(Path(const.OUTPUT_ROOT, "summary")))
const.SNAPSHOT_DIR = Path(str(Path(const.OUTPUT_ROOT, "snapshot")))
const.RESULT_DIR = Path(str(Path(const.OUTPUT_ROOT, "result")))
# Checkpoint settings
const.CHECKPOINT = args.checkpoint
if args.checkpoint_number:
    const.CHECKPOINT_NUMBER = str(int(args.checkpoint_number))

# VGG19 model directory
if args.vgg19_folder:
    const.VGG19_DIR = Path(args.vgg19_folder)

