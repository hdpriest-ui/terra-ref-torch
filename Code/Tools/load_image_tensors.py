import numpy as np
import glob
import torch
from matplotlib import pyplot as plt
from pathlib import Path


def load_tensors(source_directory):
    tensor_file_list = glob.glob(str(Path(source_directory, '*.npy')))
    for tensor_file in tensor_file_list:
        tensor_path = Path(source_directory, tensor_file)
        tensor_object = np.load(tensor_path)
        numpy_tensor = np.transpose(tensor_object)
        plt.imshow(numpy_tensor)
        plt.show()
        print(f"Loaded file: {tensor_file}")
    return True


this_directory = Path(__file__).parent
tile_tensors = Path(this_directory, "..", "..", "Data", "128px_samples", "tensors")
warped_tensors = Path(this_directory, "..", "..", "Data", "128px_warped_samples", "tensors")
raw_tensors = Path(this_directory, "..", "..", "Data", "raw_samples", "tensors")
stitched_tensors = Path(this_directory, "..", "..", "Data", "stitched_samples", "tensors")

load_tensors(tile_tensors)
load_tensors(warped_tensors)
load_tensors(raw_tensors)
load_tensors(stitched_tensors)
