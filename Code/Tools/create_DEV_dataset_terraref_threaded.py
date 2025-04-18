from tempfile import TemporaryFile
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

# adapted from https://github.com/nie-lang/DeepImageStitching-1.0

def setup_logging():
    logger = logging.getLogger()  # Gets the root logger
    fh = logging.FileHandler('image_dataset.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


#Calculate cross product
def cross(a, b, c):
    ans = (b[0] - a[0])*(c[1] - b[1]) - (b[1] - a[1])*(c[0] - b[0])
    return ans

#Check whether the quadrilateral is convex.
#If it is convex, return 1
def checkShape(a,b,c,d):
    x1 = cross(a,b,c)
    x2 = cross(b,c,d)
    x3 = cross(c,d,a)
    x4 = cross(d,a,b)

    if (x1<0 and x2<0 and x3<0 and x4<0) or (x1>0 and x2>0 and x3>0 and x4>0) :
        return 1
    else:
        print('not convex')
        return 0


#Judge whether the pixel is within the label, and set it to black if it is not
def inLabel(row, col, src_input, dst):
    if (row >= src_input[0][1]) and (row <= src_input[2][1]) and (col >= src_input[0][0]) and (col <= src_input[1][0]) :
        return 1
    else : 
        #Only handle the case of convex quadrilaterals. As for the concave quadrilateral, regenerated it until it is convex.
        a = (dst[1][0] - dst[0][0])*(row - dst[0][1]) - (dst[1][1] - dst[0][1])*(col - dst[0][0])
        b = (dst[3][0] - dst[1][0])*(row - dst[1][1]) - (dst[3][1] - dst[1][1])*(col - dst[1][0])
        c = (dst[2][0] - dst[3][0])*(row - dst[3][1]) - (dst[2][1] - dst[3][1])*(col - dst[3][0])
        d = (dst[0][0] - dst[2][0])*(row - dst[2][1]) - (dst[0][1] - dst[2][1])*(col - dst[2][0])
        if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0) :
            return 1
        else :
            return 0


def dynamic_crop(image_obj):
    y_nonzero, x_nonzero, _ = np.nonzero(image_obj)
    return image_obj.crop((np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero)))


def other_dynamic_crop(img_file_pref, image_suffix, image_obj):
    gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image_obj[y:y + h, x:x + w]
    cv2.imwrite(img_file_pref + '_cropped' + image_suffix, crop)
    return crop

# Load a random image from the dataset
def load_random_image(path_source, size):
    #The size of the randomly sampled image must be greater than width*height
    do_not_use_image = {}

    image_suffix = '.tif'
    string_path = rd.choice(glob.glob(os.path.join(path_source, '*'+image_suffix)))
    logging.info(f"Loading random image: {string_path}")
    while True:
        # if the image, after cropping, is too small, we want to avoid trying to use it again
        # its slow and wasteful to reload it, re crop it, test it, etc.
        # just don't select it in the future.
        if not string_path in do_not_use_image:
            break
        string_path = rd.choice(glob.glob(os.path.join(path_source, '*' + image_suffix)))
        logging.info(f"Loading random image: {string_path}")

    img_path = Path(string_path)
    img = Image.open(img_path)
    img = dynamic_crop(img)
    while True:
        #print(img.size)
        if img.size[0]>=size[0] and img.size[1]>=size[1] :
            break
        else:
            do_not_use_image[string_path] = True
            cropped_output = os.path.join(path_source, "../trimmed")
            img.save(os.path.join(cropped_output, img_path.stem + '_cropped' + image_suffix))
            print(f"Skipping image {img_path} - too small after crop.\n")
            print(f"required: {size[0]}x{size[1]}\n")
            print(f"actual: {img.size[0]}x{img.size[1]}\n")
        img_path = Path(rd.choice(glob.glob(os.path.join(path_source, '*'+image_suffix))))
        img = Image.open(img_path)
        img = dynamic_crop(img)
    #print('bingo')
    img_grey = img.resize(size)                
    img_data = np.asarray(img_grey)
    #imggg = Image.fromarray(img_data.astype('uint8')).convert('RGB')
    #imggg.show()
    return img_data


def save_to_file(index, image1, image2, label, path_dest, gt_4pt_shift):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    input1_path = Path(path_dest,'input1')
    input2_path = Path(path_dest, 'input2')
    label_path = Path(path_dest, 'label')
    pt4_shift_path = Path(path_dest, 'shift')
    
    if not os.path.exists(input1_path):
        os.makedirs(input1_path)
    if not os.path.exists(input2_path):
        os.makedirs(input2_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    if not os.path.exists(pt4_shift_path):
        os.makedirs(pt4_shift_path)

    input1_path = Path(path_dest,'input1' ,index + '.jpg')
    input2_path = Path(path_dest, 'input2', index + '.jpg')
    label_path = Path(path_dest, 'label', index + '.jpg')
    pt4_shift_path = Path(path_dest, 'shift', index + '.npy')
    image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
    image2 = Image.fromarray(image2.astype('uint8')).convert('RGB')
    label = Image.fromarray(label.astype('uint8')).convert('RGB')
    image1.save(input1_path)
    image2.save(input2_path)
    label.save(label_path)
    np.save(str(pt4_shift_path), gt_4pt_shift)


def process_image(argument_list):
    # this_count, path_source, path_dest, rho, height, width, data, box, overlap
    this_count, path_source, path_dest, rho, height, width, data, box, overlap = argument_list
    logging.info(f"starting {this_count}")
    # try:
    img = load_random_image(path_source, [width, height]).astype(np.uint16)
    # except e:
    #     # lazy
    #     img = load_random_image(path_source, [width, height]).astype(np.uint16)

    # define parameters
    # src_input1 = np.empty([4, 2], dtype=np.uint8)
    src_input1 = np.zeros([4, 2])
    src_input2 = np.zeros([4, 2])
    dst = np.zeros([4, 2])

    # Upper left
    src_input1[0][0] = int(width / 2 - box / 2)
    src_input1[0][1] = int(height / 2 - box / 2)
    # Upper right
    src_input1[1][0] = src_input1[0][0] + box
    src_input1[1][1] = src_input1[0][1]
    # Lower left
    src_input1[2][0] = src_input1[0][0]
    src_input1[2][1] = src_input1[0][1] + box
    # Lower right
    src_input1[3][0] = src_input1[1][0]
    src_input1[3][1] = src_input1[2][1]
    # print(src_input1)

    # The translation of input2 relative to input1
    box_x_off = rd.randint(int(box * (overlap - 1)), int(box * (1 - overlap)))
    box_y_off = rd.randint(int(box * (overlap - 1)), int(box * (1 - overlap)))
    # Upper left
    src_input2[0][0] = src_input1[0][0] + box_x_off
    src_input2[0][1] = src_input1[0][1] + box_y_off
    # Upper righ
    src_input2[1][0] = src_input1[1][0] + box_x_off
    src_input2[1][1] = src_input1[1][1] + box_y_off
    # Lower left
    src_input2[2][0] = src_input1[2][0] + box_x_off
    src_input2[2][1] = src_input1[2][1] + box_y_off
    # Lower right
    src_input2[3][0] = src_input1[3][0] + box_x_off
    src_input2[3][1] = src_input1[3][1] + box_y_off
    # print(src_input2)

    offset = np.empty(8, dtype=np.int8)
    # Generate offsets:
    # The position of each vertex after the coordinate perturbation
    while True:
        for j in range(8):
            offset[j] = rd.randint(-rho, rho)
        # Upper left
        dst[0][0] = src_input2[0][0] + offset[0]
        dst[0][1] = src_input2[0][1] + offset[1]
        # Upper righ
        dst[1][0] = src_input2[1][0] + offset[2]
        dst[1][1] = src_input2[1][1] + offset[3]
        # Lower left
        dst[2][0] = src_input2[2][0] + offset[4]
        dst[2][1] = src_input2[2][1] + offset[5]
        # Lower right
        dst[3][0] = src_input2[3][0] + offset[6]
        dst[3][1] = src_input2[3][1] + offset[7]
        # print(dst)
        if checkShape(dst[0], dst[1], dst[3], dst[2]) == 1:
            break

    source = np.zeros([4, 2])
    target = np.zeros([4, 2])
    source[0][0] = 0
    source[0][1] = 0
    source[1][0] = source[0][0] + box
    source[1][1] = source[0][1]
    source[2][0] = source[0][0]
    source[2][1] = source[0][1] + box
    source[3][0] = source[1][0]
    source[3][1] = source[2][1]
    target[0][0] = dst[0][0] - src_input1[0][0]
    target[0][1] = dst[0][1] - src_input1[0][1]
    target[1][0] = dst[1][0] - src_input1[0][0]
    target[1][1] = dst[1][1] - src_input1[0][1]
    target[2][0] = dst[2][0] - src_input1[0][0]
    target[2][1] = dst[2][1] - src_input1[0][1]
    target[3][0] = dst[3][0] - src_input1[0][0]
    target[3][1] = dst[3][1] - src_input1[0][1]
    Hab, status = cv2.findHomography(source, target)

    # Generate the shift
    gt_4pt_shift = np.zeros((8, 1), dtype=np.float32)
    for i in range(4):
        gt_4pt_shift[2 * i] = target[i][0] - source[i][0]
        gt_4pt_shift[2 * i + 1] = target[i][1] - source[i][1]

    h, status = cv2.findHomography(dst, src_input2)
    img_warped = np.asarray(cv2.warpPerspective(img, h, (width, height))).astype(np.uint8)

    # Generate the label
    label = img.copy()
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            if not (inLabel(row, col, src_input1, dst)):
                label[row][col] = 0

    x = int(src_input1[0][0] - overlap * box - rho)
    y = int(src_input1[0][1] - overlap * box - rho)
    label = label[y + 1:int(y + box + 2 * overlap * box + 2 * rho - 1),
            x + 1:int(x + box + 2 * overlap * box + 2 * rho - 1)]

    # Generate input1
    x1 = int(src_input1[0][0])
    y1 = int(src_input1[0][1])
    image1 = img[y1:y1 + box, x1:x1 + box]

    # Generate input2
    x2 = int(src_input2[0][0])
    y2 = int(src_input2[0][1])
    image2 = img_warped[y2:y2 + box, x2:x2 + box, ...]
    local_count = this_count
    for i in range(200):
        local_count += 1
        save_to_file(str(local_count + 1).zfill(6), image1, image2, label, path_dest, gt_4pt_shift)

    print("done with generating 200 copies of the same test image. Good idea?")
    return True
    logging.info(this_count + 1)

# Function to generate dataset
def generate_dataset(path_source, path_dest, rho, height, width, data, box, overlap):

    arg_array = []
    for count in range(0, data):
        this_arg_entry = [count, path_source, path_dest, rho, height, width, data, box, overlap]
        arg_array.append(this_arg_entry)

    #load row image
    try:
        # for entry in arg_array:
        process_image(arg_array[0])
        # with Pool(14) as p:
        #     p.map(process_image, arg_array)
    except IndexError as e:
        print(e)
        print("finished?")
        exit(0)
        
        
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process input parameters for the script.")
    parser.add_argument("--raw", type=str, required=True, help="Path to directory of raw images to create datasets from.")
    parser.add_argument("--training", type=str, required=True, help="destination path for training data.")
    parser.add_argument("--test", type=str, required=False, help="destination path for test data", default="")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    setup_logging()
    raw_image_path = Path(args.raw)
    training_image_path = Path(args.training)
    testing_image_path = Path(args.test)

    # raw_image_path = "C:\\Users\\hdpriest\\Large_datasets\\terra-ref\\RGB\\raw"
    box_size = int(128)
    height = int(360)
    width = int(480)
    overlap_rate = 0.5
    rho = int(box_size/5.0)

    ### generate training dataset
    print("Training dataset...")
    dataset_size = 5000
    # generate_image_path =  "C:\\Users\\hdpriest\\Large_datasets\\terra-ref\\RGB\\training"
    generate_dataset(raw_image_path, training_image_path, rho, height, width, dataset_size, box_size, overlap_rate)
    ### generate testing dataset
    print("Testing dataset...")
    dataset_size = 500
    # generate_image_path = "C:\\Users\\hdpriest\\Large_datasets\\terra-ref\\RGB\\testing"
    generate_dataset(raw_image_path, testing_image_path, rho, height, width, dataset_size, box_size, overlap_rate)


