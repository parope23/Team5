import logging
import os
import shutil
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage import color

# Define global variables for directories of interest
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_GTS_DIR = os.path.join(TRAIN_DIR, 'gt')
TRAIN_MASKS_DIR = os.path.join(TRAIN_DIR, 'mask')
TEST_DIR = os.path.join('dataset', 'test')


# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def rgb2hsv(img):
    """
    Convert image from RGB to HSV,

    :param img: Numpy array with the RGB pixel values of the image
    :return: Numpy array with the HSV pixel values of the image
    """

    return color.rgb2hsv(img)


def hsv2rgb(img):
    """
    Convert image from HSV to RGB.

    :param img: Numpy array with the HSV pixel values of the image
    :return: Numpy array with the RGB pixel values of the image
    """

    return color.hsv2rgb(img)


def get_test_img(path):
    """
    Get numpy array representation of the test image.

    :param path: Path of the image
    :return: Numpy array with the RGB pixel values of the image
    """

    return imageio.imread(os.path.join(TEST_DIR, path))


def threshold_image(img, ths, channel=0):
    """
    Get the mask of a image for the pixels that are between a set threshold values.

    :param img: Numpy array representation of the image
    :param ths: List of tuples of thresholds
    :param channel: Channel of interest of the image. Default 0
    :return: Numpy array with the mask for values between threshold values
    """

    # Create a mask with the same shape of the image filled with 'False'
    mask = np.full(img.shape[:2], False)
    # Get the channel of the image
    c = img[..., channel]

    # Iterate over thresholds
    for th in ths:
        # Add the values between the thresholds as 'True' values to the mask
        mask += np.logical_and(th[0] < c, c < th[1])

    return mask


def save_image(img, directory, name, ext='png'):
    """
    Save a numpy array as an image.

    :param img: Numpy array with the pixel values of the image
    :param directory: Folder where the image will be saved
    :param name: Filename of the image
    :param ext: Extension of de image file
    :return: Filename of the image and folder where it was saved
    """

    # Get filename without extension and the new one
    filename = '.'.join(name.split('.')[:-1])
    filename = '{name}.{ext}'.format(name=filename, ext=ext)
    # Save image
    imageio.imwrite(os.path.join(directory, filename), img)

    return filename, directory


if __name__ == '__main__':
    # Directory in the root directory where the results will be saved
    result_path = 'results'

    # If the directory already exists, delete it
    if os.path.exists(result_path):
        shutil.rmtree(result_path)

    # Create directory
    os.mkdir(result_path)

    # Get list of test images in test directory
    test_images = os.listdir(TEST_DIR)

    # Set threshold based on ranges of interest
    ths = np.array([
        [0.1, 0.2],     # Red threshold
        [0.5, 0.55],    # Blue threshold
        [0.7, 0.8]      # White threshold
    ])

    # Get elapsed time
    t0 = time.time()

    # Iterate over test image paths
    for img_path in test_images:
        # Get numpy array of the image and convert it to HSV
        img = get_test_img(img_path)
        img_hsv = rgb2hsv(img)

        # Get the mask of the HSV image
        mask = threshold_image(img_hsv, ths)

        # Create a numpy array where mask values are 255
        final = (255 * mask).astype(np.uint8)

        # Save mask as image
        fn, d = save_image(final, result_path, img_path)
        logger.info("'{filename}' saved in '{folder}' folder".format(filename=fn, folder=os.path.join(ROOT_DIR, d)))

    logger.info("%d masks saved in %.3fs" % (len(test_images), time.time() - t0))