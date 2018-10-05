import logging
import os
import shutil
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from skimage import color

# Define global variables for directories of interest
from traffic_signs.evaluation.evaluation_funcs import performance_accumulation_pixel, performance_evaluation_pixel

# Directory in the root directory where the results will be saved
RESULT_DIR = os.path.join('m1-results', 'weekX', 'test', 'method1')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join('dataset', 'train')
TRAIN_GTS_DIR = os.path.join(TRAIN_DIR, 'gt')
TRAIN_MASKS_DIR = os.path.join(TRAIN_DIR, 'mask')
TEST_DIR = os.path.join('dataset', 'test')


# Logger setup
logging.basicConfig(
    # level=logging.DEBUG,
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


def get_img(folder_dir, img_dir):
    """
    Get numpy array representation of the image.

    :param folder_dir: Folder path
    :param img_dir: Image path
    :return: Numpy array with the RGB pixel values of the image
    """

    img_path = os.path.join(folder_dir, img_dir)
    logger.debug('Getting image {path}'.format(path=img_path))
    return imageio.imread(img_path)


def img_name_to_mask_name(filename):
    """
    Get numpy array representation of the test image.

    :param path: Path of the image
    :return: Numpy array with the RGB pixel values of the image
    """

    return 'mask.{filename}'.format(filename=filename.replace('jpg' or 'png', 'txt'))


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


def confusion_matrix(results_dir, masks_dir):
    result_imgs = os.listdir(results_dir)

    tf_values = np.zeros(4)

    for img_path in result_imgs:
        mask_path = img_name_to_mask_name(img_path)

        tf_val = np.array(performance_accumulation_pixel(
            get_img(results_dir, img_path),
            get_img(masks_dir, mask_path)
        ))

        tf_values += tf_val

    return tf_values.tolist()


def print_confusion_matrix(values):
    values = np.array(values).reshape((2, 2))
    np.set_printoptions(suppress=True)
    print(values / 10000)
    np.set_printoptions(suppress=False)


if __name__ == '__main__':
    # If the directory already exists, delete it
    if os.path.exists(RESULT_DIR):
        shutil.rmtree(RESULT_DIR)

    # Create directory
    os.makedirs(RESULT_DIR)

    # Get list of test images in test directory
    test_images = os.listdir(TRAIN_DIR)
    # test_images = os.listdir(TEST_DIR)

    # Set threshold based on ranges of interest
    ths = np.array([
        [0.0, 0.03],    # Red threshold
        [0.59, 0.62],   # Blue threshold
        [0.98, 1.0]     # Res threshold
    ])

    # Get elapsed time
    t0 = time.time()
    t_frame = 0

    # Iterate over test image paths
    for img_dir in test_images:
        t_frame_0 = 0
        # Get numpy array of the image and convert it to HSV
        img = get_img(TRAIN_DIR, img_dir)
        # img = get_test_img(img_path)
        img_hsv = rgb2hsv(img)
        # img_hsv = ndimage.filters.gaussian_filter(img_hsv, sigma=3)

        # Get the mask of the HSV image
        mask = threshold_image(img_hsv, ths)

        # Create a numpy array where mask values are 255
        final = (255 * mask).astype(np.uint8)

        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(final)
        #
        # plt.show()

        # Save mask as image
        fn, d = save_image(final, RESULT_DIR, img_dir)
        logger.info("'{filename}' saved in '{folder}' folder".format(filename=fn, folder=os.path.join(ROOT_DIR, d)))

        t_frame += time.time() - t_frame_0

    logger.info("%d masks saved in %.3fs (%.3s per frame)" % (len(test_images), time.time() - t0, t_frame / len(test_images)))

    # conf_m = confusion_matrix(RESULT_DIR, TRAIN_MASKS_DIR)
    # print_confusion_matrix(conf_m)
    #
    # print(performance_evaluation_pixel(*conf_m))
