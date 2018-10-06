# -*- coding: utf-8 -*-

# Built-in modules
import logging
import os

# 3rd party modules
import imageio
import numpy as np

from skimage import color

# Local modules
from traffic_signs.evaluation.evaluation_funcs import performance_accumulation_pixel, performance_evaluation_pixel


# Logger setup
logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def get_files_from_dir(directory):
    """
    Get only files from directory.

    :param directory: Directory path
    :return: List of files in directory
    """

    logger.debug("Getting files in '{path}'".format(path=os.path.abspath(directory)))
    l = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    logger.debug("Retrieving {num_files} files from '{path}'".format(num_files=len(l), path=os.path.abspath(directory)))

    return l


def rgb2hsv(img):
    """
    Convert image from RGB to HSV,

    :param img: Numpy array with the RGB pixel values of the image
    :return: Numpy array with the HSV pixel values of the image
    """

    logger.debug("Converting image from RGB to HSV")
    return color.rgb2hsv(img)


def hsv2rgb(img):
    """
    Convert image from HSV to RGB.

    :param img: Numpy array with the HSV pixel values of the image
    :return: Numpy array with the RGB pixel values of the image
    """

    logger.debug("Converting image from HSV to RGB")
    return color.hsv2rgb(img)


def get_img(folder_dir, img_dir):
    """
    Get numpy array representation of the image.

    :param folder_dir: Folder path
    :param img_dir: Image path
    :return: Numpy array with the RGB pixel values of the image
    """

    img_path = os.path.join(folder_dir, img_dir)
    logger.debug("Getting image '{path}'".format(path=img_path))

    return imageio.imread(img_path)


def img_name_to_mask_name(filename):
    """
    Get numpy array representation of the test image.

    :param filename: Image name
    :return: Numpy array with the RGB pixel values of the image
    """

    mask_name = 'mask.{filename}'.format(filename=filename.replace('jpg' or 'png', 'txt'))
    logger.debug("'{filename}' converted to '{mask_name}".format(filename=filename, mask_name=mask_name))

    return mask_name


def threshold_image(img, ths, channel=0):
    """
    Get the mask of a image for the pixels that are between a set threshold values.

    :param img: Numpy array representation of the image
    :param ths: List of tuples of thresholds
    :param channel: Channel of interest of the image. Default 0
    :return: Numpy array with the mask for values between threshold values
    """

    logger.debug("Getting image pixels of channel {channel} that are between the thresholds".format(channel=channel))
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
    img_path = os.path.join(directory, filename)

    logger.debug("Saving image in {path}".format(path=img_path))

    # Save image
    imageio.imwrite(img_path, img)

    return filename, directory


def confusion_matrix(results_dir, masks_dir):
    """
    Calculate confusion matrix.

    :param results_dir: Directory with calculated masks
    :param masks_dir: Ground truth masks
    :return: Confusion matrix
    """

    # Getting calculated masks
    result_imgs = get_files_from_dir(results_dir)

    # List with values TP, FP, FN, TN
    tf_values = np.zeros(4)

    # Iterate over image paths
    for img_path in result_imgs:
        # Convert image path to mask path
        mask_path = img_name_to_mask_name(img_path)

        # Compute perfomance measures
        tf_val = np.array(performance_accumulation_pixel(
            get_img(results_dir, img_path),
            get_img(masks_dir, mask_path)
        ))

        # Add them up
        tf_values += tf_val

    return tf_values.tolist()


def print_confusion_matrix(values):
    """
    Print a human-readable representation of a confusion matrix.

    :param values: Values of the confusion matrix
    :return: Nothing
    """

    # Reshape matrix values
    values = np.array(values).reshape((2, 2))

    # Turn off scientific notation for float values
    np.set_printoptions(suppress=True)
    print(values)
    np.set_printoptions(suppress=False)


if __name__ == '__main__':
    print(get_files_from_dir('.'))
