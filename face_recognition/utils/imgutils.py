"""
author : Anish Basnet, Pankaj Bhattrai
Email : anishbasnetworld@gmail.com
"""
from collections import OrderedDict

import numpy as np
from PIL import ImageFile, Image


def return_image(file_path=None):
    """
    This function convert the image into rgb format.
    :param file_path: str -> path of an image
    :return: image -> return image
    """
    if file_path is None:
        return None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        image = Image.open(file_path).convert('RGB')
    except IOError:
        return None
    return image


def get_image_array(files=None):
    """
    This method returns the images vector.
    :param files: list -> list of image path
    :return: OrderedDict -> path with corresponding vector
    """
    if files is None:
        return None
    images_vector = OrderedDict()
    for file in files:
        vector = np.array(return_image(file))
        images_vector[file] = vector
    return images_vector
