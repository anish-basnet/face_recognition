"""
author : Anish Basnet, Pankaj Bhattrai
Email : anishbasnetworld@gmail.com
"""

from PIL import ImageFile, Image


def return_image(file_path):
    """
    This function convert the image into rgb format.
    :param file_path: str -> path of an image
    :return: image -> return image
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(file_path).convert('RGB')
    return image
