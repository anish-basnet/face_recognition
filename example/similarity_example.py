"""
- author: pramod khatiwada
- email : pramodkhatiwada03@gmail.com
"""
import os

import cv2
import numpy as np
import sys

from face_recognition.utils.imgutils import return_image
from face_recognition.utils.metrics import similarity, image_ssim
from scipy import sparse

if __name__ == '__main__':
    path1 = os.path.join('data', 'dataset', 'test', '0000_02176.pgm')
    path2 = os.path.join('data', 'dataset', 'test', '0000_02179.pgm')

    image1 = cv2.cvtColor(np.array(return_image(path1)), cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(np.array(return_image(path2)), cv2.COLOR_RGB2GRAY)

    resized_image_1 = cv2.resize(image1, (64, 64))
    resized_image_2 = cv2.resize(image2, (64, 64))

    result = image_ssim(resized_image_1, resized_image_2)
    print("{} - {} | Similarity score using Structural Similarity Index : {} .".format(path1, path2, result))
