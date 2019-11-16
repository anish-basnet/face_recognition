"""
author : Anish Basnet, Pankaj Bhattrai
Email : anishbasnetworld@gmail.com
"""


import os

import cv2
import numpy as np

from face_recognition.utils.imgutils import return_image

if __name__ == '__main__':
    path = os.path.join('data', 'dataset', 'test', '0000_02176.pgm')
    image = cv2.cvtColor(np.array(return_image(path)), cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
    cv2.waitKey(0)
