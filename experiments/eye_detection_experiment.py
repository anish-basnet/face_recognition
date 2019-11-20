import os

import cv2
import numpy as np

from face_recognition.utils.imgutils import return_image

if __name__ == '__main__':
    path = os.path.join('data', 'dataset', 'test', '0000_02176.pgm')
    image = cv2.cvtColor(np.array(return_image(path)), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
    cv2.imshow('img', img)
    cv2.imshow('image', image)
    cv2.waitKey(0)
