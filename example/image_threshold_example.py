"""
author : Pankaj Bhattarai
Email : pankajbhattarai100@gmail.com
"""
import os
import pickle
from collections import orderddict

import cv2
import numpy as np

from face_recognition.utils.imgutils import return_image

if __name__ == '__main__':
    path = os.path.join('data', 'dataset', 'test')
    files = os.listdir(path)
    write_path = os.path.join('data', 'processing', 'threshold', 'test.pkl')

    vectors = orderddict
    for file in files:
        bgr_image = cv2.cvtColor(np.array(return_image(file)), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
        vectors[file] = np.array(threshold_image)
        # cv2.waitKey(0)

    with open(write_path, 'wb') as fid:
        pickle.dump(vectors, fid)
        fid.close()
