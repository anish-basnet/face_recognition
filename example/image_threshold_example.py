"""
author : Pankaj Bhattarai
Email : pankajbhattarai100@gmail.com
"""
import os
import pickle
from collections import OrderedDict

import cv2
import numpy as np

from face_recognition.utils.imgutils import return_image
from face_recognition.utils.utils import list_files

if __name__ == '__main__':
    path = os.path.join('data', 'dataset', 'training-synthetic')
    write_path = os.path.join('data', 'processing', 'threshold', 'training-synthetic.pkl')

    files = list_files(path)
    vectors = OrderedDict()
    for file in files:
        image = return_image(file)
        if image is None:
            vectors[file] = None
        else:
            bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 20)
            vectors[file] = np.array(threshold_image)
    print(vectors)
    with open(write_path, 'wb') as fid:
        pickle.dump(vectors, fid)
        fid.close()
