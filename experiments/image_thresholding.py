import os

import cv2
import numpy as np

from face_recognition.utils.imgutils import return_image

if __name__ == '__main__':
    root_path = os.path.join('data', 'dataset', 'test')
    names = os.listdir(root_path)
    Image_dict = {}
    for i in names:
        path = os.path.join('data', 'dataset', 'test')
        path += ('/{}'.format(i))

        image = cv2.cvtColor(np.array(return_image(path)), cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
        # cv2.imshow('adaptive', thres)
        imageMat = [thres]
        Image_dict[path] = np.array(imageMat)
        cv2.waitKey(0)
    print(Image_dict)


