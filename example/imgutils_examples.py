"""
author : Anish Basnet, Pankaj Bhattrai
Email : anishbasnetworld@gmail.com
"""


import os
import sys
import cv2
import numpy as np

import pickle

from face_recognition.utils.imgutils import return_image
from face_recognition.utils.utils import get_all_files

if __name__ == '__main__':
    path = os.path.join('data', 'dataset', 'test', '0000_02176.pgm')
    image = cv2.cvtColor(np.array(return_image(path)), cv2.COLOR_RGB2BGR)
    # cv2.imshow('image', image)
    # img = cv2.imread(path, 1)
    # gray_image = cv2.imread(path, 0)



    # creating pickle of image by pramod
    path_to_list_image = os.path.join('data', 'dataset', 'test')

    imageList1 = get_all_files(path_to_list_image)
    imageList = imageList1[:10]

    image_pkl_dict = {}
    for each_image in imageList:
        imageNumpyArray = np.array(each_image)
        image_pkl_dict[each_image] = imageNumpyArray

    pickle_out = open(os.path.join('data', 'processing', 'train', 'train.pickle'), "wb")
    pickle.dump(image_pkl_dict, pickle_out)
    pickle_out.close()





