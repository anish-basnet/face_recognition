"""
author : Pramod
Email :
"""

import os
import pickle

from face_recognition.utils.imgutils import get_image_array
from face_recognition.utils.utils import list_files

if __name__ == '__main__':
    path = os.path.join('data', 'dataset', 'test')
    write_path = os.path.join('data', 'processing', 'test', 'test.pkl')
    files = list_files(path)
    vectors = get_image_array(files)

    with open(write_path, 'wb') as fid:
        pickle.dump(vectors, fid)
        fid.close()
