import os
from face_recognition.utils.imgutils import display_image
if __name__ == '__main__':
    path = os.path.join('data', 'dataset', 'test', '0000_02176.pgm')
    display_image(path)
