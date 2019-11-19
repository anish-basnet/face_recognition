"""
- author : Anish Basnet
- Email : anishbasnetworld@gmail.com
"""
import os
import unittest

from face_recognition.utils.imgutils import return_image


class ImageUtilsTest(unittest.TestCase):
    @staticmethod
    def test_return_image():
        path = os.path.join('tests', 'data', 'ImageUtils', 'pspbrwse.jbf')
        image = return_image(path)
        assert image is None
        assert return_image(None) is None
        assert return_image('tests/data/ImageUtils/0008_003.jpg') is None
        assert return_image('tests/data/ImageUtils/0008_003.jpg') is None

    @staticmethod
    def test_get_image_array():
        pass
