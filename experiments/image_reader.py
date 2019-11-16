from PIL import Image


def display_image(path: str):
    img = Image.open(path)
    img.show()

# display_image('/home/vipero_7/PycharmProjects/face_recognition/data/dataset/test/0000_02176.pgm')
