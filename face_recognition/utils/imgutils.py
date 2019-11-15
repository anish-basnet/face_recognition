from PIL import Image


def display_image(path):
    img = Image.open(path)
    img.show()

