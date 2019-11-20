"""
author: pramod khatiwada
email: pramodkhatiwada03@gmail.com
"""
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity


def similarity(source_vector=None, destination_vector=None):
    """
    This method provides the cosine similarity between two vectors.
    :param source_vector: -> list
    :param destination_vector: -> list
    :return: ->
    """
    if source_vector is None or destination_vector is None:
        return None
    return cosine_similarity(source_vector, destination_vector)


def image_structural_similarity(source_image=None, destination_image=None):
    """
    :param source_image:
    :param destination_image:
    :return:
    """
    if source_image is None or destination_image is None:
        return None
    return structural_similarity(source_image, destination_image)
