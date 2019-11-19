"""
author: pramod khatiwada
email: pramodkhatiwada03@gmail.com
"""
from sklearn.metrics.pairwise import cosine_similarity


def similarity(source_vector=None, destination_vector=None):
    """
    :param source_vector: -> list
    :param destination_vector: -> list
    :return: ->
    """
    if source_vector is None or destination_vector is None:
        return None
    return cosine_similarity(source_vector, destination_vector)
