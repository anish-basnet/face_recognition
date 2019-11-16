"""
 -- author : Anish Basnet
 -- email : anishbasnetworld@gmail.com
 -- date : 11/16/2019
"""

import os

import numpy as np
import pandas as pd


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except NotADirectoryError:
            pass


def list_files(directory, extensions=None, shuffle=False):
    """
    Lists files in a directory
    :return: list -> gives all files path
    """

    filenames = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)

            if extensions is not None:
                if file_path.endswith(tuple(extensions)):
                    filenames.append(file_path)
            else:
                filenames.append(file_path)
    if shuffle:
        np.random.shuffle(filenames)
    return filenames


def read_from_csv(path, column_name):
    df = pd.read_csv(path)
    return list(df[column_name])


def get_all_directory(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for filename in dirs:
            files_list.append(filename)
        return files_list


def get_all_files(path):
    files_list = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            files_list.append(filename)
        return files_list


def check_directory(path):
    if os.path.exists(path):
        return True
    else:
        print(path, " doesn't exist!")
        return False

