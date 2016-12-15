from __future__ import print_function

import os
import cv2
import random

from dataset import DataSet

def load_data(directory, data_processor_module):
    class_list = os.listdir(directory)
    data_set = DataSet(data_processor_module)

    for index, folder in enumerate(class_list):

        full_class_path = os.path.join(directory, folder)
        if not os.path.isdir(full_class_path):
            continue
        file_list = os.listdir(full_class_path)
        label = folder

        for i, f in enumerate(file_list):

            full_filename = os.path.join(full_class_path, f)
            img = cv2.imread(full_filename)
            if img is None:
                print("WARNING: File: '{}' could not be loaded".format(full_filename))
                continue

            # if label != "nao_part":
            data_set.add_image(img, label)

            if i % 100:
                print("{}/{} - {:5.2f}%".format(i, len(file_list), i*100./len(file_list)), end="\r")

    print("                        ", end="\r")

    if not data_set.loaded_images:
        raise ValueError("Could not load any images.")

    data_set.process()

    return data_set
















































