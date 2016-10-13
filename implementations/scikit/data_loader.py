from __future__ import print_function

import os
import cv2

from data_set import DataSet

def load_data(directory, data_processor_module):
    file_list = os.listdir(directory)
    data_set = DataSet(data_processor_module)

    for i, f in enumerate(file_list):
        full_filename = os.path.join(directory, f)
        img = cv2.imread(full_filename)
        if img is None:
            print("WARNING: File: '{}' could not be loaded".format(full_filename))
            continue
        # The files will be of the type:
        # CLASS-source-frame-itemnumber.jpg
        label = f.split("-")[0].lower()
        data_set.add_image(img, label)

        if i % 100:
            print("{}/{} - {:5.2f}%".format(i, len(file_list), i*100./len(file_list)), end="\r")

    print("                        ", end="\r")

    if not data_set.loaded_images:
        raise ValueError("Could not load any images.")

    data_set.process()

    return data_set
