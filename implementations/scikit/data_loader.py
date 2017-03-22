from __future__ import print_function

import os
import cv2
import random
import numpy as np
import struct

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

        for i, filename in enumerate(file_list):

            extension = os.path.splitext(filename)[1]
            full_filename = os.path.join(full_class_path, filename)
            if extension == ".yuv":
                with open(full_filename, 'rb') as f:
                    width = struct.unpack('i', f.read(4))[0]
                    height = struct.unpack('i', f.read(4))[0]
                    img = np.asarray([struct.unpack('B', f.read(1)) for i in range(width*height*3)], dtype=np.uint8).reshape((width, height, 3))
                    #gray = img[:,:,0]
                    #print(gray.shape)
                    #cv2.imshow(filename, gray)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
            else:
                #img = cv2.imread(full_filename)
                img = None
            #if img is None:
            #    print("WARNING: File: '{}' could not be loaded".format(full_filename))
            #    continue

            # if label != "nao_part":
            if img is not None:
                data_set.add_image(img, label)

            if i % 100:
                print("{}/{} - {:5.2f}%".format(i, len(file_list), i*100./len(file_list)), end="\r")

    print("                        ", end="\r")

    if not data_set.loaded_images:
        raise ValueError("Could not load any images.")

    data_set.process()

    return data_set
