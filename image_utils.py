from __future__ import division

import cv2
import numpy as np


def image_resize_inscale(image, size):
    if image.shape[0] > image.shape[1]:
        scaled = (size[0], int(size[0]/image.shape[0]*image.shape[1]))
    else:
        scaled = (int(size[1]/image.shape[1]*image.shape[0]), size[1])

    image = cv2.resize(image, scaled, interpolation=cv2.INTER_AREA)
    square = np.zeros(size, dtype=image.dtype)
    square[0:image.shape[0], 0:image.shape[1]] = image

    return square.flatten()/256


def get_histogram(image, histogram_scale):
    h = cv2.calcHist([image], [0], None, [histogram_scale], [0, 256]).flatten()
    return h/256
