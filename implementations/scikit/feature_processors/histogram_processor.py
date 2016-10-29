import cv2
import numpy as np

def feature_processor():
    return create_histogram_processor()

def create_histogram_processor(histogram_scale=8):
    def histogram_processor(image):
        h = cv2.calcHist([image], [0, 1, 2], None, [histogram_scale]*3, [0, 256]*3)
        h = h.flatten()
        # normalise
        if not h.sum():
            import pdb;pdb.set_trace()
        return h/h.sum()

    return histogram_processor
