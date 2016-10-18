import cv2
import numpy as np

def feature_processor():
    return create_histogram_plus_processor()

def create_histogram_plus_processor(histogram_scale=16):
    def histogram_plus_processor(image):
        h = cv2.calcHist([image], [0, 1, 2], None, [histogram_scale]*3, [0, 256]*3)
        h = h.flatten()
        # normalise
        return np.concatenate(
                (h/h.sum(),
                 [image.shape[0], image.shape[1],
                  (image.shape[0]*1.)/image.shape[1]]
        ))

    return histogram_plus_processor
