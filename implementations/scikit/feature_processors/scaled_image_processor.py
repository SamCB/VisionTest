import cv2
import numpy as np

def feature_processor():
    return create_scaled_image_processor()

def create_scaled_image_processor(size=(8, 8)):
    def scaled_image_processor(image):
        if image.shape[0] > image.shape[1]:
            scaled = (size[0], int(size[0]/image.shape[0]*image.shape[1]))
        else:
            scaled = (int(size[1]/image.shape[1]*image.shape[0]), size[1])

        # Make sure none of the dimensions we ask for is 0
        scaled = tuple(max(i, 1) for i in scaled)

        image = cv2.resize(image, scaled, interpolation=cv2.INTER_AREA)
        square = np.zeros_like(image)
        square[0:image.shape[0], 0:image.shape[1]] = image

        return square.flatten()/256

    return scaled_image_processor
