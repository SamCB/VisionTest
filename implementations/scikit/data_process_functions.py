import cv2
import numpy as np

def create_histogram_processor(histogram_scale):
    def histogram_processor(image):
        h = cv2.calcHist([image], [0, 1, 2], None, [histogram_scale]*3, [0, 256]*3)
        return h.flatten()/256

    return histogram_processor

def create_hsv_histogram_processor(hue_scale, sat_scale, val_scale):
    def hsv_histogram_processor(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [hue_scale, sat_scale, val_scale], [0, 180, 0, 255, 0, 255])
        return h.flatten()/256

    return hsv_histogram_processor


def create_stretched_image_processor(output_size):
    def stretched_image_processor(image):
        img = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
        return img.flatten()/256

    return stretched_image_processor


def create_scaled_image_processor(size):
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
