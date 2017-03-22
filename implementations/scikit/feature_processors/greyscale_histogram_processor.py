import cv2
import matplotlib.pyplot as plt

def feature_processor():
    return create_grey_histogram_processor()

def create_grey_histogram_processor(histogram_scale=16):
    def grey_histogram_processor(image):
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image[:,:,0]
        h = cv2.calcHist([gray], [0], None, [histogram_scale], [0, 256])
        h = h.flatten()
        # normalise
        return h/h.sum()

    return grey_histogram_processor
