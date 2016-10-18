import cv2

def feature_processor():
    return create_grey_histogram_processor()

def create_grey_histogram_processor(histogram_scale=32):
    def grey_histogram_processor(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([gray], [0], None, [histogram_scale], [0, 256])
        h = h.flatten()
        # normalise
        return h/h.sum()

    return grey_histogram_processor
