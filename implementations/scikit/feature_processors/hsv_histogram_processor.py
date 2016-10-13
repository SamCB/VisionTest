import cv2

def feature_processor():
    return create_hsv_histogram_processor()

def create_hsv_histogram_processor(hue_scale=8, sat_scale=4, val_scale=4):
    def hsv_histogram_processor(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [hue_scale, sat_scale, val_scale], [0, 180, 0, 255, 0, 255])
        return h.flatten()/256

    return hsv_histogram_processor
