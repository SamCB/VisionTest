import cv2
import pbcvt
import hog

def feature_processor():
    return create_hog_image_processor()

def create_hog_image_processor(output_size=(64, 64), threshold=25):
    def hog_image_processor(image):
        img = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
        h = hog.HogDescriptor(img, 8, 9, threshold)
        grads = h.computeHog()
        return grads.flatten()

    return hog_image_processor