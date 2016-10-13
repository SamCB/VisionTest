import cv2

def feature_processor():
    return create_stretched_image_processor()

def create_stretched_image_processor(output_size=(8, 8)):
    def stretched_image_processor(image):
        img = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
        return img.flatten()/256

    return stretched_image_processor
