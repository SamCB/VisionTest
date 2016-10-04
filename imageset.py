import cv2


def initialise(images):
    return ImageSetInput(images).read

class ImageSetInput():

    def __init__(self, image_names):
        self.images = []
        for image_name in image_names:
            _, image = cv2.imread(image_name)
            if image is not None:
                self.images.append(image_name, image)

        self._index = 0

    def read(self):
        try:
            return self.images[self._index]
        except IndexError:
            return None
