import cv2

from image_utils import resize


def initialise(*args):
    scale = 1.
    for arg in args:
        if arg[0] == "s":
            scale = float(arg[1:])

    return CameraInput(scale).read


class CameraInput():

    def __init__(self, scale):
        self.scale = scale
        self.cam = cv2.VideoCapture(0)

    def read(self):
        result, img = self.cam.read()
        if not result:
            raise ValueError("Could not read from webcam")
        return resize(img, self.scale), "camera"
