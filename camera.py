import cv2

from image_utils import resize
from pylab import array, plot, show, axis, arange, figure, uint8 


def initialise(*args):
    scale = 1.
    for arg in args:
        if arg[0] == "s":
            scale = float(arg[1:])

    return CameraInput(scale).read


class CameraInput():

    def __init__(self, scale):
        self.scale = scale
        self.cam = cv2.VideoCapture(1)

    def read(self):
        result, img = self.cam.read()
        if not result:
            raise ValueError("Could not read from webcam")
        return resize(img, self.scale), None
