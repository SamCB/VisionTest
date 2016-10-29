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

        # maxIntensity = 255.0 # depends on dtype of image data

        # # Parameters for manipulating image data
        # phi = 1
        # theta = 1

        # # Decrease intensity such that
        # # dark pixels become much darker, 
        # # bright pixels become slightly dark 
        # newImage = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
        # newImage = array(newImage,dtype=uint8)

        # return resize(newImage, self.scale), "camera"
        return resize(img, self.scale), "camera"
