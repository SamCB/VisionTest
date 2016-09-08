import cv2


def use():
    return CameraInput().read


class CameraInput():

    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def read(self):
        result, img = self.cam.read()
        if not result:
            raise ValueError("Could not read from webcam")
        return img, "camera"
