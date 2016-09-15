import cv2

class VideoInput():

    def initialise(self):
        return self.read

    def __init__(self, file):
        self.cam = cv2.VideoCapture(file)

    def read(self):
        result, img = self.cam.read()
        if not result:
            raise ValueError("Could not read from webcam")
        return img, "camera"
