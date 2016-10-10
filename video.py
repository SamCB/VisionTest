import cv2

from image_utils import resize


def initialise(video_file, scale=1):
    try:
        scale = float(scale)
    except ValueError:
        scale = 1.
    return VideoInput(video_file, scale).read


class VideoInput():

    def __init__(self, file, scale):
        self.file = file
        self.scale = scale
        self.cam = cv2.VideoCapture(file)

    def read(self):
        result, img = self.cam.read()
        if not result:
            return None
        return resize(img, self.scale), self.file
