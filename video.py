import cv2

from image_utils import resize


def initialise(video_file, *args):
    scale, rate = 1., 1
    for arg in args:
        if arg[0] == "s":
            scale = float(arg[1:])
        elif arg[0] == "r":
            rate = int(arg[1:])

    return VideoInput(video_file, scale, rate).read


class VideoInput():

    def __init__(self, file, scale, rate):
        self.file = file
        self.scale = scale
        self.rate = rate
        self.cam = cv2.VideoCapture(file)

    def read(self):
        for _ in range(self.rate):
            result, img = self.cam.read()
            if not result:
                return None

        return resize(img, self.scale), self.file
