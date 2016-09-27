import cv2


def initialise(video_file):
    return VideoInput(video_file).read


class VideoInput():

    def __init__(self, file):
        self.file = file
        self.cam = cv2.VideoCapture(file)

    def read(self):
        result, img = self.cam.read()
        if not result:
            raise ValueError("Could not read from file")
        return img, self.file
