import cv2
import os

directory = os.path.dirname(__file__)
FILE = "{}/0.jpg".format(directory)


def use():
    return Fake().read


class Fake():

    def __init__(self):
        self.img = cv2.imread(FILE)
        if self.img is None:
            raise ValueError("Unable to load image '{}'".format(FILE))

    def read(self):
        return self.img.copy(), FILE
