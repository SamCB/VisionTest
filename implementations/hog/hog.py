"""
Hog Visual Function just draws lines on the image
"""
import cv2
import pbcvt
import hog
import time

def initialise(*args):
    return HogVisual(*args).answer

class HogVisual:
    # def __init__(self, cell, bin, threshold):
    #     self.cell = cell
    #     self.bin = bin
    #     self.threshold = threshold

    def answer(self, img):
        result = []
        # h = hog.HogDescriptor(img, self.cell, self.bin, self.threshold)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        start = time.clock()
        h = hog.HogDescriptor(img, 8, 9, 0.0)
        h.computeHog()
        print "compute time:", time.clock() - start
        h.visualise()
        return result
