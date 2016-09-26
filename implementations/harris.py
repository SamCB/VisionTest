"""
    This script provides a class Harris which works with the COMP3431 Vision Tester.
    It receives an image, runs a corner detection on it and puts red dots on the image where it
    believes corners are.
    It returns zero annotations
"""
import cv
import numpy as np

def initialise():
    return Harris().answer

class Harris():
    def answer(self, img):
        # make the image gray scale for corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # OpenCV corner detection
        dst = cv2.cornerHarris(gray,9,3,0.04)
        # annotate the corners on to the original image
        img[dst>0.01*dst.max()] = [0,0,255]
        return []
