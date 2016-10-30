# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import statistics
import h5py
import copy
import json
import time
import random

import sys
sys.path.insert(0, './implementations/ColourROI')
# sys.path.insert(0, './implementations/ColourROI/CPP/ctypes')
sys.path.insert(0, './example_implementations')
sys.path.insert(0, './crop_functions')
# from ROIFindColourCPP import ROIFindColour
# from ROIFindColour import ROIFindColour
from colourROI import ROIFindColour
from harris_crop import retrieve_subsections
from subarea_crop import subarea_crop
from naive_harris_function import initialise as naive_harris_initialise
import pbcvt

"""
Run with (specific to my file layout):
python main.py ./implementations/ColourROI/ROITest.py SPQRRead.py SPQRAnnotations.py
"""

def initialise(*args):
    """
    Creates the network and returns the annotation function.
    
    Returns
    -------
    function
        A function that, when called, returns the annotations for the image
        passed.
    """
    
    # return ColourROI
    return HarrisROI
    
def ColourROI(im):
    
    finalROI = []
    roi = ROIFindColour(im)
    for i in range(0, len(roi), 4):
        h = roi[i]
        w = roi[i+1]
        x = roi[i+2]
        y = roi[i+3]
        # r = random.randrange(0, 5)
        r = 2.5
        if r < 1:
            region = ('ball', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        elif r < 2:
            region = ('ball_part', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        elif r < 3:
            region = ('nao', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        elif r < 4:
            region = ('nao_part', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        else:
            region = ('nothing', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
    return finalROI, 0.0#, 0.0, 1
    
def HarrisROI(im):
    finalROI = []
    classificationTime = 0.0
    grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    numClass = 0
    for x, y, w, h, t, l in subarea_crop(retrieve_subsections(grayIm)):
        # r = random.randrange(0, 5)
        r = 2.5
        if r < 1:
            region = ('ball', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        elif r < 2:
            region = ('ball_part', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        elif r < 3:
            region = ('nao', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        elif r < 4:
            region = ('nao_part', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)
        else:
            region = ('nothing', {'height': h, 'width': w, 'x': x, 'y': y})
            finalROI.append(region)

    return finalROI, 0.0#, 0.0, 1
    
