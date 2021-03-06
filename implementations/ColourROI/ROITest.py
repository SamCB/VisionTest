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

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.regularizers import l2
from keras.regularizers import l1l2
from keras.preprocessing import image as kIm
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from scipy.ndimage import imread
from scipy import misc
from scipy import stats
from skimage import transform as tf

from skimage.measure import label as makeBlobs

import matplotlib.pyplot as plt

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
    
    # return filteredColourROI
    return filteredHarrisROI
    
def filteredColourROI(im):
    
    finalROI = []
    roi = ROIFindColour(im)
    classificationTime = 0.0
    numClass = 0
    for i in range(0, len(roi), 4):
        height = roi[i]
        width = roi[i+1]
        x = roi[i+2]
        y = roi[i+3]
        if y+height > im.shape[0]:
            y -= (y+height)-im.shape[0]
        imReg = im[y:(y+height), x:(x+width)]
        classificationStart = time.clock()
        classification = net.run(imReg)
        classificationTime += time.clock()-classificationStart
        numClass += 1
        if classification[0] > 0.9:
            region = ('ball', {'height': height, 'width': width, 'x': x, 'y': y})
            finalROI.append(region)
        # region = ('ball', {'height': height, 'width': width, 'x': x, 'y': y})
        # finalROI.append(region)
    print("Number of classifications: " + str(numClass))
    print("Total classification time: " + str(classificationTime))
    if numClass != 0:
        print("Average classification time: " + str(classificationTime/float(numClass)))
    return finalROI
    
def filteredHarrisROI(im):
    finalROI = []
    classificationTime = 0.0
    grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    numClass = 0
    for x, y, w, h in subarea_crop(retrieve_subsections(grayIm)):
        image = im[y:y+h,x:x+w]
        imShape = image.shape
        if float(imShape[1])/float(imShape[0]) < 3.0 and \
                                      float(imShape[0])/float(imShape[1]) < 3.0:
            numClass += 1
            classificationStart = time.clock()
            classification = net.run(image)
            classificationTime += time.clock()-classificationStart
            
            if classification[0] > 0.7 or classification[1] > 0.7:
                region = ('ball', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
            if classification[2] > 0.7:
                region = ('ball_part', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
            if classification[3] > 0.7:
                region = ('goal_part', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
            if classification[4] > 0.7:
                region = ('goal_post', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
            if classification[5] > 0.7 or classification[6] > 0.7 or \
                    classification[7] > 0.7 or classification[8] > 0.7 \
                    or classification[9] > 0.7  or classification[10] > 0.7:
                region = ('field', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
            if classification[11] > 0.7:
                region = ('nao', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
            if classification[12] > 0.7:
                region = ('nao_part', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
            if classification[14] > 0.7:
                region = ('penalty_spot', {'height': h, 'width': w, 'x': x, 'y': y})
                finalROI.append(region)
                
    print("Number of classifications: " + str(numClass))
    print("Total classification time: " + str(classificationTime))
    if(numClass != 0):
        print("Average classification time: " + str(classificationTime/float(numClass)))
    return finalROI
    
class Network():
    """
    Neural network for classifying ROI. Should probably be in a separate file,
    but it's just a test script.
    """

    def __init__(self, networkPath):
        """
        Prepares the conv net for use.
        
        Parameters
        ----------
        networkPath: string
            The path from which the network structure and weights may be loaded.
        """
        
        # Load and compile the network.
        self.net = self.loadModel(networkPath)
        
        # The structure of the network doesn't give the class labels, so they 
        # are hard coded here.
        self.classes = ['Ball', 'Nao']
     
    def run(self, image):
        """
        Analyses image with the network and returns the class confidence.
        
        Parameters
        ----------
        image: np.array
            The image to be analysed.
        
        Returns
        -------
        np.array
            A list of class confidence, of the form [Ball, Nao].
        """
        
        # Make the prediction.
        #start = time.clock()
        im = cv2.resize(image, (16, 16))
        im = im.transpose(2,0,1)
        im = np.array(im, dtype=float)
        im /= 255
        #print("Region prep time: " + str(time.clock()-start))
        #start = time.clock()
        prediction = (self.net.predict(np.array([im]), verbose=0))[0]
        #print("Prediction time: " + str(time.clock()-start))                                                        
        # Return the class confidence list.
        return prediction

    def loadModel(self, path):
        """
        Loads the neural network saved at path.
        
        Parameters
        ----------
        path: string
            The path to the stored model.
            
        Returns:
        Keras.models.Sequential
            The neural network, compiled and ready to run.
        """
        
        model = model_from_json(open(os.path.join(path + 'Structure.json')).read())
        model.load_weights(os.path.join(path + 'Weights.h5'))
        model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                                                              metrics=['accuracy'])
        return model

net = Network(os.path.join('implementations/ColourROI/network/'))


































