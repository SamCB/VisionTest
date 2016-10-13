import cv2
import glob
import os
import json
import sys
import numpy as np

# Probably should be passed in is some way, but I wanted to be quick.
folderPath = '../SPQR Annotations'
jsonName = 'annotationsManual.json'

def initialise():

    # Read the JSON file.
    with open(os.path.join(folderPath, jsonName)) as jsonFile:
        annotationData = json.load(jsonFile)
        
    return ImageInput(folderPath, annotationData).read

class ImageInput():

    def __init__(self, folderPath, annotationData):
        self.folderPath = folderPath
        self.annotationData = annotationData
        self.curIndex = 0

    def read(self):
        
        # Only accept big images from SPQR.
        goodImage = False
        while not goodImage:
        
            # Loop the image feed.
            if self.curIndex >= len(self.annotationData):
                self.curIndex = 0
            
            # Read the image.
            name = self.annotationData[self.curIndex]['filename']
            img = self.readImage(folderPath, name)
            
            # Only accept 640X480 images.
            if img.shape == (480, 640, 3):
                goodImage = True
            
            # BGR to RGB.
            #img[:,:,0], img[:,:,2] = img[:,:,2], img[:,:,0]
            
            # Move to the next image.
            self.curIndex += 1
                
        return img, self.curIndex
        
    def readImage(self, folder, name):
        """Reads the image specified from the folder specified.
        
        Parameters
        ----------
        folder: string
            Path to the folder from which the image is to be read.
            
        Returns
        -------
        List
            The image read, in BGR format.
        """
        
        # Read the image.
        fileName = os.path.join(folder, name)
        im = cv2.imread(fileName)
            
        # If the file isn't an image imread returns None.
        if im is None:
            # Might be the wrong error type. Not very knowledgable about errors.
            raise ValueError("No image found at " + fileName)
        
        # Return the image list.
        return im
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
