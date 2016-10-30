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
        
    return ImageAnnotations(annotationData).read

class ImageAnnotations():

    def __init__(self, annotationData):
        self.annotationData = annotationData

    def read(self, index):
        
        # Grab the appropriate annotation.
        return self.annotationData[index]['annotations']
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
