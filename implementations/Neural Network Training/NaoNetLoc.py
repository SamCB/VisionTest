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

import matplotlib.pyplot as plt

# Whether data should be loaded from the cache.
useCache = 1
# Colour type: 1 - grey, 3 - BGR?
colourTypeGlobal = 3
# Whether to read test data and output predictions.
prepSubmission = False
# Dimensions of the image input to the conv net.
imgRows, imgCols = 120, 160
# Batch size to use in training.
batchSize = 128
# The number of epoch to run training over.
numEpoch = 20
# The number of folds to do in cross validation.
numFolds = 4
# Whether to subtract the mean, rather than divide by 255.
subMean = False
# The number of segments to load each train (memory management).
segmentsToLoad = 101
# The random seed to be used in generating the cross validation sets.
randomState = 51
# The location at which cache data is to be stored.
cache = 'cache'
# Any extra unique identifier for cache entries.
cacheAppend = ''
# Path to the folder containing the training images.
trainPath = os.path.join('data')
# Path to the folder containing the testing images.
testPath = os.path.join('data')
# The number of classes available.
numClasses = 1
# The valid classes to be added to the labels.
classes = ['Ball']
# The size of gaussians around points and lines respectively, scaled to the
# output image dimensions.
pointSize = 5
lineSize = 3
# The size of the localisation output image.
outSize = (15, 20)

np.seterr(all='raise') 
np.seterr(under='ignore')

def getIm(path, imgRows, imgCols, colourType=1):
    """
    Reads a single image. If the image has the wrong aspect ratio it is split
    into several images with the correct aspect ratio. Each new image begins
    half way through the last.
    
    Parameters
    ----------
    path: string
        The location from which the image should be read.
    imgRows: int
        The number of rows that the output image should have.
    imgCols: int
        The number of columsn that the output image should have.
    colourType: int
        The colour type to use. 1 is greyscale, 3 is BGR?
        
    Returns
    -------
    images: List
        A list of the images generated from this image. Note that these images 
        still point to the same data that made up the original image and may
        have overlapping shared pixels.
    starts: List
        The row, column offsets of the starting points of the sub images. Each 
        pair is itself a list.
    """

    # Load as grayscale.
    if colourType == 1:
        img = cv2.imread(path, 0)
    # Or colour.
    elif colourType == 3:
        img = cv2.imread(path)
        
    # Resize.
    
    # First check aspect ratio error direction.
    imShape = np.shape(img)
    virtError = False
    noError = True
    if float(imShape[0]/float(imShape[1])) > float(imgRows)/float(imgCols):
        virtError = True
    if abs(float(imShape[0]/float(imShape[1])) - float(imgRows)/float(imgCols))\
                                                                         > 0.01:
        noError = False

    # If there is error, create several images to return.
    if not noError:
    
        # If the extra size is on the vitical axis...
        if virtError:
            
            # Calculate the number of pixels for a sub image.
            subSize = int(float(imShape[1])*(float(imgRows)/float(imgCols)))
            
            # Create the sub images.
            images = []
            starts = []
            ended = False
            for start in range(0, imShape[0], subSize/2):
            
                # If another full image can be made, do so.
                if start + subSize < imShape[0]:
                    sub = img[start:(start+subSize), :]
                    coords = [start, 0]
                    
                # If we are at the end shift start earlier to fit the full 
                # image.
                else:
                    coords = [imShape[0]-subSize, 0]
                    sub = img[coords[0]:imShape[0], :]
                    ended = True
                    
                # Add the new image and start point.
                images.append(sub)
                starts.append(coords)
                
                # Stop after hitting the end of the image.
                if ended:
                    break
        
        # Otherwise the extra size is on the horizonal axis.
        else:
        
            # Calculate the number of pixels for a sub image.
            subSize = int(float(imShape[0])*(float(imgCols)/float(imgRows)))
            
            # Create the sub images.
            images = []
            starts = []
            ended = False
            for start in range(0, imShape[1], subSize/2):
            
                # If another full image can be made, do so.
                if start + subSize < imShape[1]:
                    sub = img[:, start:(start+subSize)]
                    coords = [0, start]
                    
                # If we are at the end shift start earlier to fit the full 
                # image.
                else:
                    coords = [0, imShape[1]-subSize]
                    sub = img[:, coords[1]:imShape[1]]
                    ended = True
                    
                # Add the new image and start point.
                images.append(sub)
                starts.append(coords)
                
                # Stop after hitting the end of the image.
                if ended:
                    break

    # Otherwise just make a one element list for consistancy.
    else:
        images = [img]
        starts = [[0,0]]
        
    # Return the results.
    return images, starts

def loadTrain(dataFolder, imgRows, imgCols, colourType=1):
    """
    Loads a set of training data from cropped images, along with folder labels.
    
    Parameters
    ----------
    dataFolder: string
        Path to the folder containing the labeled folders containing the cropped
        images.
    imgRows: int
        The number of rows that each image should be reshaped to have.
    imgCols: int
        The number of columns that each image should be reshaped to have.
    colourType: int
        The colour type to use. 1 is grey, 3 is BGR?
        
    Returns
    -------
    trainData: List
        A list of numpy.array representing all the images read.
    trainLabels: List
        A list of image labels, such that each image has one corresponding entry
        in this list. Each entry contains a list of all annotations for the
        image. Labels are in the dictionary format used by Sloth.
    """
    
    # The images and labels read.
    trainData = []
    trainLabels = []
    
    # Go through all folders, grabbing images.
    path = os.path.join(dataFolder, '*')
    folders = glob.glob(path)
    for folder in range(len(folders)):
        
        # Get the path to this folder.
        folderPath = folders[folder]
        
        # Read the folder's JSON file.
        jsonPath = glob.glob(os.path.join(folderPath, '*.json'))
        with open(jsonPath[0]) as jsonFile:
            annotationData = json.load(jsonFile)
        
        # Run through all the images.
        for imData in annotationData:
            images, starts = getIm(os.path.join(folderPath, imData['filename']),
                                                   imgRows, imgCols, colourType)
            
            # Run through the images, offsetting the annotations.
            for image in range(len(images)):
                
                # Calculate the rescale factors.
                imShape = np.shape(images[image])
                xScale = float(imgCols)/float(imShape[1])
                yScale = float(imgRows)/float(imShape[0])
                
                # Add the image labels, with appropriate offset.
                trainLabels.append([])
                for annotation in imData['annotations']:
                    
                    # Create a copy of the annotation offset to the sub image
                    # position.
                    imAnn = copy.deepcopy(annotation)
                    validAnn = True # Whether the annotation is in this image.
                    
                    # Points and rectangles.
                    if 'x' in imAnn and 'y' in imAnn:
                        imAnn['x'] -= starts[image][1]
                        imAnn['y'] -= starts[image][0]
                        
                        # Check whether the annotation is in this image.
                        if 'width' in imAnn and 'height' in imAnn:
                            if imAnn['x'] > imShape[1] or imAnn['x'] + \
                                    imAnn['width'] < 0 or imAnn['y'] > \
                                    imShape[0] or imAnn['y'] + imAnn['height'] \
                                                                            < 0:
                                validAnn = False
                        else:
                            if imAnn['x'] > imShape[1] or imAnn['x'] < 0 or \
                                    imAnn['y'] > imShape[0] or imAnn['y'] < 0:
                                validAnn = False
                    
                        # Finally rescale the annotation.
                        imAnn['x'] = int(imAnn['x'] * xScale)
                        imAnn['y'] = int(imAnn['y'] * yScale)
                        if 'width' in imAnn and 'height' in imAnn: 
                            imAnn['width'] = int(imAnn['width'] * xScale)
                            imAnn['height'] = int(imAnn['height'] * yScale)
                    
                    # Lines.
                    if 'x1' in imAnn and 'x2' in imAnn and 'y1' in imAnn and \
                            'y2' in imAnn:
                        imAnn['x1'] -= starts[image][1]
                        imAnn['y1'] -= starts[image][0]
                        imAnn['x2'] -= starts[image][1]
                        imAnn['y2'] -= starts[image][0]
                        
                        # Check whether the annotation is in the image.
                        validAnn = lineRectIntersect(imAnn['x1'], imAnn['y1'],
                                                     imAnn['x2'], imAnn['y2'],
                                                     0, 0, imShape[1], 
                                                                     imShape[0])
                                                                     
                        # Finally rescale the annotation.
                        imAnn['x1'] = int(imAnn['x1'] * xScale)
                        imAnn['y1'] = int(imAnn['y1'] * yScale)
                        imAnn['x2'] = int(imAnn['x2'] * xScale)
                        imAnn['y2'] = int(imAnn['y2'] * yScale)
                    
                    # Add the annotation if it appears in the image.
                    if validAnn:
                        trainLabels[-1].append(imAnn)
                
                # Add the image.
                img = cv2.resize(images[image], (imgCols, imgRows))
                trainData.append(img)
    
    # Return the collected images and labels.
    return trainData, trainLabels
    
def lineRectIntersect(x1, y1, x2, y2, x, y, width, height):
    """
    Determines whether there is an intersection between a line and an axis 
    aligned rectangle. NEED TO VERIFY THIS WORKS.
    
    Parameters
    ----------
    x1: int
        x coordinate of one end of the line.
    y1: int
        y coordinate of one end of the line.
    x2: int
        x coordinate of the other end of the line.
    y2: int
        y coordinate of the other end of the line.
    x: int
        x coordinate of the upper left of the rectangle.
    y: int
        y coordinate of the upper left of the rectangle.
    width: int
        The width of the rectangle.
    height: int
        The height of the rectangle.
    """
    
    # Calculate rectangle side values.
    left = x
    right = x+width
    top = y
    bottom = y+height
    
    # Check if all four corners are on the same side of the line (no intersect).
    # Technically might fail if a corner is on the line.
    a = y2-y1
    b = x1-x2
    c = x2*y1-x1*y2
    UL = a*left + b*top + c < 0
    UR = a*right + b*top + c < 0
    LR = a*right + b*bottom + c < 0
    LL = a*left + b*bottom + c < 0
    if UL == UR and UR == LR and LR == LL:
        return False
        
    # Check for the line being off to one side of the rectangle.
    if x1 < left and x2 < left:
        return False
    if x1 > right and x2 > right:
        return False
    if y1 < top and y2 < top:
        return False
    if y1 > bottom and y2 > bottom:
        return False
        
    # If the function gets this far there must be an intersection.
    return True
    
def cacheData(data, path):
    """
    Caches the read data in a pickle file, so that it can be restored more 
    quickly for the next run.
    
    Parameters
    ----------
    data: Any
        The data to cache.
    path: string
        The location at which the data should be cached.
    """

    # If the relevant directory exists, save the data.
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
        
    # Otherwise don't save.
    else:
        print('Directory doesnt exist')

def restoreData(path):
    """
    Loads data previously stored with cacheData.
    
    Parameters
    ----------
    path: string
        The location of the data to load.
        
    Returns
    -------
    Any
        The data read, or None if the directory does not exist.
    """

    # If the data exists load and return it.
    data = None
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def saveModel(model, path):
    """
    Saves the struture of a neural network model to a JSON file and the weights
    to a h5 file. 
    
    Parameters
    ----------
    model: Model
        The Keras neural network model to be saved.
    path: str
        The path to the location at which the files are to be saved. Note that 
        the end of the path should give the name of the network but not the file
        extensions, which are added automatically.
    """
    
    json_string = model.to_json()
    open(path + 'Structure.json', 'w').write(json_string)
    model.save_weights(os.path.join(path + 'Weights.h5'), overwrite=True)

def readModel(path):
    """
    Loads the model found at path.
    
    Parameters
    ----------
    path: str
        The path to the model to be loaded. Note that the end of the path should
        give the name of the network but not the file extensions, which are
        added automatically.
        
    Returns
    -------
    Model
        A Keras neural network with the structure and weights of the one found
        at path.
    """
    
    model = model_from_json(open(os.path.join(path + 'Structure.json')).read())
    model.load_weights(os.path.join(path + 'Weights.h5'))
    return model

def splitValidationSet(train, target, testPortion):
    """
    Splits off a validation set from the data given.
    
    Parameters
    ----------
    train: List
        The training data.
    target: List
        The labels corresponding to the training data.
    testPortion:
        The proportion of the data to split off.
    """

    # Use sklearn to create the validation split.
    trainData, validationData, trainLabels, validationLabels = \
                        train_test_split(train, target, test_size=testPortion, 
                                                       random_state=randomState)
    return trainData, validationData, trainLabels, validationLabels

def create_submission(predictions, testID, info):
    """
    Not yet implemented.
    """
    return None

def readAndNormalizeLabelledData(folderPath, imgRows, imgCols, colourType=1):
    """
    Reads and normalises the trainind data found in folderPath.
    
    Parameters
    ----------
    folderPath: string
        The folder containing the training images. Each class should be in its
        own labelled folder within this folder.
    imgRows: int
        The number of rows that each image should be reshaped to have.
    imgCols: int
        The number of columns that each image should be reshaped to have.
    colourType: int
        The colour type to use. 1 is grey, 3 is BGR?
        
    Returns
    -------
    trainData: List
        A numpy.array of numpy.array representing all the images read. Note that
        it is reshaped to have dimensions in the order Image, Channel, Row, 
        Column.
    trainLabels: List
        A list of image labels, such that each image has one corresponding label
        in this list. Labels are integers corresponding to the order the class
        folders are found in croppedFolder.
    weightMatrix: np.array
        An array of the sample weights to assign to balance the classes.
    """    
    
    # Create the path to the cache.
    cachePath = os.path.join(cache, 'trainR' + str(imgRows) + 'C' + 
                             str(imgCols) + 'T' + str(colourType) + 'OR' + 
                             str(outSize[0]) + 'OC' + str(outSize[1]) +
                                                           cacheAppend + '.dat')
    
    # If there is no cache load the images fresh and create a cache.
    if not os.path.isfile(cachePath) or useCache == 0:
    
        # Read the images and labels.
        trainData, trainLabels = loadTrain(folderPath, imgRows, imgCols, 
                                                                     colourType)
                                                                     
        # Convert labels to heatmap images.
        trainLabels = gausFormat(trainLabels, pointSize, lineSize, (imgRows, 
                                                     imgCols), outSize, classes)
        # Works, but might want to check it some more.
#        for index in range(len(trainData)):
#            im = overlayHeatmap(trainData[index], trainLabels[index][0])
#            plt.imshow(im)
#            plt.show()
        
        # Finally reshape to the format expected by the conv net.
        #trainLabels = trainLabels.transpose(0, 3, 1, 2)
        
        cacheData((trainData, trainLabels), cachePath)
    
    # If the cache exists, load it.
    else:
        print('Restore train from cache!')
        (trainData, trainLabels) = restoreData(cachePath)

    # Prepare images.
    trainData = np.array(trainData, dtype=float)
    trainLabels = np.array(trainLabels, dtype=float)
    if colourType == 1:
        trainData = trainData.reshape(trainData.shape[0], 1, imgRows, imgCols)
    else:
        trainData = trainData.transpose((0, 3, 1, 2))
        
    # Convert labels to a categorical crossentropy compatible format.
    trainLabels, weightMatrix = toCategorical(trainLabels)
    
    # Preparation.
    if subMean:
        trainData -= np.mean(trainData, axis = 0)
    else:
        trainData /= 255.0
    
    # Display the shape of the data read.
    print('Train shape: ' + str(trainData.shape))
    print('Label shape: ' + str(trainLabels.shape))
    print(str(trainData.shape[0]) + ' train samples.')
    
    # Return the resulting data, labels and weights.
    return trainData, trainLabels, weightMatrix

def toCategorical(trainLabels):
    """
    To be trained in Keras data needs to be shape (samples, pixels, categories), 
    where category is a probability array of size 1+numClasses. The extra class
    is none.
    
    Parameters
    ----------
    trainLabels: np.array
        The set of label images to be converted to trainable format. Should be
        shape (numClasses, outsize[0], outSize[1]).
        
    Returns
    -------
    trainLabels: np.array
        The labels in trainable format.
    weightMatrix: np.array
        A matrix of the sample weights to use if the different classifications
        are to be balanced.
    """
    
    # Transpose to have the categories last.
    trainLabels = trainLabels.transpose(0, 2, 3, 1)
    
    # Create the "None" images.
    noneIms = np.array(np.sum(trainLabels,3)<=0.0,dtype=float)
    noneShape = noneIms.shape
    noneIms = np.reshape(noneIms, (noneShape[0], noneShape[1], noneShape[2], 1))
    trainLabels = np.append(trainLabels, noneIms, 3) 
    
    # Make sure all category arrays sum to 1.
    catSum = np.sum(trainLabels,3)
    sumShape = catSum.shape
    catSum = np.reshape(catSum, (sumShape[0], sumShape[1], sumShape[2], 1))
    trainLabels *= 1.0/catSum
    
    # Finally reshape to the required dimensions.
    trainShape = trainLabels.shape
    trainLabels = trainLabels.reshape((trainShape[0], trainShape[1] * 
                                                  trainShape[2], trainShape[3]))
                                                  
    # Calculate imbalance of category appearances.
    weightTotals = np.sum(np.sum(trainLabels,0),0)
    weightMultipliers = np.sum(weightTotals)/weightTotals
    print('Category occurances: ' + str(weightTotals))
    
    # Set multiplier average to 1.
    """
    (x+y+z)/3 = 1
    x = n z
    y = m z

    (n z + m z + z)/3 = 1
    (n+m+1)z/3 = 1
    (n+m+1)z = 3
    z = 3/(n+m+1)
    """
    #n = weightMultipliers[0]/weightMultipliers[2]
    #m = weightMultipliers[1]/weightMultipliers[2]
    #weightMultipliers[2] = 3.0/(n+m+1)
    #weightMultipliers[0] = n*weightMultipliers[2]
    #weightMultipliers[1] = m*weightMultipliers[2]
    n = weightMultipliers[0]/weightMultipliers[1]
    weightMultipliers[1] = 2.0/(n+1)
    weightMultipliers[0] = n*weightMultipliers[1]
    print('Appropriate Multipliers: ' + str(weightMultipliers))
    print('Average check (should be 1): ' + str(np.mean(weightMultipliers)))
    
    # And finally create a weight matrix.
    weightMatrix = np.sum(trainLabels*weightMultipliers, 2)
    
    # Return the result.
    return trainLabels, weightMatrix
    
def gausFormat(labels, pointSize, lineSize, inSize, outSize, classes):
    """
    Converts the input labels to gaussian distribution values in an image of 
    size outSize. One image corresponds to each class, so several instances of 
    the same class are combined into the same image.
    
    Parameters
    ----------
    labels: List
        A list of image labels, such that each image has one corresponding entry
        in this list. Each entry contains a list of all annotations for the
        image. Labels are in the dictionary format used by Sloth.
    pointSize: int
        The rough radius of gaussian distributions placed around point 
        annotations.
    lineSize: int
        The rough radius of gaussian distributions placed around line 
        annotations.
    inSize: Tuple
        The size of the images input to the network.
    outSize: Tuple
        The size of the image output by the network.
    classes: List
        A list of the names of the classes that should have images created for
        them, in the order that those images should appear in the output.
        
    Returns
    -------
    List
        A list containing one entry for each input image. Each entry is itself
        a list of images, one corresponding to each class label.
    """
    
    # Run through all the images.
    outLabels = []
    for annSet in labels:
        
        # Initialise the label images.
        labelImages = []
        for label in classes:
            labelImages.append(np.zeros(outSize))
            
        # Add each annotation in this image.
        for annotation in annSet:
        
            # Check the class is valid.
            if annotation['class'] in classes:
                
                # Get the index of the annotation's image.
                index = classes.index(annotation['class'])
                
                # Determine the type of annotation, and add it.
                
                # Rectangle.
                if 'x' in annotation and 'y' in annotation and 'width' in \
                                          annotation and 'height' in annotation:
                    addGausRect(annotation, inSize, outSize, labelImages[index])
                    
                # Point.
                elif 'x' in annotation and 'y' in annotation:
                    addGausPoint(annotation, pointSize, inSize, outSize, 
                                                             labelImages[index])
                    
                # Line.
                elif 'x1' in annotation and 'x2' in annotation and 'y1' in \
                                              annotation and 'y2' in annotation:
                    addGausLine(annotation, lineSize, inSize, outSize, 
                                                             labelImages[index])
        
        # Save the generated label images.
        outLabels.append(labelImages)
    
    # Return the gaussian label set.
    return outLabels
    
def addGausRect(annotation, inSize, outSize, image):
    """
    Adds a gaussian with centre and extent based on the rectangle given in
    annotation to image. The input image IS altered.
    
    Parameters
    ----------
    annotation: Dictionary
        A Sloth style rectangle annotation.
    inSize: Tuple
        The size of the images input to the network.
    outSize: Tuple
        The size of the image output by the network.
    image: numpy.array
        The image to add a gaussian to.
    """
    
    # Calculate the scaling from input image to output image.
    scaleFactor = float(outSize[0])/float(inSize[0])
    
    # Convert annotation to its scaled sizes.
    x = int(float(annotation['x'])*scaleFactor)
    y = int(float(annotation['y'])*scaleFactor)
    width = max(1, int(float(annotation['width'])*scaleFactor))
    height = max(1, int(float(annotation['height'])*scaleFactor))
    
#    print('y before crop: ' + str(int(float(annotation['y'])*scaleFactor)))
#    print('height: ' + str(int(float(annotation['height'])*scaleFactor)))
#    print('x before crop: ' + str(int(float(annotation['x'])*scaleFactor)))
#    print('width: ' + str(int(float(annotation['width'])*scaleFactor)))
    
    # Create an appropriate gaussian.
    size = max(width, height)
    gaussian = makeGaussian(size, float(size)/2.0)
    gaussian = cv2.resize(gaussian, (width, height))
#    print('Guassian shape initial: ' + str(np.shape(gaussian)))
    # Add the gaussian to the correct part of the image.
    
    # Crop the gaussian as needed to fit within the image.
    
    # Crop left.
    if x < 0:
        gaussian = gaussian[:,x*-1:]
        x = 0
        
    # Crop top.
    if y < 0:
        gaussian = gaussian[y*-1:, :]
        y = 0
    
    # Crop right.
    diff = outSize[1] - (x+width)
    if diff < 0:
        gaussian = gaussian[:,:diff]
    
    # Crop bottom.
    diff = outSize[0] - (y+height)
    if diff < 0:
        gaussian = gaussian[:diff,:]
        
    # Now add the gaussian.
    gausShape = np.shape(gaussian)    
#    print('y: ' + str(y))
#    print('gausShape[0]: ' + str(gausShape[0]))
#    print('x: ' + str(x))
#    print('gausShape[1]: ' + str(gausShape[1]))
#    print('np.shape(image): ' + str(np.shape(image)))
    image[y:(y+gausShape[0]), x:(x+gausShape[1])] += gaussian
    
def addGausPoint(annotation, pointSize, inSize, outSize, image):
    """
    Adds a gaussian with centre and extent based on the point given in 
    annotation to image. The input image IS altered.
    
    Parameters
    ----------
    annotation: Dictionary
        A Sloth style point annotation.
    pointSize: int
        The radius of the gaussian around the point.
    inSize: Tuple
        The size of the images input to the network.
    outSize: Tuple
        The size of the image output by the network.
    image: numpy.array
        The image to add a gaussian to.
    """
    
    # Calculate the scaling from input image to output image.
    scaleFactor = float(outSize[0])/float(inSize[0])
    
    # Convert annotation to its scaled sizes.
    x = int(float(annotation['x'])*scaleFactor)
    y = int(float(annotation['y'])*scaleFactor)
    
    # Create an appropriate gaussian.
    gaussian = makeGaussian(pointSize*2, pointSize)
    
    # Add the gaussian to the correct part of the image.
    
    # Crop the gaussian as needed to fit within the image.
    
    # Crop left.
    diff = x-pointSize
    if diff < 0:
        gaussian = gaussian[:,x*-1:]
    
    # Crop top.
    diff = y-pointSize
    if diff < 0:
        gaussian = gaussian[y*-1:,:]
    
    # Crop right.
    diff = outSize[1] - (x+pointSize)
    if diff < 0:
        gaussian = gaussian[:,:diff]
        
    # Crop bottom.
    diff = outSize[0] - (y+pointSize)
    if diff < 0:
        gaussian = gaussian[:diff,:]
        
    # Now add the gaussian.
    x = max(0, x-pointSize)
    y = max(0, y-pointSize)
    gausShape = np.shape(gaussian)
    image[y:(y+gausShape[0]), x:(x+gausShape[1])] += gaussian
    
def addGausLine(annotation, lineSize, inSize, outSize, image):
    """
    Adds a gaussian with centre and extent based on the rectangle given in
    annotation to image. The input image IS altered. NOT YET IMPLEMENTED.
    
    Parameters
    ----------
    annotation: Dictionary
        A Sloth style line annotation.
    pointSize: int
        The radius of the gaussian around the point.
    inSize: Tuple
        The size of the images input to the network.
    outSize: Tuple
        The size of the image output by the network.
    image: numpy.array
        The image to add a gaussian to.
    """
    
    # Calculate the scaling from input image to output image.
    scaleFactor = float(outSize[0])/float(inSize[0])
    
    # Convert annotation to its scaled sizes.
    x1 = int(float(annotation['x1'])*scaleFactor)
    y1 = int(float(annotation['y1'])*scaleFactor)
    x2 = int(float(annotation['x2'])*scaleFactor)
    y2 = int(float(annotation['y2'])*scaleFactor)
    
    # Create an appropriate gaussian.
    gaussian = makeGaussian(lineSize*2, lineSize)
    
    # Add the gaussian to the correct part of the image.
    
    # Crop the gaussian as needed to fit within the image.
    diff = outSize[1] - (x+pointSize)
    if diff < 0:
        gaussian = gaussian[:,:diff]
    diff = outSize[0] - (y+pointSize)
    if diff < 0:
        gaussian = gaussian[:diff,:]
        
    # Now add the gaussian.
    gausShape = np.shape(gaussian)
    image[(y-pointSize):(y+pointSize), (x-pointSize):(x+pointSize)] += gaussian
    
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    Credit: http://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussi
                                                                  an-with-python

    Parameters
    ----------
    size: int
        The length of a side of the square.
    fwhm: int
        Full-width-half-maximum, which can be thought of as an effective radius.
    
    Returns
    -------
    np.array:
        An array containing the approximate values of a 2D gaussian.
    """

    # Just arrays of all x and y values.
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    # Determine the centre point values to use.
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    # Calculate gaussian values. Hopefully uses C++ vector math implementation.
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
def makeLineGaussian(size, x1, y1, x2, y2, fwhm = 3):
    """ Make a gaussian value set based on distance around a line.
    Based on: http://stackoverflow.com/questions/7687679/how-to-generate-2d-gaus
                                                                sian-with-python
    NOT YET IMPLEMENTED.

    Parameters
    ----------
    size: int
        The shape of the target annotation box.
    fwhm: int
        Full-width-half-maximum, which can be thought of as an effective radius.
    
    Returns
    -------
    np.array:
        An array containing the approximate values of a 2D gaussian.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    x0=1
    y0=2

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def dictToList(d):
    """
    Creates a list from the given dictionary.
    
    Parameters
    ----------
    d: Dictionary
        The dictionary to convert.
    
    Returns
    -------
    List
        The dictionary items as a list with entries of the form [key, value].
    """
    
    newList = [value for key, value in d.items()]
    return newList


def mergeSeveralFoldsMean(data):
    """
    Creates an ensemble using the mean results of several CV folds.
    
    Parameters
    ----------
    data: List
        The data from which the mean is to be found.
        
    Returns
    -------
    List
        The predictions made by the ensemble for each image.
    """

    return np.mean(data, 0).tolist()


def mergeSeveralFoldsGeom(data):
    """
    Creates an ensemble using the geometric mean results of several CV folds.
    
    Parameters
    ----------
    data: List
        The data from which the mean is to be found.
        
    Returns
    -------
    List
        The predictions made by the ensemble for each image.
    """
    
    return stats.mstats.gmean(data).tolist()

def createModel(imgRows, imgCols, colourType=1):
    """
    Creates a convolutional neural network for training. The parameters of that
    network are intended to be tweaked through alterations to this function.
    
    Parameters
    ----------
    imgRows: int
        The number of rows that each image is expected to have.
    imgCols: int
        The number of columns that each image is expected to have.
    colourType: int
        The number of colour channels each image is expected to have.
        
    Returns
    -------
    Model
        A Keras model containing a convolutional neural network ready for 
        training.
    """
    
    # The number of convolutional filters to use in each layer.
    numFilters = [4, 8, 16, 32, numClasses+1]
    # The size of the pooling areas for max pooling.
    sizePool = [2, 2, 2, 2]
    # The convolution kernel size.
    sizeConv = [3, 3, 3, 3, 3]
    # The number of neurons in the dense layer.
    denseNeurons = 64
    # The dropout after each max pool.
    poolDropout = 0.0
    # The dropout after each dense layer.
    denseDropout = 0.0
    # The type of initialisation to use.
    initialisation = 'he_normal'
    # The type of activation function to used on non classifcation layers.
    activ = 'relu'
    
    # Initialise the model.
    model = Sequential()
    
    # Set up conv layer 1.
    model.add(ZeroPadding2D((1, 1), input_shape=(colourType, imgRows, imgCols)))
    model.add(Convolution2D(numFilters[0], sizeConv[0], sizeConv[0], 
                                         init=initialisation, activation=activ))
    
    # Set up pooling layer 1.
    model.add(MaxPooling2D(pool_size=(sizePool[0], sizePool[0])))
    model.add(Dropout(poolDropout))
    
    # Set up conv layer 2.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(numFilters[1], sizeConv[1], sizeConv[1], 
                                         init=initialisation, activation=activ))
    
    # Set up pooling layer 2.
    model.add(MaxPooling2D(pool_size=(sizePool[1], sizePool[1])))
    model.add(Dropout(poolDropout))
    
    # Set up conv layer 3.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(numFilters[2], sizeConv[2], sizeConv[2], 
                                         init=initialisation, activation=activ))
    
    # Set up pooling layer 3.
    model.add(MaxPooling2D(pool_size=(sizePool[2], sizePool[2])))
    model.add(Dropout(poolDropout))
    
    # Set up conv layer 4.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(numFilters[3], sizeConv[3], sizeConv[3], 
                                         init=initialisation, activation=activ))
    
    # Output shape: (numFilters[3], 15, 20)
    
    # Set up pooling layer 4.
    #model.add(MaxPooling2D(pool_size=(sizePool[3], sizePool[3])))
    #model.add(Dropout(denseDropout))

    # Set up the output conv layer.
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(numFilters[4], sizeConv[4], sizeConv[4], 
                                                           init=initialisation))
    # Output shape: (numClasses,15,20)
    
    # Set up output dense layer.
    #model.add(Flatten())
    #model.add(Dense((numClasses+1)*outSize[0]*outSize[1], init=initialisation, 
    #                                                      activation='softmax'))
    
    model.add(Permute((2,3,1)))
    model.add(Reshape((outSize[0]*outSize[1], numClasses+1)))
    model.add(Activation('softmax'))

    # Default optimiser is 'adadelta'.
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', 
                            metrics=['accuracy'], sample_weight_mode='temporal')
    return model


def runCrossValidation(nFolds=10, loadModel = False, startFold=0):
    """
    Creates an trains a convolutional neural network to solve the classification
    task. Validataion is done through cross validation.
    
    Parameters
    ----------
    nFolds: int
        The number of cross validation folds to perform.
    loadModel: bool
        Whether the cached model should be loaded to continue training.
    """
    # https://github.com/fchollet/keras/issues/1169
    # If we are loading a model restore its predictions.
    if loadModel:
        fullTrainPred = restoreData(os.path.join('cache', 'savedModel', 
                                                               'fullTrainPred'))
        fullTestPred = restoreData(os.path.join('cache', 'savedModel', 
                                                                'fullTestPred'))
    else:
        fullTrainPred = dict()
        fullTestPred = []
    
    # Load all the images.
    trainDataFull, trainLabelsFull, weightMatrixFull = \
                                                   readAndNormalizeLabelledData(
                                                      trainPath, imgRows, 
                                                      imgCols, colourTypeGlobal)
    """
    trainLabelsFullTemp = []
    for label in range(len(trainLabelsFull)):
        trainLabelsFullTemp.append(trainLabelsFull[label] + 0.00001)
    trainLabelsFull = np.asarray(trainLabelsFullTemp)
    print(trainLabelsFull[0])
    
    print(np.isnan(np.sum(trainDataFull)))
    print(np.isnan(np.sum(trainLabelsFull)))
    """
    
    # If needed read the test data.
    if prepSubmission:
        testData, testLabels = readAndNormalizeLabelledData(testPath, imgRows, 
                                                      imgCols, colourTypeGlobal)
    
    # Create a set of indexes for cross validation.
    kf = KFold(len(trainLabelsFull), n_folds=nFolds, shuffle=True, 
                                                       random_state=randomState)
    
    # Create a record of the scores for each fold.
    scoreRecord = []
    
    # Perform each fold.
    numFold = 0
    for trainIndex, validationIndex in kf:
        
        # Skip folds that were completed on previous runs.
        if numFold >= startFold:

            # Report progress.
            print('Start KFold number {} from {}'.format(numFold, nFolds))
                    
            # Create the model.
            model = createModel(imgRows, imgCols, colourTypeGlobal)
            
            # Partition the drivers.
            trainData = trainDataFull[trainIndex]
            trainLabels = trainLabelsFull[trainIndex]
            trainWeightMatrix = weightMatrixFull[trainIndex]
            validationData = trainDataFull[validationIndex]
            validationLabels = trainLabelsFull[validationIndex]
            validationWeightMatrix = weightMatrixFull[validationIndex]
            
            """
            # Create morphed images.
            print('Creating transformed images...')
            newX = []
            newY = []
            transforms = [tf.SimilarityTransform(scale=1, rotation=math.pi/6, 
                          translation = (0,0)), tf.SimilarityTransform(scale=1, 
                                      rotation=-math.pi/6, translation = (0,0))]
            for index in range(len(trainData)):
                im = trainData[index]
                label = trainLabels[index]
                
                for transform in transforms:
                    newIm = []
                    #for i in range(3):
                    #    newIm.append(transform(im[i]))
                    newX.append(tf.warp(im, transform))
                    newY.append(label)
            
            # Add morphed images.
            trainData = np.concatenate((trainData, newX))
            trainLabels = np.concatenate((trainLabels, newY))
            """
            """
            # Create data generator.
            datagen = kIm.ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=20,
                width_shift_range=0.0,
                height_shift_range=0.0,
                horizontal_flip=False)
                
            # Should only really be needed if featurewise_center or 
            # featurewise_std_normalization is on.
            datagen.fit(trainData)
            """

            # Make the fit.
            print("Starting fit...")
            
            """
            # Make the fit.
            save_model(model, 'fold_' + str(numFold))
            model.fit(trainData, trainLabels, batch_size=batchSize, 
                      nb_epoch=numEpoch, verbose=1, 
                      validation_data=(validationData, validationLabels),
                      callbacks=[ModelCheckpoint(os.path.join('cache', 
                      'savedModel', 'fold_'+str(numFold)+'_model_weights.h5'),
                                               save_best_only=True, verbose=1)])
            """
            
            # Do epoch separately.
            score = (1000000.0, 0.0)
            predictionsValid = []
            for epoch in range(numEpoch):
            
                # Display the epoch.
                print('Epoch ' + str(epoch+1))
            
                # The different ways of making the fit.
                #model.fit_generator(datagen.flow(trainData, trainLabels, 
                #                    batch_size=batchSize), 
                #                    samples_per_epoch=len(trainData), 
                #                    verbose=1, validation_data=(validationData,
                #                                              validationLabels))
                #model.fit(trainData, trainLabels, batch_size=batchSize, 
                #          verbose=1, validation_data=(validationData, 
                #                                              validationLabels))
                model.fit(trainData, trainLabels, batch_size=batchSize, 
                         nb_epoch=1, verbose=1, sample_weight=trainWeightMatrix)
                
                # Validate the model.
                tempScore = model.evaluate(validationData, validationLabels,
                                           batch_size=batchSize, verbose=1,
                                           sample_weight=validationWeightMatrix)
                print('Validation log_loss: ' + str(tempScore[0]))
                print('Validation accuracy: ' + str(tempScore[1]))
                       
                tempPredictions = model.predict(validationData, 
                                                batch_size=batchSize, verbose=1)

                # Keep the best model.
                if tempScore[0] < score[0]:
                    score = tempScore
                    saveModel(model, os.path.join('savedFolds', 'fold_' + 
                                                                  str(numFold)))
                    predictionsValid = tempPredictions
            
            # Restore the best model.
            model = readModel( os.path.join('savedFolds', 'fold_' + 
                                                                  str(numFold)))
            model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                            metrics=['accuracy'], sample_weight_mode='temporal')
            
            # Display best model results.
            im = overlayHeatmap(validationData[0].transpose(1,2,0)*255.0, 
                                      predictionsValid[0][:,0].reshape((15,20)))
            plt.figure()
            plt.imshow(im)
            plt.figure()
            print('Highest ball certainty: ' + 
                         str(np.max(predictionsValid[0][:,0].reshape((15,20)))))
            plt.imshow(predictionsValid[0][:,0].reshape((15,20))*255, 
                                                                 cmap='Greys_r')
            print('Best score log_loss: ', score[0])
            print('Best score accuracy: ', score[1])
            
            # Save the score.
            scoreRecord.append(score)
            
            # Store valid predictions
            for i in range(len(validationIndex)):
                fullTrainPred[validationIndex[i]] = predictionsValid[i]
                
            # Store test predictions
            if prepSubmission:
                testPredictions = model.predict(test_data, 
                                                batch_size=batchSize, verbose=1)
                fullTestPred.append(test_prediction)
    
            # Save the data.
            cacheData(fullTrainPred, os.path.join('cache', 'savedModel', 
                                                               'fullTrainPred'))
            cacheData(fullTestPred, os.path.join('cache', 'savedModel', 
                                                                'fullTestPred'))
        
        # If skipping, say so.
        else:
            print('Skipped fold ' + str(numFold))
        
        # Update the fold counter.
        numFold += 1
    
    # Calculate overall log loss of the system.
    scoreRecord = np.asarray(scoreRecord)
    scoreAv = np.mean(scoreRecord,0)
    scoreWorst = [max(scoreRecord[:,0]), min(scoreRecord[:,1])]
    print('Average log loss: {} average accuracy: {} '.format(scoreAv[0], 
          scoreAv[1]) + 'worst log loss: {} worst accuracy: {} '.format(
          scoreWorst[0], scoreWorst[1]) + 'rows: {} cols: {} nFolds: {}'.format(
                      imgRows, imgCols, nFolds) + ' epoch: {}'.format(numEpoch))
    infoString = 'aLoss_' + str(scoreAv[0]) \
                    + '_aAcc_' + str(scoreAv[1]) \
                    + '_wLoss_' + str(scoreWorst[0]) \
                    + '_wAcc_' + str(scoreWorst[1]) \
                    + '_r_' + str(imgRows) \
                    + '_c_' + str(imgCols) \
                    + '_folds_' + str(nFolds) \
                    + '_ep_' + str(numEpoch)
                    
    # Show the sample images.
    plt.show()

    # Create a final submission if appropriate.
    if prepSubmission:
	    test_res = mergeSeveralFoldsMean(fullTestPred, nFolds)
	    create_submission(test_res, testID, infoString)
        
def overlayHeatmap(originImage, heatmap):
    """
    Creates a copy of image, overlain with heatmap.
    
    Parameters
    ----------
    originImage: np.array
        A standard BGR image.
    heatmap: np.array
        A single channel array of the same aspect ratio as image. It will be
        scaled to the same size as image, converted to "heatmap" colours and
        overlain on image.
        
    Returns
    -------
    np.array
        A copy of image with heatmap overlain on it.
    """
    
    # Create a copy of the origin image for annotation.
    image = originImage.astype(float)
    
    # Rescale heatmap if neccesary.
    imageShape = np.shape(image)
    heatmap = cv2.resize(heatmap, (imageShape[1], imageShape[0]))
    
    # Add the heatmap.
    image[:,:,0] += (heatmap*512)
    image /= 2
    image[:,:,0] = np.minimum(image[:,:,0], np.ones(imageShape[0:2])*255)
    
    # Return the overlain image.
    return image.astype(np.uint8)
   
runCrossValidation(numFolds, False, 0)










































